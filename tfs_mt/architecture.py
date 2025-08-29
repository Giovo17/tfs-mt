import math

import torch
import torch.nn as nn


class DModelNotDivisibleByNumHeads(Exception):
    def __init__(self, d_model, num_heads):
        msg = f"d_model is not divisible by the number of heads, got d_model = {d_model} and num_heads = {num_heads}."
        super().__init__(msg)


class MultiHeadAttention(nn.Module):
    """MultiHead Attention for Transformer encoder and decoder. It handles both self and cross attention operations.

    Following the implementation described in *Speech and Language Processing* by *Daniel Jurafsky* [[link](https://web.stanford.edu/~jurafsky/slp3/)].

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of attention heads.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_prob: float = 0.1):
        super().__init__()

        # TODO Remove
        # if d_model % num_heads != 0:
        #     raise DModelNotDivisibleByNumHeads(d_model, num_heads)

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = int(d_model / num_heads)  # Query, key and value embeddings dimension. d_k = d_v = d_head
        self.dropout_prob = dropout_prob  # TODO Use it

        # Learnable projection matrices. Bias term is omitted since they are used as projections matrices.
        # Every head should have its projection matrix, but rather considering a set of QKV matrices for each head,
        # here 3 bigger matrices are considered. The following example involves the query projection matrix W_Q but the reasoning applies to all of them.
        # Considering D = d_model, d_k = D_v = d_head and A = num_heads.
        # W_Q is a DxD matrix and each W_Q_i (query projection matrix for i-th head) should be a DxD_k matrix.
        # W_Q can be reshaped as a DxAxD_k matrix since A*d_k = D due to initial assertion. (in practice the output projection will be reshaped as mentioned)
        # This way we can properly take advantage of GPU parallelization thanks to torch broadcasting,
        # instead of executing one projection operation at a time for each head in a loop.
        self.W_Q = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.W_K = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.W_V = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.W_O = nn.Linear(num_heads * self.d_head, d_model, bias=False)  # Output projection

        self.scaling_factor = math.sqrt(self.d_head)  # To avoid computing it every time attention method is called

    def forward(
        self,
        x_query: torch.Tensor,
        x_key: torch.Tensor,
        x_value: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """MultiHead attention.

        Args:
            x_query (torch.Tensor): Matrix of input embeddings of shape [B, S, D], where B is the batch size, S is the sequence length and D is d_model.
            x_key (torch.Tensor): _description_
            x_value (torch.Tensor): _description_
            attention_mask (torch.Tensor | None, optional): Attention mask to avoid computing attention to padding tokens. It's also used to apply causal masking in decoder self attention. Defaults to None.

        Returns:
            torch.Tensor: Processed output tensor.
        """

        batch_size = x_query.shape[0]

        # W_Q(x)          [B, S, D]
        # After reshape   [B, S, A, d_k]
        # After transpose [B, A, S, _k] where A is num_heads and S is the sequence length
        query_matrices = self.W_Q(x_query).reshape(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        key_matrices = self.W_K(x_key).reshape(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        value_matrices = self.W_V(x_value).reshape(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # Concatenated heads outputs, shape [B, A*d_k]
        attn_output = self.attention(query_matrices, key_matrices, value_matrices, attention_mask=attention_mask)

        # Reshape back from [B, A, S, d_k] to [B, S, D]
        attn_reshaped = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_head)

        # Attention scores, shape [B, S, D]. Combine heads outputs into a single D-dimensional output.
        return self.W_O(attn_reshaped)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """Implements attention as follows:

        $$
        Attention(Q,K,V) = softmax ( \\frac{QK^T}{\\sqrt{d_{model}}} ) * V
        $$

        Args:
            query (_type_): _description_
            key (_type_): _description_
            value (_type_): _description_
            attention_mask (torch.BoolTensor | None, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: Attention matrix.
        """

        QKt = torch.matmul(query, key.transpose(-2, -1)) / self.scaling_factor

        print(QKt)
        if attention_mask is not None:
            print("Applying mask")
            QKt.masked_fill_(not attention_mask, float("-inf"))
        print(QKt)

        QKt_norm = torch.softmax(QKt, dim=-1)

        return torch.matmul(QKt_norm, value)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # scale
        self.beta = nn.Parameter(torch.zeros(d_model))  # shift
        self.eps = eps  # Avoids ZeroDivisionError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=-1, keepdim=True)
        # There's no interest in applying Bessel's correction.
        # It is usually done to make sure var is an unbiased estimator of the sample variance,
        # but here the main goal is to make the output have sum of squares equal to 1
        var = torch.var(x, dim=-1, correction=0, keepdim=True)

        z = (x - mean) / torch.sqrt(var + self.eps)
        return z * self.gamma + self.beta


class EncoderBlock(nn.Module):
    """Transformer Encoder block.

    Using prenorm approach which stabilizes training and allows the model to converge faster,
    according to *On Layer Normalization in the Transformer Architecture* [[link](https://arxiv.org/abs/2002.04745)]

    Args:
        d_model (int): _description_
        num_heads (int): _description_
        d_ff (int): _description_
        dropout_prob (float, optional): _description_. Defaults to 0.1.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_prob: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_prob)
        self.feedforward = FeedForward(d_model, d_ff, dropout_prob)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        t1 = self.layer_norm1(x)
        t2 = self.self_attention(x_query=t1, x_key=t1, x_value=t1, attention_mask=attention_mask)
        t3 = t2 + x
        t4 = self.layer_norm2(t3)
        t5 = self.feedforward(t4)
        h = t5 + t3

        return h
