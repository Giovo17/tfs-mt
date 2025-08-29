import math

import torch
import torch.nn as nn

from .embeddings import Embedding, SinusoidalPositionalEncoding


class MissingArgumentsError(Exception):
    def __init__(self, emb_type):
        msg_init = "Source" if emb_type == "src" else "Target"
        msg = f"{msg_init} embeddings initialization from pretrained has been requested, but {emb_type}_emb_pretrained_type or {emb_type}_emb_pretrained_path have not been provided."
        super().__init__(msg)


class MissingArgumentsGloVeError(Exception):
    def __init__(self, emb_type):
        msg_init = "Source" if emb_type == "src" else "Target"
        msg = f"{msg_init} embeddings initialization from GloVe pretrained has been requested, but {emb_type}_tokenizer has not been provided."
        super().__init__(msg)


class LanguageDirectionInvalidFormat(Exception):
    def __init__(self, language_direction):
        msg = (
            f"Invalid language direction format: '{language_direction}'. "
            "Expected format is '<src_lang>-<tgt_lang>', e.g., 'en-it'."
        )
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

        print(QKt)  # debug
        if attention_mask is not None:
            print("Applying mask")  # debug
            QKt.masked_fill_(attention_mask == False, float("-inf"))
        print(QKt)  # debug

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


class DecoderBlock(nn.Module):
    """Transformer Decoder block.

    Using prenorm approach as in EncoderBlock.

    Args:
        d_model (int): _description_
        num_heads (int): _description_
        d_ff (int): _description_
        dropout_prob (float, optional): _description_. Defaults to 0.1.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_prob: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_prob)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout_prob)
        self.feedforward = FeedForward(d_model, d_ff, dropout_prob)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        t1 = self.layer_norm1(x)
        t2 = self.self_attention(x_query=t1, x_key=t1, x_value=t1, attention_mask=attention_mask)
        t3 = t2 + x
        t4 = self.layer_norm2(t3)
        # TODO check mask
        t5 = self.cross_attention(
            x_query=t4, x_key=encoder_output, x_value=encoder_output, attention_mask=attention_mask
        )
        t6 = t5 + t3
        t7 = self.layer_norm3(t6)
        t8 = self.feedforward(t7)
        h = t8 + t6

        return h


class Transformer(nn.Module):
    """Transformer model.

    Using Language Model head to map decoder output representation to tokens in vocabulary.

    Args:

    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        num_encoder_blocks: int = 6,
        num_decoder_blocks: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout_prob: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        # Source embedding init
        if kwargs.get("src_emb_from_pretrained"):
            if "src_emb_pretrained_type" not in kwargs or "src_emb_pretrained_path" not in kwargs:
                raise MissingArgumentsError("src")
            if "src_tokenizer" not in kwargs and kwargs["src_emb_pretrained_type"] == "GloVe":
                raise MissingArgumentsGloVeError("src")
            self.src_embeddings = Embedding(
                src_vocab_size,
                d_model,
                from_pretrained=True,
                pretrained_emb_type=kwargs["src_emb_pretrained_type"],
                pretrained_emb_path=kwargs["src_emb_pretrained_path"],
                tokenizer=kwargs["src_tokenizer"],
            )
        else:
            self.src_embeddings = Embedding(src_vocab_size, d_model)

        # Target embedding init
        if kwargs.get("tgt_emb_from_pretrained"):
            if "tgt_emb_pretrained_type" not in kwargs or "tgt_emb_pretrained_path" not in kwargs:
                raise MissingArgumentsError("tgt")
            if "tgt_tokenizer" not in kwargs and kwargs["tgt_emb_pretrained_type"] == "GloVe":
                raise MissingArgumentsGloVeError("tgt")
            self.tgt_embeddings = Embedding(
                tgt_vocab_size,
                d_model,
                from_pretrained=True,
                pretrained_emb_type=kwargs["tgt_emb_pretrained_type"],
                pretrained_emb_path=kwargs["tgt_emb_pretrained_path"],
                tokenizer=kwargs["tgt_tokenizer"],
            )
        else:
            self.tgt_embeddings = Embedding(tgt_vocab_size, d_model)

        self.src_pos_embeddings = SinusoidalPositionalEncoding(d_model)
        self.tgt_pos_embeddings = SinusoidalPositionalEncoding(d_model)

        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout_prob) for _ in range(num_encoder_blocks)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout_prob) for _ in range(num_decoder_blocks)
        ])

        self.unembedding_matrix = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def init_params(self, skip_embeddings: str | None = None):
        """Xavier initialize uninitialized layers.

        This weight initialization strategy was first introduced [here](https://proceedings.mlr.press/v9/glorot10a.html). It's used to stabilize gradients during training.

        Args:
            skip_embeddings (str | None, optional): _description_. Defaults to None.
        """

        for name, p in self.named_parameters():
            if skip_embeddings is not None and f"{skip_embeddings}_embeddings" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain("relu"))

    def forward(
        self, src_sequence: torch.Tensor, tgt_sequence: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        src_x = self.src_embeddings(src_sequence)
        tgt_x = self.tgt_embeddings(tgt_sequence)
        src_x = self.src_pos_embeddings(src_x)
        tgt_x = self.tgt_pos_embeddings(tgt_x)

        encoder_representation = src_x
        for i in range(len(self.encoder)):
            encoder_representation = self.encoder[i](encoder_representation, src_mask)

        decoder_output = tgt_x
        for i in range(len(self.decoder)):
            decoder_output = self.decoder[i](decoder_output, encoder_representation, tgt_mask)


def build_model(model_size: str, src_tokenizer, tgt_tokenizer, language_direction="en-it") -> Transformer:
    import os
    import re

    from .configs.load_config import load_config

    CONFIG = load_config()

    if not re.match(r"^[a-z]{2}-[a-z]{2}$"):  # Ensure languange direction is in the default like format
        raise LanguageDirectionInvalidFormat(language_direction)

    if "en" not in language_direction:  # GloVe embeddings available only for English language
        model = Transformer(
            src_tokenizer.vocab_size,
            tgt_tokenizer.vocab_size,
            num_encoder_blocks=CONFIG["model_configs"]["nano"]["num_encoder_layers"],
            num_decoder_blocks=CONFIG["model_configs"]["nano"]["num_decoder_layers"],
            d_model=CONFIG["model_configs"]["nano"]["d_model"],
            num_heads=CONFIG["model_configs"]["nano"]["num_heads"],
            d_ff=CONFIG["model_configs"]["nano"]["d_ff"],
            dropout_prob=CONFIG["model_parameters"]["dropout"],
        )
        model.init_params()

    else:
        glove_version = CONFIG["model_configs"]["nano"]["glove_version"]
        glove_filename = CONFIG["model_configs"]["nano"]["glove_filename"]

        glove_path = os.path.join(os.getcwd(), f"data/{glove_version}/{glove_filename}.txt")

        if "en-" in language_direction:  # English is the source language
            model = Transformer(
                src_tokenizer.vocab_size,
                tgt_tokenizer.vocab_size,
                num_encoder_blocks=CONFIG["model_configs"]["nano"]["num_encoder_layers"],
                num_decoder_blocks=CONFIG["model_configs"]["nano"]["num_decoder_layers"],
                d_model=CONFIG["model_configs"]["nano"]["d_model"],
                num_heads=CONFIG["model_configs"]["nano"]["num_heads"],
                d_ff=CONFIG["model_configs"]["nano"]["d_ff"],
                dropout_prob=CONFIG["model_parameters"]["dropout"],
                src_emb_from_pretrained=True,
                src_emb_pretrained_type=CONFIG["model_configs"]["pretrained_word_embeddings"],
                src_emb_pretrained_path=glove_path,
                src_tokenizer=src_tokenizer,
            )
            model.init_params(skip_embeddings="src")

        else:  # English is the target language
            model = Transformer(
                src_tokenizer.vocab_size,
                tgt_tokenizer.vocab_size,
                num_encoder_blocks=CONFIG["model_configs"]["nano"]["num_encoder_layers"],
                num_decoder_blocks=CONFIG["model_configs"]["nano"]["num_decoder_layers"],
                d_model=CONFIG["model_configs"]["nano"]["d_model"],
                num_heads=CONFIG["model_configs"]["nano"]["num_heads"],
                d_ff=CONFIG["model_configs"]["nano"]["d_ff"],
                dropout_prob=CONFIG["model_parameters"]["dropout"],
                tgt_emb_from_pretrained=True,
                tgt_emb_pretrained_type=CONFIG["model_configs"]["pretrained_word_embeddings"],
                tgt_emb_pretrained_path=glove_path,
                tgt_tokenizer=tgt_tokenizer,
            )
            model.init_params(skip_embeddings="tgt")

    return model
