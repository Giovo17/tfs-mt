import math

import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from .data_utils import WordTokenizer
from .embeddings import Embedding, SinusoidalPositionalEncoding


class DimError(Exception):
    def __init__(self, d_model, num_heads):
        msg = f"d_model is not divisible by num_heads, got d_model = {d_model} and num_heads = {num_heads}."
        super().__init__(msg)


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


class ModelSizeNotChoosen(Exception):
    def __init__(self, msg="Model size not choosen. Add chosen_model_size to config."):
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

        # NOTE Usually transformer models have d_model divisible by num_heads.
        # This guarantees that the attention heads' outputs when stacked together are d_model shaped vectors (considering each embedding vector)
        # In this implementation it has been preferred to remove this contraint. When exploiting the support of GloVe pretrained embeddings,
        # d_model is fixed to GloVe embeddings sizes, namely 25, 50, 100, 200, 300, so in this scenario num_heads would be limited to predefined set of values due to int quantization.
        # Considering the intermidiate output dimensions there will be no problems since the 3 initial projections matrices' shapes have been adjusted in order to map
        # from d_model to num_heads * self.d_head (see below why the projection matrices are not splitted into head-specific matrices).
        # Considering the final output dimensions the W_O matrix will project num_heads*d_head dimensional vectors into d_model vectors, so the whole operation
        # continues to be mathematically consistent ensuring input and output dimension of this module are the same.
        # eg. d_model = 50, num_heads = 4, d_head = int(d_model/num_heads) = 12
        # q = x * W_Q   q shape is 48 (same goes to k and v), W_Q shape is 50x48   (in this example q is the concatention of the q vectors )
        # output = attention_output * W_O   output shape is 50, W_O shape is 48x50
        # if d_model % num_heads != 0:
        #    raise DimError(d_model, num_heads)

        self.d_head = int(d_model / num_heads)  # Query, key and value embeddings dimension. d_k = d_v = d_head

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

        self.dropout = nn.Dropout(dropout_prob)

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
            x_query (torch.Tensor): Matrix of input query embeddings of shape [B, S, D], where B is the batch size, S is the sequence length and D is d_model.
            x_key (torch.Tensor): Matrix of input key embeddings of shape [B, S, D].
            x_value (torch.Tensor): Matrix of input value embeddings of shape [B, S, D].
            attention_mask (torch.BoolTensor | None, optional): Attention mask to avoid computing attention to padding tokens. It's also used to apply causal masking in decoder self attention. Defaults to None.

        Returns:
            torch.Tensor: Processed output tensor.
        """
        batch_size = x_query.shape[0]

        # W_Q(x)          [B, S, D]
        # After reshape   [B, S, A, d_k]
        # After transpose [B, A, S, d_k] where A is num_heads and S is the sequence length
        query_matrices = self.W_Q(x_query).reshape(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        key_matrices = self.W_K(x_key).reshape(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        value_matrices = self.W_V(x_value).reshape(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # Concatenated heads outputs, shape [B, A*d_k]
        attn_output = self.attention(query_matrices, key_matrices, value_matrices, attention_mask=attention_mask)

        # Reshape back from [B, A, S, d_k] to [B, S, A*d_k]
        attn_output_reshaped = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_head)

        # Attention scores, shape [B, S, D]. Combine heads outputs into a single D-dimensional output.
        return self.W_O(attn_output_reshaped)

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
            query (torch.Tensor): Matrix of input query embeddings of shape [B, S, D], where B is the batch size, S is the sequence length and D is d_model.
            key (torch.Tensor): Matrix of input key embeddings of shape [B, S, D].
            value (torch.Tensor): Matrix of input value embeddings of shape [B, S, D].
            attention_mask (torch.BoolTensor | None, optional): Attention mask to avoid computing attention to padding tokens. It's also used to apply causal masking in decoder self attention. Defaults to None.

        Returns:
            torch.Tensor: Attention matrix.
        """
        QKt = torch.matmul(query, key.transpose(-2, -1)) / self.scaling_factor

        # NOTE Moved in Transformer forward method for efficiency
        # Reshape from [B, S] to [B, 1, 1, S] to properly broadcast attention mask to all QKt matrices
        # Broadcasting doc: https://docs.pytorch.org/docs/stable/notes/broadcasting.html
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # NOTE Moved in Transformer forward method for efficiency
        # Build attention mask matrix with shape [B, 1, S, S] to properly mask QKt matrix
        # eg. with considering only the last 2 dimensions
        # input = [[ True,  True, False]]
        # output = [[ True,  True, False],
        #           [ True,  True, False],
        #           [False, False, False]]
        # attention_mask = torch.matmul(attention_mask.to(torch.int).transpose(-1,-2), attention_mask.to(torch.int)).to(torch.bool)

        if attention_mask is not None:
            # NOTE Adding this control to correctly process masking considering that target input sequence will be shrinked by one token
            # This is expecially needed when computing cross attention in decoder blocks due to the usage of src_mask which cannot be shrinked accordingly a priori
            if attention_mask.shape[-1] > QKt.shape[-1] or attention_mask.shape[-2] > QKt.shape[-2]:
                attention_mask = attention_mask[:, :, : QKt.shape[-2], : QKt.shape[-1]]
            QKt.masked_fill_(attention_mask == False, float("-inf"))

        # Applying the softmax on last dim makes results in a QKt matrix with normalized rows
        QKt_norm = torch.softmax(QKt, dim=-1)

        QKt_norm = self.dropout(QKt_norm)

        # Fix nan propagation due to softmax processing of masked matrix containing entire rows full of -inf
        QKt_norm = QKt_norm.masked_fill(torch.isnan(QKt_norm), 0.0)

        return torch.matmul(QKt_norm, value)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_prob: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
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
        d_model (int): Model dimension.
        num_heads (int): Number of attention heads.
        d_ff (int): Size of middle feedforward layer.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_prob: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_prob)
        self.feedforward = FeedForward(d_model, d_ff, dropout_prob)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        t1 = self.layer_norm1(x)
        t2 = self.self_attention(x_query=t1, x_key=t1, x_value=t1, attention_mask=attention_mask)
        # We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized (Attentio is all you need page 8)
        t2 = self.dropout(t2)
        t3 = t2 + x

        t4 = self.layer_norm2(t3)
        t5 = self.feedforward(t4)
        t5 = self.dropout(t5)
        h = t5 + t3

        return h


class DecoderBlock(nn.Module):
    """Transformer Decoder block.

    Using prenorm approach as in EncoderBlock.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of attention heads.
        d_ff (int): Size of middle feedforward layer.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_prob: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_prob)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout_prob)
        self.feedforward = FeedForward(d_model, d_ff, dropout_prob)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        encoder_representation: torch.Tensor,
        tgt_mask: torch.BoolTensor,
        src_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        t1 = self.layer_norm1(x)
        t2 = self.self_attention(x_query=t1, x_key=t1, x_value=t1, attention_mask=tgt_mask)
        # We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized (Attention is all you need page 8)
        t2 = self.dropout(t2)
        t3 = t2 + x

        t4 = self.layer_norm2(t3)
        t5 = self.cross_attention(
            x_query=t4, x_key=encoder_representation, x_value=encoder_representation, attention_mask=src_mask
        )
        t5 = self.dropout(t5)
        t6 = t5 + t3

        t7 = self.layer_norm3(t6)
        t8 = self.feedforward(t7)
        t8 = self.dropout(t8)
        h = t8 + t6

        return h


class Transformer(nn.Module):
    """Transformer model.

    Using Language Model head to map decoder output representation to tokens in vocabulary.

    Args:
        src_vocab_size (int): Size of source language vocabulary.
        tgt_vocab_size (int): Size of target language vocabulary.
        num_encoder_blocks (int, optional): Number of encoder blocks. Defaults to 6.
        num_decoder_blocks (int, optional): Number of decoder blocks. Defaults to 6.
        d_model (int, optional): Model dimension. Defaults to 512.
        num_heads (int, optional): Number of heads in MultiHead Attention. Defaults to 8.
        d_ff (int, optional): Size of middle feedforward layer. Defaults to 2048.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.1.

    Raises:
        MissingArgumentsError: Raised when `src_emb_from_pretrained` is supplied in `kwargs`, but `src_emb_pretrained_type` and `src_emb_pretrained_path` are not supplied. This error also applies for `tgt_emb_from_pretrained`.
        MissingArgumentsGloVeError: Raised when GloVe embeddings from pretrained are wanted to be loaded and `src_tokenizer` is not supplied. This error also applies for `tgt_emb_from_pretrained`.
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
        max_seq_len: int = 128,
        **kwargs,
    ):
        super().__init__()

        # Source embedding init
        if kwargs.get("src_emb_from_pretrained") is not None:
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
        if kwargs.get("tgt_emb_from_pretrained") is not None:
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

        self.src_pos_embeddings = SinusoidalPositionalEncoding(d_model, dropout_prob, max_sequence_length=max_seq_len)
        self.tgt_pos_embeddings = SinusoidalPositionalEncoding(d_model, dropout_prob, max_sequence_length=max_seq_len)

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
            skip_embeddings (str | None, optional): Which type of embedding should skip Xavier initialization. Useful when initializing embeddings with GloVe. Defaults to None.
        """
        for name, p in self.named_parameters():
            if skip_embeddings is not None and f"{skip_embeddings}_embeddings" in name:
                print(f"Skipping Xavier init for {skip_embeddings} embeddings")
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

        # Reshape from [B, S] to [B, 1, 1, S] to properly broadcast attention mask to all QKt matrices
        # Broadcasting doc: https://docs.pytorch.org/docs/stable/notes/broadcasting.html
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(2)

        # Build attention mask matrix with shape [B, 1, S, S] to properly mask QKt matrix
        # eg. with considering only the last 2 dimensions
        # input = [[ True,  True, False]]
        # output = [[ True,  True, False],
        #           [ True,  True, False],
        #           [False, False, False]]
        src_mask = torch.matmul(src_mask.to(torch.float).transpose(-1, -2), src_mask.to(torch.float)).to(torch.bool)
        tgt_mask = torch.matmul(tgt_mask.to(torch.float).transpose(-1, -2), tgt_mask.to(torch.float)).to(torch.bool)

        # Apply causal masking
        # This speeds up computation since only one masked_fill will be applied in each decoder attention module
        causal_mask = (
            torch.triu(torch.ones((tgt_mask.shape[0], 1, tgt_mask.shape[-1], tgt_mask.shape[-1])), diagonal=1) == 0
        ).to(tgt_mask.device)
        tgt_mask = tgt_mask & causal_mask  # Extract intersection between pad_mask and causal mask

        encoder_representation = src_x
        for i in range(len(self.encoder)):
            encoder_representation = self.encoder[i](encoder_representation, src_mask)

        decoder_representation = tgt_x
        for i in range(len(self.decoder)):
            decoder_representation = self.decoder[i](decoder_representation, encoder_representation, tgt_mask, src_mask)

        decoder_output = self.unembedding_matrix(decoder_representation)

        return decoder_output

    # Methods needed during decoding
    def encode(self, src_sequence: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_x = self.src_embeddings(src_sequence)
        src_x = self.src_pos_embeddings(src_x)

        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        src_mask = torch.matmul(src_mask.to(torch.float).transpose(-1, -2), src_mask.to(torch.float)).to(torch.bool)

        encoder_representation = src_x
        for i in range(len(self.encoder)):
            encoder_representation = self.encoder[i](encoder_representation, src_mask)

        return encoder_representation

    def decode(
        self,
        tgt_sequence: torch.Tensor,
        encoder_representation: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_x = self.tgt_embeddings(tgt_sequence)
        tgt_x = self.tgt_pos_embeddings(tgt_x)

        tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(2)
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        tgt_mask = torch.matmul(tgt_mask.to(torch.float).transpose(-1, -2), tgt_mask.to(torch.float)).to(torch.bool)
        src_mask = torch.matmul(src_mask.to(torch.float).transpose(-1, -2), src_mask.to(torch.float)).to(torch.bool)

        causal_mask = (
            torch.triu(torch.ones((tgt_mask.shape[0], 1, tgt_mask.shape[-1], tgt_mask.shape[-1])), diagonal=1) == 0
        ).to(tgt_mask.device)
        tgt_mask = tgt_mask & causal_mask

        decoder_representation = tgt_x
        for i in range(len(self.decoder)):
            decoder_representation = self.decoder[i](decoder_representation, encoder_representation, tgt_mask, src_mask)

        decoder_output = self.unembedding_matrix(decoder_representation)

        return decoder_output


def build_model(
    config: DictConfig | ListConfig, src_tokenizer: WordTokenizer, tgt_tokenizer: WordTokenizer
) -> nn.Module:
    """Build Transformer model according to a config file.

    Args:
        config (DictConfig | ListConfig): Project config file.
        src_tokenizer (WordTokenizer): Source text tokenizer.
        tgt_tokenizer (WordTokenizer): Target text tokenizer.

    Raises:
        ModelSizeNotChoosen: Raised when config doesn't have `chosen_model_size` key.

    Returns:
        nn.Module: Initialized Transformer model according to config yaml file and choosen model size.
    """

    import os

    if "chosen_model_size" not in config:
        raise ModelSizeNotChoosen()

    # GloVe embeddings available only for English language
    if (
        config.dataset.src_lang != "en" and config.dataset.tgt_lang != "en"
    ) or config.model_configs.pretrained_word_embeddings is None:
        model = Transformer(
            src_tokenizer.vocab_size,
            tgt_tokenizer.vocab_size,
            num_encoder_blocks=config.model_configs[config.chosen_model_size].num_encoder_layers,
            num_decoder_blocks=config.model_configs[config.chosen_model_size].num_decoder_layers,
            d_model=config.model_configs[config.chosen_model_size].d_model,
            num_heads=config.model_configs[config.chosen_model_size].num_heads,
            d_ff=config.model_configs[config.chosen_model_size].d_ff,
            dropout_prob=config.model_parameters.dropout,
            max_seq_len=config.tokenizer.max_seq_len,
        )
        model.init_params()

    else:
        glove_version = config.model_configs[config.chosen_model_size].glove_version
        glove_filename = config.model_configs[config.chosen_model_size].glove_filename

        glove_path = os.path.join(config.base_path, f"data/{glove_version}/{glove_filename}.txt")

        if config.dataset.src_lang == "en":  # English is the source language
            model = Transformer(
                src_tokenizer.vocab_size,
                tgt_tokenizer.vocab_size,
                num_encoder_blocks=config.model_configs[config.chosen_model_size].num_encoder_layers,
                num_decoder_blocks=config.model_configs[config.chosen_model_size].num_decoder_layers,
                d_model=config.model_configs[config.chosen_model_size].d_model,
                num_heads=config.model_configs[config.chosen_model_size].num_heads,
                d_ff=config.model_configs[config.chosen_model_size].d_ff,
                dropout_prob=config.model_parameters.dropout,
                max_seq_len=config.tokenizer.max_seq_len,
                src_emb_from_pretrained=True,
                src_emb_pretrained_type=config.model_configs.pretrained_word_embeddings,
                src_emb_pretrained_path=glove_path,
                src_tokenizer=src_tokenizer,
            )
            model.init_params(skip_embeddings="src")

        else:  # English is the target language
            model = Transformer(
                src_tokenizer.vocab_size,
                tgt_tokenizer.vocab_size,
                num_encoder_blocks=config.model_configs[config.chosen_model_size].num_encoder_layers,
                num_decoder_blocks=config.model_configs[config.chosen_model_size].num_decoder_layers,
                d_model=config.model_configs[config.chosen_model_size].d_model,
                num_heads=config.model_configs[config.chosen_model_size].num_heads,
                d_ff=config.model_configs[config.chosen_model_size].d_ff,
                dropout_prob=config.model_parameters.dropout,
                max_seq_len=config.tokenizer.max_seq_len,
                tgt_emb_from_pretrained=True,
                tgt_emb_pretrained_type=config.model_configs.pretrained_word_embeddings,
                tgt_emb_pretrained_path=glove_path,
                tgt_tokenizer=tgt_tokenizer,
            )
            model.init_params(skip_embeddings="tgt")

    return model
