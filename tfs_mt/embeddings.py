import torch
import torch.nn as nn


class TokenizerNotProvidedError(Exception):
    def __init__(
        self,
        msg="Tokenizer not provided. When loading pretrained Glove embeddings the tokenizer has to be provided in order to map GloVe words to vocab entries.",
    ):
        super().__init__(msg)


class VocabNotBuiltError(Exception):
    def __init__(self, msg="Tokenizer vocabulary not built."):
        super().__init__(msg)


class EmbeddingDimError(Exception):
    def __init__(self, d_model, from_pretrained):
        msg = f"d_model cannot be None while from_pretrained is False, got d_model = {d_model} and from_pretrained = {from_pretrained}."
        super().__init__(msg)


class EmbeddingTypePathError(Exception):
    def __init__(self, from_pretrained, pretrained_emb_type, pretrained_emb_path):
        msg = f"pretrained_emb_type and pretrained_emb_path cannot be None while from_pretrained is true, \
                got from_pretrained = {from_pretrained}, pretrained_emb_type = {pretrained_emb_type} and pretrained_emb_path = {pretrained_emb_path}."
        super().__init__(msg)


class EmbeddingTypeNotImplementedError(Exception):
    def __init__(self, emb_type):
        msg = f"Embedding type not implemented, got emb_type = {emb_type}"
        super().__init__(msg)


class IncompatibleEmbeddingsDimError(Exception):
    def __init__(self, token_embeddings_shape):
        msg = f"Expected (batch size, max token sequence length, model dimension) got {token_embeddings_shape}"
        super().__init__(msg)


class Embedding(nn.Module):
    """Transformer embeddings layer.

    This implementation uses a randomly initialized embedding lookup table with dimension `[vocab_size, d_model]`.

    There's the possibility of loading pretrained embeddings from GloVe. This choice has been made to achieve acceptable
    performances with low resources training and limited time training.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int | None = None,
        from_pretrained: bool = False,
        pretrained_emb_type: str | None = None,
        pretrained_emb_path: str | None = None,
        **kwargs,
    ):
        super().__init__()

        if d_model is None and not from_pretrained:
            raise EmbeddingDimError(d_model, from_pretrained)

        if from_pretrained:
            if d_model is not None:
                print(f"Ignoring provided d_model ({d_model}). The embeddings dim will be inferred from pretrained.")
            if pretrained_emb_type is None or pretrained_emb_path is None:
                raise EmbeddingTypePathError(from_pretrained, pretrained_emb_type, pretrained_emb_path)
            if pretrained_emb_type == "GloVe" and "tokenizer" not in kwargs:
                raise TokenizerNotProvidedError()
            if kwargs["tokenizer"].vocab_size == 0:
                raise VocabNotBuiltError()

            if pretrained_emb_type == "GloVe":
                embeddings_dim, embeddings_lut = self._load_pretrained(
                    pretrained_emb_path, pretrained_emb_type, tokenizer=kwargs["tokenizer"]
                )
            else:
                embeddings_dim, embeddings_lut = self._load_pretrained(pretrained_emb_path, pretrained_emb_type)

        else:
            embeddings_dim = d_model
            embeddings_lut = nn.Embedding(
                vocab_size, d_model
            )  # Randomly initialized lookup table. If so the initialization will be handled in architecture.Transformer class

        self.d_model = embeddings_dim
        self.embeddings_lut = embeddings_lut

    def _load_pretrained(self, embeddings_path: str, emb_type: str = "GloVe", **kwargs) -> tuple[int, nn.Embedding]:
        if emb_type == "GloVe":
            tokenizer = kwargs["tokenizer"]

            with open(embeddings_path, encoding="utf-8") as f:
                embeddings_dim = len(f.readline().strip().split()) - 1
            embeddings_lut = nn.Embedding(tokenizer.vocab_size, embeddings_dim)

            # NOTE The vocab extension with GloVe tokens is handled by the tokenizer.
            # Here GloVe token embeddings are mapped to the corresponding entry in the embeddings lookup table
            with open(embeddings_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    idx = tokenizer.encode(parts[0])[1]  # The first token coming out of tokenizer.encode is SOS_TOKEN
                    if len(parts[1:]) != embeddings_dim:  # Skip unhandled tokens with spaces, eg. "1 3/4"
                        continue
                    try:
                        token_emb = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
                    except ValueError:
                        continue
                    else:
                        embeddings_lut.weight.data[idx].copy_(token_emb)
        else:
            raise EmbeddingTypeNotImplementedError(emb_type)

        return embeddings_dim, embeddings_lut

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings.

        Args:
            token_ids (torch.Tensor): Input batch of token_ids. Expected shape: [`batch_size`, `sequence_length`]

        Returns:
            torch.Tensor: Output batch of token embeddings. Shape: [`batch_size`, `sequence_length`, `d_model`]
        """

        embeddings = self.embeddings_lut(token_ids)

        return embeddings


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding implementation from the [original paper](https://arxiv.org/abs/1706.03762).

    $$
    \\begin{align*}
    PE_{(pos,2i)} &= \\sin(pos/10000^{2i/d_{model}}) \\\\
    PE_{(pos,2i+1)} &= \\cos(pos/10000^{2i/d_{model}})
    \\end{align*}
    $$
    """

    def __init__(self, d_model: int, max_sequence_length: int = 512):
        super().__init__()

        # Vector of all possible positions in the sequence [max_sequence_length, 1]
        position_id = torch.arange(0, max_sequence_length).unsqueeze(1)
        i_idx = torch.arange(0, d_model, 2, dtype=torch.float32) / d_model  # 2i / d_model
        freq_vec = torch.pow(10000.0, -i_idx)  # 1 / (10000^(2i/d_model))

        pe_lut = torch.zeros(max_sequence_length, d_model)  # Init positional encoding lookup table
        pe_lut[:, 0::2] = torch.sin(position_id * freq_vec)  # Assign sine on even positions
        pe_lut[:, 1::2] = torch.cos(position_id * freq_vec)  # Assing cosine on odd positions

        # Registering this weights as buffers so that they will be saved in model state_dict,
        # but they won't appear in model.parameters so that optimizer will not change them
        self.register_buffer("pe_lut", pe_lut)

    def forward(self, token_embeddings: nn.Embedding) -> nn.Embedding:
        if token_embeddings.ndim != 3 or token_embeddings.size(-1) != self.pe_lut.size(1):
            raise IncompatibleEmbeddingsDimError(token_embeddings.shape)

        positional_encodings = self.pe_lut[: token_embeddings.size(1)]

        return token_embeddings + positional_encodings
