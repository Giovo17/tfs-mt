# Transformer embeddings
#
# Author: Giovanni Spadaro - https://giovannispadaro.it
# Project: https://github.com/Giovo17/tfs-mt
# Documentation: https://giovo17.github.io/tfs-mt
#
# Copyright (c) Giovanni Spadaro.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import math
import os

import torch
import torch.nn as nn
from jaxtyping import Float

from .data_utils import download_glove


class TokenizerNotSuppliedError(Exception):
    def __init__(
        self,
        msg="Tokenizer not supplied. When loading pretrained Glove embeddings the tokenizer has to be supplied in order to map GloVe words to vocab entries.",
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

    Note: There's the possibility of loading pretrained embeddings from GloVe.
        This choice has been made to achieve acceptable performances with low resources training and limited time training.

    Note:
        GloVe embeddings available for English only.

    Args:
        vocab_size (int): Number of tokens in vocabulary.
        d_model (int | None, optional): Model dimension. Defaults to None.
        from_pretrained (bool, optional): Load embeddings from pretrained. Defaults to False.
        pretrained_emb_type (str | None, optional): Type of pretrained embeddings. Defaults to None.
        pretrained_emb_path (str | None, optional): Path of pretrained embeddings. Defaults to None.

    Raises:
        EmbeddingDimError: Raised when the provided embedding dimension does not match the expected size.
        EmbeddingTypePathError: Raised when the embedding type or file path is invalid or not found.
        TokenizerNotSuppliedError: Raised when a tokenizer is required but has not been supplied.
        VocabNotBuiltError: Raised when an operation requiring a built vocabulary is attempted before the vocabulary is constructed.
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
                print(f"Ignoring d_model ({d_model}). The embeddings dim will be inferred from pretrained.")
            if pretrained_emb_type is None or pretrained_emb_path is None:
                raise EmbeddingTypePathError(from_pretrained, pretrained_emb_type, pretrained_emb_path)
            if pretrained_emb_type == "GloVe" and "tokenizer" not in kwargs:
                raise TokenizerNotSuppliedError()
            if kwargs["tokenizer"].vocab_size == 0:
                raise VocabNotBuiltError()

            if pretrained_emb_type == "GloVe":
                embeddings_dim, embeddings_lut = self.load_pretrained(
                    pretrained_emb_path, pretrained_emb_type, tokenizer=kwargs["tokenizer"]
                )
            else:
                embeddings_dim, embeddings_lut = self.load_pretrained(pretrained_emb_path, pretrained_emb_type)

            # The following operations consist in rescaling the norm of GloVe pretrained embeddings.
            # This is done to have source and target embeddings with a comparable norm to stabilize training.
            embeddings_lut_dummy = nn.Embedding(embeddings_lut.weight.shape[0], embeddings_lut.weight.shape[1])
            nn.init.xavier_uniform_(embeddings_lut_dummy.weight)

            embeddings_lut_dummy_norm = torch.mean(embeddings_lut_dummy.weight.data.norm(dim=1))
            embeddings_lut_norm = torch.mean(embeddings_lut.weight.data.norm(dim=1))

            rescale_coeff = embeddings_lut_dummy_norm / embeddings_lut_norm

            embeddings_lut.weight.data = embeddings_lut.weight.data * rescale_coeff

        else:
            embeddings_dim = d_model
            # Randomly initialized lookup table.
            embeddings_lut = nn.Embedding(vocab_size, embeddings_dim)
            nn.init.xavier_uniform_(embeddings_lut.weight)

        self.d_model = embeddings_dim
        self.scaling_factor = math.sqrt(self.d_model)
        self.embeddings_lut = embeddings_lut

    def load_pretrained(self, embeddings_path: str, emb_type: str = "GloVe", **kwargs) -> tuple[int, nn.Embedding]:
        """Loads pretrained GloVe embedding into the embedding lookup table."""
        if emb_type == "GloVe":
            tokenizer = kwargs["tokenizer"]

            if not os.path.isfile(embeddings_path):
                output_dir = "/".join(embeddings_path.split("/")[:-2])
                download_glove(output_dir, glove_version=embeddings_path.split("/")[-2])

            with open(embeddings_path, encoding="utf-8") as f:
                embeddings_dim = len(f.readline().strip().split()) - 1
            embeddings_lut = nn.Embedding(tokenizer.vocab_size, embeddings_dim)

            # NOTE The vocab extension with GloVe tokens is handled by the tokenizer.
            # Here GloVe token embeddings are mapped to the corresponding entry in the embeddings lookup table
            with open(embeddings_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    idx, _ = tokenizer.encode(parts[0])
                    # If tokenization if non perfectly compatible with the GloVe one, idx can be a sequence of tokens longer then 3
                    # (Consider the first and last tokens coming out of tokenizer.encode are SOS_TOKEN and EOS_TOKEN)
                    # This is a useful check to avoid overwriting in the embeddings_lut matrix
                    if len(idx) > 3:
                        continue
                    idx = idx[1]  # The first token coming out of tokenizer.encode is SOS_TOKEN
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

    def forward(self, token_ids: Float[torch.Tensor, "B S"]) -> Float[torch.Tensor, "B S D"]:
        """Get token embeddings.

        Args:
            token_ids (Float[torch.Tensor, "B S"]): Input batch of token_ids. Where B is the batch size and S is the sequence length.

        Returns:
            Float[torch.Tensor, "B S D"]: Output batch of token embeddings. Where D is d_model.
        """

        embeddings = self.embeddings_lut(token_ids)

        # In the embedding layers, we multiply those weights by sqrt(d_model) (Attention is all you need page 5)
        embeddings = embeddings * self.scaling_factor

        return embeddings


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding implementation from the [original paper](https://arxiv.org/abs/1706.03762).

    $$
    \\begin{align*}
    PE_{(pos,2i)} &= \\sin(pos/10000^{2i/d_{model}}) \\\\
    PE_{(pos,2i+1)} &= \\cos(pos/10000^{2i/d_{model}})
    \\end{align*}
    $$

    Args:
        d_model (int): Model dimension.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.1.
        max_sequence_length (int, optional): Max sequence length. Defaults to 128.
    """

    def __init__(self, d_model: int, dropout_prob: float = 0.1, max_sequence_length: int = 128):
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

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, token_embeddings: Float[torch.Tensor, "B S D"]) -> Float[torch.Tensor, "B S D"]:
        """Get token embeddings with positional information.

        Args:
            token_embeddings (Float[torch.Tensor, "B S D"]): Input batch of token embeddings.

        Raises:
            IncompatibleEmbeddingsDimError: Raised when the input embedding dimension is invalid.

        Returns:
            Float[torch.Tensor, "B S D"]: Token embeddings with added positional information.
        """
        if token_embeddings.ndim != 3 or token_embeddings.size(-1) != self.pe_lut.shape[1]:
            raise IncompatibleEmbeddingsDimError(token_embeddings.shape)

        positional_encodings = self.pe_lut[: token_embeddings.size(1)]

        # we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks (Attention is all you need page 8)
        final_embedding = self.dropout(token_embeddings + positional_encodings)

        return final_embedding
