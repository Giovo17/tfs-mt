# Transformer decoding utils
#
# Author: Giovanni Spadaro - https://giovannispadaro.it
# Project: https://github.com/Giovo17/tfs-mt
# Documentation: https://giovo17.github.io/tfs-mt
#
# Copyright (c) Giovanni Spadaro.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from .architecture import Transformer
from .data_utils import BaseTokenizer


@torch.inference_mode()
def greedy_decoding(
    model: Transformer,
    tgt_tokenizer: BaseTokenizer,
    src_tokens: np.ndarray | torch.Tensor,
    src_mask: np.ndarray | torch.Tensor,
    max_target_tokens: int = 128,
    output_mode: str = "str",
) -> list[str] | list[list[str]]:
    """
    Supports batch (decode multiple source sentences) greedy decoding.

    Note: Example
        We input `<s>` and do a forward pass. We get intermediate activations for `<s>` and at the output at position
        0, after the doing linear layer we get e.g. token `<I>`. Now we input `<s>`,`<I>` but `<s>`'s activations will remain
        the same. Similarly say we now got `<am>` at output position 1, in the next step we input `<s>`,`<I>`,`<am>` and so `<I>`'s
        activations will remain the same as it only looks at/attends to itself and to `<s>` and so forth.

    Note:
        Decoding could be further optimized to cache old token activations because they can't look ahead and so
        adding a newly predicted token won't change old token's activations.

    Args:
        model (Transformer): Encoder-decoder translation model.
        tgt_tokenizer (BaseTokenizer): Target text tokenizer.
        src_tokens (torch.Tensor): Source tokens.
        src_mask (torch.Tensor): Source tokens mask.
        max_target_tokens (int, optional): Max target tokens to output. Defaults to 128.
        output_mode (str, optional): Output mode, if `str` it will return the sequence as a string, if `tokens` it will return as list of tokens. Defaults to "str".

    Returns:
        list[str] | list[list[str]]: Decoded sequences as strings or as list of tokens.
    """

    device = next(model.parameters()).device

    sos_token = tgt_tokenizer.decode([tgt_tokenizer.sos_token_idx])[0]
    eos_token = tgt_tokenizer.decode([tgt_tokenizer.eos_token_idx])[0]

    # Get encoder representation
    if isinstance(src_tokens, np.ndarray) and isinstance(src_tokens, np.ndarray):
        src_tokens = torch.tensor(src_tokens)
        src_mask = torch.tensor(src_mask)
    if src_tokens.ndim == 1 and src_mask.ndim == 1:
        src_tokens.unsqueeze_(0)
        src_mask.unsqueeze_(0)

    src_tokens = src_tokens.to(device)
    src_mask = src_mask.to(device)

    encoder_representation = model.encode(src_tokens, src_mask)

    # Generate a batch of sequences starting with SOS token, batch size is inferred by the encoder representation tensor
    tgt_sequence_batch_text = [[sos_token] for _ in range(encoder_representation.shape[0])]
    tgt_sequence_batch = torch.tensor(
        [[tgt_tokenizer.sos_token_idx] for _ in range(encoder_representation.shape[0])], device=device
    )

    # This list handles when to stop the tokens generation for each sequence in the batch
    is_decoded = [False] * encoder_representation.shape[0]

    while True:
        tgt_mask = tgt_tokenizer.encode(tgt_sequence_batch, return_only_mask=True)

        # Due to cross attention max tgt sequences cannot be longer than max src sequences
        if tgt_sequence_batch.shape[1] > encoder_representation.shape[1]:
            dummy_tensor = torch.ones_like(encoder_representation, device=encoder_representation.device)
            dummy_tensor = dummy_tensor[:, 0, :].unsqueeze(1)
            encoder_representation = torch.cat((encoder_representation, dummy_tensor), dim=1)

            addon_mask = torch.zeros((src_mask.shape[0], 1), dtype=torch.bool, device=src_mask.device)
            src_mask = torch.cat((src_mask, addon_mask), dim=1)

        # Shape = (B*T, V) where T is the current token-sequence length and V target vocab size
        decoder_output = model.decode(tgt_sequence_batch, encoder_representation, tgt_mask, src_mask)

        # Extract only the indices of last token for every target sentence
        num_of_tgt_tokens = tgt_sequence_batch.shape[1]
        decoder_output = decoder_output[:, num_of_tgt_tokens - 1 :: num_of_tgt_tokens]

        # Greedy decode tokens selecting the most probable one and discard other tokens
        most_probable_last_token_indices = torch.argmax(decoder_output, dim=-1).cpu().numpy()

        # Find target tokens associated with these indices
        predicted_words = []
        for row in most_probable_last_token_indices:
            predicted_words.append(tgt_tokenizer.decode(row)[0])

        for idx, predicted_word in enumerate(predicted_words):
            tgt_sequence_batch_text[idx].append(predicted_word)
            # Once EOS token is generated for a sentence in the batch it gets flagged in is_decoded list
            if predicted_word == eos_token:
                is_decoded[idx] = True

        if all(is_decoded) or num_of_tgt_tokens == max_target_tokens:
            break

        # Prepare the input for the next iteration: merge old token ids with the new column of most probable token ids
        tgt_sequence_batch = torch.cat(
            (tgt_sequence_batch, torch.tensor(most_probable_last_token_indices, device=device)), dim=1
        )

    # Post process the sentences: remove everything after the EOS token
    post_processed_sequences = []
    for tgt_sequence in tgt_sequence_batch_text:
        try:
            target_index = tgt_sequence.index(eos_token) + 1
        except ValueError:
            target_index = None

        tgt_sequence = tgt_sequence[:target_index]
        post_processed_sequences.append(tgt_sequence)

    if output_mode == "str":
        post_processed_sequences_str = []
        for i in range(len(post_processed_sequences)):
            seq = post_processed_sequences[i]
            seq.remove(sos_token)
            if eos_token in seq:
                seq.remove(eos_token)
            sequence_as_str = " ".join(seq)
            post_processed_sequences_str.append(sequence_as_str)
        post_processed_sequences = post_processed_sequences_str

    return post_processed_sequences


@torch.inference_mode()
def beam_decoding(
    model: Transformer,
    tgt_tokenizer: BaseTokenizer,
    src_tokens: np.ndarray | torch.Tensor,
    src_mask: np.ndarray | torch.Tensor,
    max_target_tokens: int = 128,
    output_mode: str = "str",
) -> list[str]:
    """
    TBA
    """

    pass
