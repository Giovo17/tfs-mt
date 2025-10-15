import torch
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from .architecture import Transformer
from .data_utils import WordTokenizer


@torch.inference_mode()
def greedy_decoding(
    config: DictConfig | ListConfig,
    model: Transformer,
    encoder_representation: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_tokenizer: WordTokenizer,
    max_target_tokens: int = 128,
) -> list[str]:
    """
    Supports batch (decode multiple source sentences) greedy decoding.

    Decoding could be further optimized to cache old token activations because they can't look ahead and so
    adding a newly predicted token won't change old token's activations.

    Example: we input `<s>` and do a forward pass. We get intermediate activations for `<s>` and at the output at position
    0, after the doing linear layer we get e.g. token `<I>`. Now we input `<s>`,`<I>` but `<s>`'s activations will remain
    the same. Similarly say we now got `<am>` at output position 1, in the next step we input `<s>`,`<I>`,`<am>` and so `<I>`'s
    activations will remain the same as it only looks at/attends to itself and to `<s>` and so forth.

    Args:
        config (DictConfig | ListConfig): _description_
        model (Transformer): _description_
        encoder_representation (torch.Tensor): _description_
        src_mask (torch.Tensor): _description_
        tgt_tokenizer (WordTokenizer): _description_
        max_target_tokens (int, optional): _description_. Defaults to 128.

    Returns:
        list[str]: _description_

    Examples:
        ```
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        config = OmegaConf.load("path/to/config")

        _, _, _, _, src_tokenizer, tgt_tokenizer = build_data_utils(config, return_all=True)

        src_sequences = [
            "The bottle floated out in the river",
            "The green witch arrived"
        ]
        src_sequences_enc, src_masks = src_tokenizer.encode(src_sequences)
        src_sequences_enc = torch.tensor(src_sequences_enc, device=device)
        src_masks = torch.tensor(src_masks, device=device)

        model = build_model(config, src_tokenizer, tgt_tokenizer)

        # Load model from pretrained

        model = model.to(device)
        model = model.eval()

        with torch.inference_mode():
            encoder_representation = model.encode(src_sequences, src_masks)

        encoded_sequence = greedy_decoding(
            config=config,
            model=model,
            encoder_representation=encoder_representation,
            src_mask=src_masks,
            tgt_tokenizer=tgt_tokenizer,
            max_target_tokens=config.tokenizer.max_seq_len
        )
        ```
    """

    device = next(model.parameters()).device

    # Generate a batch of sequences starting with SOS token, batch size is inferred by the encoder representation tensor
    tgt_sequence_batch_text = [[config.tokenizer.sos_token] for _ in range(encoder_representation.shape[0])]
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

            if (
                predicted_word == config.tokenizer.eos_token
            ):  # Once EOS token is generated for a sentence in the batch it gets flagged in is_decoded list
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
            target_index = tgt_sequence.index(config.tokenizer.eos_token) + 1
        except ValueError:
            target_index = None

        tgt_sequence = tgt_sequence[:target_index]
        post_processed_sequences.append(tgt_sequence)

    return post_processed_sequences


@torch.inference_mode()
def beam_decoding(
    config: DictConfig | ListConfig,
    model: Transformer,
    encoder_representation: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_tokenizer: WordTokenizer,
    max_target_tokens: int = 128,
) -> list[str]:
    """
    TBA
    """

    pass
