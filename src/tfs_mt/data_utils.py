# Transformer data utils
#
# Author: Giovanni Spadaro - https://giovannispadaro.it
# Project: https://github.com/Giovo17/tfs-mt
# Documentation: https://giovo17.github.io/tfs-mt
#
# Copyright (c) Giovanni Spadaro.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import json
import os
import re
import string
import zipfile
from abc import ABC, abstractmethod
from collections import Counter
from functools import partial
from itertools import chain

import numpy as np
import requests
import torch
import torch.nn.functional as F
from datasets import load_dataset
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class VocabNotBuiltError(Exception):
    def __init__(self, msg="Vocabulary not built. Call build_vocab first."):
        super().__init__(msg)


class CheckpointEmptyVocabException(Exception):
    def __init__(self, msg="The provided json file has an empty vocab."):
        super().__init__(msg)


class GloVeVersionError(Exception):
    def __init__(self, glove_version, glove_available_versions):
        msg = f"GloVe version is not available, got {glove_version}, available versions: {glove_available_versions}."
        super().__init__(msg)


class DatasetLanguagesNotAvailableError(Exception):
    def __init__(self, src_lang, tgt_lang):
        msg = (
            f"Choosen languages are not available in dataset, got src_lang = '{src_lang}' and tgt_lang = '{tgt_lang}'."
        )
        super().__init__(msg)


class BaseTokenizer(ABC):
    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def sos_token_idx(self):
        pass

    @property
    @abstractmethod
    def eos_token_idx(self):
        pass

    @property
    @abstractmethod
    def pad_token_idx(self):
        pass

    @property
    @abstractmethod
    def unk_token_idx(self):
        pass

    @abstractmethod
    def build_vocab(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: str):
        pass

    @abstractmethod
    def to_json(self, path: str) -> None:
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def encode(
        self,
        input_sequence: str | list[str] | np.ndarray | torch.Tensor,
        pad_to_len: int | None = None,
        return_only_mask: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def decode(self, tokens: list[int] | np.ndarray) -> list[str]:
        pass


def download_glove(output_dir: str, glove_version: str = "glove.2024.wikigiga.50d") -> str:
    """Download GloVe embeddings and returns the filepath."""
    glove_folder_path = output_dir + f"/{glove_version}"
    os.makedirs(glove_folder_path, exist_ok=True)

    url = f"https://nlp.stanford.edu/data/wordvecs/{glove_version}.zip"
    zip_path = output_dir + f"/{glove_version}.zip"

    glove_filepath = None
    for file in os.listdir(glove_folder_path):
        if file.endswith(".txt"):
            glove_filepath = os.path.join(glove_folder_path, file)
            break

    if glove_filepath is None:
        print(f"GloVe not found in {glove_folder_path}. Downloading GloVe ({glove_version})...")

        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(glove_folder_path)
        os.remove(zip_path)

        for file in os.listdir(glove_folder_path):
            if file.endswith(".txt"):
                glove_filepath = os.path.join(glove_folder_path, file)
                break

    return glove_filepath


class WordTokenizer(BaseTokenizer):
    """Word tokenizer.
    Mainly used to let the model be compatible with pretrained GloVe embeddings.

    Args:
        special_tokens (dict[str, str] | None, optional): Special tokens to be considered, eg. SOS_TOKEN, EOS_TOKEN. Defaults to None.
        contractions (list[str] | None, optional): Contractions to be considered, eg. 's, 'll. If None the following set of contractions will be considered: `'s`, `'re`, `'ve`, `'m`, `'ll`, `'d`, `n't`, `'t`, `n'ts`. Defaults to None.
        tokenizer_max_len (int, optional): Tokenizer max sequence length. Mainly used to limit memory usage and performance impact during training and inference due to attention quadratic complexity. Defaults to 128.
        max_vocab_size (int, optional): Maximum number of token in vocabulary. Defaults to 100_000.
    """

    def __init__(
        self,
        special_tokens: dict[str, str] | None = None,
        contractions: list[str, str] | None = None,
        tokenizer_max_len: int = 128,
        max_vocab_size: int = 100_000,
    ):
        self.vocab: dict[str, int] = {}
        self.vocab_reverse: dict[int, str] = {}  # Useful for efficient decoding

        self.tokenizer_max_len = tokenizer_max_len
        self.max_vocab_size = max_vocab_size

        self.special_tokens = special_tokens or {
            "sos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<PAD>",
            "unk_token": "<UNK>",
        }

        # Setup contractions management utilities
        contractions_init = contractions or ["'s", "'re", "'ve", "'m", "'ll", "'d", "n't", "'t", "n'ts"]

        # Prepare a dict of the following type to address the detection of nested contractions:
        # nested_contractions_dict = {
        #     "ANYTHING_n't_ANYTHING": ["n't"],
        #     "n't": ["'t"]
        # }
        # NOTE "'t" is technically a sub contraction of both "ANYTHING_n't_ANYTHING" and "n't" but it will be only considered as sub contraction of n't
        self.nested_contractions_dict: dict[str, list[str]] = {}

        for contraction in contractions_init:
            other_contractions = contractions_init.copy()
            other_contractions.remove(contraction)
            for comp_contraction in other_contractions:
                if contraction in comp_contraction:  # Detect sub contraction
                    # Create new mapping in dict
                    if comp_contraction not in self.nested_contractions_dict:
                        self.nested_contractions_dict[comp_contraction] = [contraction]
                    # Detect sub contraction already present in the dict
                    elif comp_contraction in self.nested_contractions_dict and contraction not in list(
                        chain.from_iterable(self.nested_contractions_dict.values())
                    ):
                        # Merge all elements in sub contractions lists in a single list
                        self.nested_contractions_dict[comp_contraction].append(contraction)

        # Final sets of sub contractions and non-nested contractions
        # NOTE the plain list of all sub contractions will be used during tokenizationin order to better detect correct splitting when the `'` is encountered
        self.all_sub_contractions = list(chain.from_iterable(self.nested_contractions_dict.values()))
        self.contractions = [c for c in contractions_init if c not in self.all_sub_contractions]

        self.glove_available_versions = [
            "glove.2024.dolma.300d",
            "glove.2024.wikigiga.300d",
            "glove.2024.wikigiga.200d",
            "glove.2024.wikigiga.100d",
            "glove.2024.wikigiga.50d",
            "glove.42B.300d",
            "glove.6B",
            "glove.840B.300d",
            "glove.twitter.27B",
        ]

    @property
    def vocab_size(self):
        return len(self.vocab.items()) if self.vocab else 0

    @property
    def sos_token_idx(self):
        return self.vocab.get(self.special_tokens["sos_token"], 0)

    @property
    def eos_token_idx(self):
        return self.vocab.get(self.special_tokens["eos_token"], 1)

    @property
    def pad_token_idx(self):
        return self.vocab.get(self.special_tokens["pad_token"], 2)

    @property
    def unk_token_idx(self):
        return self.vocab.get(self.special_tokens["unk_token"], 3)

    @classmethod
    def from_pretrained(cls: type["WordTokenizer"], path: str) -> "WordTokenizer":
        if path.startswith(("https://", "http://")):
            try:
                response = requests.get(path, timeout=100)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                raise RuntimeError(f"Failed to download tokenizer from {path}: {e}") from e
        else:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)

        if data.get("vocab") is None:
            raise CheckpointEmptyVocabException()

        tokenizer = cls(special_tokens=data.get("special_tokens", {}), contractions=data.get("contractions", {}))
        tokenizer.vocab = data.get("vocab")
        tokenizer.vocab_reverse = {idx: token for token, idx in tokenizer.vocab.items()}

        return tokenizer

    def to_json(self, output_path: str) -> None:
        if self.vocab_size == 0:
            raise VocabNotBuiltError()

        to_save = {
            "type": "WordTokenizer",
            "special_tokens": self.special_tokens,
            "contractions": self.contractions,
            "vocab": self.vocab,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)

    def build_vocab(
        self,
        tokens: list[str],
        min_freq: int = 2,
        extend_with_glove: bool = False,
        glove_version: str = "glove.2024.wikigiga.50d",
        **kwargs,
    ) -> None:
        """Build vocabulary method.

        Args:
            tokens (list[str]): Tokens from dataset to build vocabulary on.
            min_freq (int, optional): Minimum number of times a token has to appear in the dataset to be included in the vocabulary. Defaults to 2.
            extend_with_glove (bool, optional): Enable vocabulary extension with GloVe tokens. Defaults to False.
            glove_version (str, optional): GloVe version to use if `extend_with_glove` is `True`. Defaults to "glove.2024.wikigiga.50d".

        Raises:
            GloVeVersionError: Raised when supplied glove_version is unavailable.
        """
        vocab_list = []
        vocab_list.extend(self.special_tokens.values())
        vocab_set = set(vocab_list)

        if min_freq > 1:
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                # len(vocab_set) is O(1) operation cause the length is stored as a set attribute
                if (count >= min_freq and self.max_vocab_size == -1) or (
                    count >= min_freq and self.max_vocab_size > 0 and len(vocab_set) < self.max_vocab_size
                ):
                    vocab_set.add(token.lower())
        else:
            for token in tokens:
                if self.max_vocab_size == -1 or (self.max_vocab_size > 0 and len(vocab_set) < self.max_vocab_size):
                    vocab_set.add(token.lower())

        vocab_list = list(vocab_set)
        del vocab_set

        if extend_with_glove and (
            self.max_vocab_size == -1 or (self.max_vocab_size > 0 and len(vocab_list) < self.max_vocab_size)
        ):
            print("Extending vocab with GloVe tokens...")

            if glove_version not in self.glove_available_versions:
                raise GloVeVersionError(glove_version, self.glove_available_versions)

            data_path = os.getcwd() + "/data" if "data_path" not in kwargs else kwargs["data_path"]

            glove_tokens = []

            try:
                glove_filepath = download_glove(data_path, glove_version)

                print(f"Loading GloVe {glove_version} tokens from file...")

                with open(glove_filepath, encoding="utf-8") as f:
                    lines = f.readlines()

                # Parse GloVe tokens
                for line in lines:
                    parts = line.strip().split()
                    try:
                        float(parts[1])
                    except ValueError:
                        continue
                    else:
                        token = parts[0].lower()
                        glove_tokens.append(token)

                initial_size = len(vocab_list)
                if self.max_vocab_size != -1:
                    for tok in glove_tokens:
                        if tok not in vocab_list:
                            vocab_list.append(tok)
                        if len(vocab_list) == self.max_vocab_size:
                            break
                else:
                    vocab_list.extend(glove_tokens)
                    vocab_list = list(set(vocab_list))

                print(f"Added {len(vocab_list) - initial_size} tokens from GloVe")

            except Exception as e:
                print(f"Error with GloVe processing GloVe: {e}")

        # Create mappings
        self.vocab = {token: idx for idx, token in enumerate(vocab_list)}
        self.vocab_reverse = dict(enumerate(vocab_list))

        del vocab_list

        print(f"Built vocabulary with {len(self.vocab.items())} tokens.")

    def tokenize(self, text: str) -> list[str]:
        """Tokenizer based on GloVe word tokenizer in order to let the model be compatible with GloVe pretrained embeddings.

        Note:
            Max word length is 1000. Contractions are treated as distinct tokens, eg. `n't`, `'s`, `'ll`.

        Note: Reference
            GloVe tokenizer source code available [here](https://github.com/stanfordnlp/GloVe/blob/master/src/common.c#L75)

        Args:
            text (str): text to be tokenized.

        Returns:
            list[str]: List of string tokens from text.
        """

        text = text.strip().lower()

        for contraction in self.contractions:
            pattern = r"([a-zA-Z]+)" + re.escape(contraction) + r"\b"
            text = re.sub(pattern, r"\1 " + contraction, text)

        for complete_contraction, sub_list in self.nested_contractions_dict.items():
            pattern = r"([a-zA-Z]+)" + re.escape(complete_contraction) + r"\b"
            if not re.compile(complete_contraction).search(text):
                for sub_contraction in sub_list:
                    pattern = r"([a-zA-Z]+)" + re.escape(sub_contraction) + r"\b"
                    text = re.sub(pattern, r"\1 " + sub_contraction, text)
            else:
                text = re.sub(pattern, r"\1 " + complete_contraction, text)

        words = text.split()

        tokens = []
        for word in words:
            # Split words when encountering a `'` that's not involved in a quote or a contraction.
            # eg. "Quell'ultimo" gets splitted into "Quell'" and "ultimo"
            if (
                "'" in word
                and word[0] != "'"
                and word[-1] != "'"
                and word not in self.contractions
                and word not in self.all_sub_contractions
            ):
                parts = word.split("'")
                for i, part in enumerate(parts):
                    if part:
                        if i < len(parts) - 1:
                            part += "'"
                        tokens.append(part)

            # Handle trailing punctuation
            elif word and word[-1] in string.punctuation and word[-1] != "'":
                # Strip all trailing punctuation
                core_word = word.rstrip(string.punctuation)
                punct = word[len(core_word) :]
                if core_word:
                    tokens.append(core_word)
                for p in punct:
                    tokens.append(p)

            else:
                if word:
                    tokens.append(word)

        # Truncate sequences by default
        return tokens[: self.tokenizer_max_len]

    def encode(
        self,
        input_sequence: str | list[str] | np.ndarray | torch.Tensor,
        pad_to_len: int | None = None,
        return_only_mask: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tokenizer encode function.

        It also returns the mask to be used during attention in order not compute it with respect to `PAD` tokens.

        Note:
            The mask is designed to be True where there's a token with the model has to compute attention to, False otherwise.

        Args:
            input_sequence (str | list[str] | np.ndarray | torch.Tensor): Sequence to be encoded or already encoded (useful in decoding stage when this method is only used to provide the mask).
            pad_to_len (int | None, optional): Sequence length to pad `input_sequence` with. Defaults to None.
            return_only_mask (bool, optional): Return only attention mask. Defaults to False.

        Raises:
            VocabNotBuiltError: Raised when vocabulary is not built.

        Returns:
            tuple[np.ndarray, np.ndarray]: Array of token ids and mask.
        """

        if self.vocab_size == 0:
            raise VocabNotBuiltError()

        if return_only_mask:
            if isinstance(input_sequence, np.ndarray):
                mask = input_sequence != self.pad_token_idx
            elif isinstance(input_sequence, torch.Tensor):
                mask = (input_sequence != self.pad_token_idx).to(input_sequence.device)
            else:
                raise ValueError()
            return mask

        # Useful when building a TranslationDataset in order to execute tokenize method only once
        tokens = self.tokenize(input_sequence) if isinstance(input_sequence, str) else input_sequence

        # Add SOS and EOS tokens to given sequence
        if tokens[0] != self.special_tokens["sos_token"]:
            tokens.insert(0, self.special_tokens["sos_token"])
        if tokens[-1] != self.special_tokens["eos_token"]:
            tokens.append(self.special_tokens["eos_token"])

        token_ids = [self.vocab.get(token, self.unk_token_idx) for token in tokens]

        if pad_to_len is not None:  # Pad sequence to pad_to_len
            pad_to_len += 2  # Considering SOS and EOS tokens
            token_ids.extend([self.pad_token_idx for _ in range(pad_to_len - len(tokens))])

        token_ids = np.array(token_ids, dtype=np.long)

        # Mask to disable attention to pad tokens
        mask = np.array([token != self.pad_token_idx for token in token_ids], dtype=np.bool)

        return token_ids, mask

    def decode(self, token_ids: np.ndarray | list[str]) -> list[str]:
        """Decode token IDs.
        Returns the unknown token if the input token is not present in the vocabulary.

        Args:
            token_ids (np.ndarray | list[str]): Array or list of tokens ids to decode into text.

        Raises:
            VocabNotBuiltError: Vocabulary is not built.

        Returns:
            list[str]: Decoded text.
        """
        if self.vocab_size == 0:
            raise VocabNotBuiltError()
        return [self.vocab_reverse.get(idx, self.special_tokens["unk_token"]) for idx in token_ids]


class BPETokenizer(BaseTokenizer):
    pass


class TranslationDataset(Dataset):
    """Translation Dataset.

    Args:
        src_texts (list[str]): List of source texts.
        tgt_texts (list[str]): List of target texts.
        src_tokenizer (BaseTokenizer): Tokenizer used to preprocess the source language text.
        tgt_tokenizer (BaseTokenizer): Tokenizer used to preprocess the target language text.
        src_lang (str): Identifier for the source language, e.g., `"en"` for English.
        tgt_lang (str): Identifier for the target language, e.g., `"it"` for Italian.
        max_sequence_length (int | None, optional): Maximum sequence length for tokenization. If None, sequences are not truncated. Defaults to None.
    """

    def __init__(
        self,
        src_texts: list[str],
        tgt_texts: list[str],
        src_tokenizer: BaseTokenizer,
        tgt_tokenizer: BaseTokenizer,
        src_lang: str,
        tgt_lang: str,
        max_sequence_length: int | None = None,
        **kwargs,
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_sequence_length = max_sequence_length

        # Build tokenizers vocab if empty
        if self.src_tokenizer.vocab_size == 0 or self.tgt_tokenizer.vocab_size == 0:
            if "extend_vocab_with_glove" in kwargs and "glove_version" in kwargs:
                self._build_vocabs(
                    vocab_min_freq=kwargs.get("vocab_min_freq", 2),
                    extend_with_glove=kwargs.get("extend_vocab_with_glove", True),
                    glove_version=kwargs.get("glove_version", "glove.2024.wikigiga.50d"),
                )
            else:
                self._build_vocabs(kwargs.get("vocab_min_freq", 2))

        # Filter input data excluding texts longer than max_sequence_length
        if max_sequence_length is not None and max_sequence_length > 0:
            print(f"Max sequence length set to {max_sequence_length}.")
            self.src_texts, self.tgt_texts = [], []
        else:
            self.src_texts = src_texts
            self.tgt_texts = tgt_texts

        # Prepare tokenized texts here to have them ready in __getitem__ method
        # This will speed up the batch preparation in the dataloader but it will raise system memory usage since the tokenized sequences are cached as dataset attribute
        # (The masks are also cached but their impact is negligible since they are arrays of boolean values)
        print("Caching encoded sequences in memory, this may take some time...")
        self.src_encoded_sequences, self.tgt_encoded_sequences = [], []
        self.src_masks, self.tgt_masks = [], []
        for src_text, tgt_text in zip(src_texts, tgt_texts, strict=False):
            src_text_tokenized = src_tokenizer.tokenize(src_text)
            tgt_text_tokenized = tgt_tokenizer.tokenize(tgt_text)

            # Exclude sequences that exceed max_sequence_length
            if max_sequence_length is not None and max_sequence_length > 0:
                if (
                    len(src_text_tokenized) > max_sequence_length - 2
                ):  # -2 accounts for SOS and EOS tokens that will be added in the __getitem__ method
                    continue
                if len(tgt_text_tokenized) > max_sequence_length - 2:
                    continue
                self.src_texts.append(src_text)
                self.tgt_texts.append(tgt_text)

            # src and tgt sequence lengths must be the same to properly compute cross attention
            # The smaller sequence will be padded to the length of the longer sequence
            # Attention mask ensure no attention is computed with pad tokens
            max_seq_len = max(len(src_text_tokenized), len(tgt_text_tokenized))

            # Tokenize texts
            src_tokens, src_mask = self.src_tokenizer.encode(src_text_tokenized, pad_to_len=max_seq_len)
            tgt_tokens, tgt_mask = self.tgt_tokenizer.encode(tgt_text_tokenized, pad_to_len=max_seq_len)

            self.src_encoded_sequences.append(src_tokens)
            self.tgt_encoded_sequences.append(tgt_tokens)
            self.src_masks.append(src_mask)
            self.tgt_masks.append(tgt_mask)

    def _build_vocabs(self, vocab_min_freq: int = 2, extend_with_glove: bool = False, **kwargs) -> None:
        """Build vocabularies for tokenizers."""

        print("Building vocabs, it may take a few minutes...")

        # Provides lists of tokens. Here the lists are not converted to sets cause the tokenizer may need the token frequencies
        src_tokens = [token for text in self.src_texts for token in self.src_tokenizer.tokenize(text)]
        tgt_tokens = [token for text in self.tgt_texts for token in self.tgt_tokenizer.tokenize(text)]

        self.src_tokenizer.build_vocab(
            src_tokens,
            min_freq=vocab_min_freq,
            extend_with_glove=bool(
                extend_with_glove and self.src_lang == "en"
            ),  # GloVe is trained on english only datasets so it doesn't make sense to extend non english vocabs
            glove_version=kwargs.get("glove_version", "glove.2024.wikigiga.50d"),
        )
        self.tgt_tokenizer.build_vocab(
            tgt_tokens,
            min_freq=vocab_min_freq,
            extend_with_glove=bool(extend_with_glove and self.tgt_lang == "en"),
            glove_version=kwargs.get("glove_version", "glove.2024.wikigiga.50d"),
        )

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        src_tokens = self.src_encoded_sequences[idx]
        tgt_tokens = self.tgt_encoded_sequences[idx]
        src_mask = self.src_masks[idx]
        tgt_mask = self.tgt_masks[idx]
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        return {
            "src": torch.tensor(src_tokens, dtype=torch.long),
            "tgt": torch.tensor(tgt_tokens, dtype=torch.long),
            "src_mask": torch.tensor(src_mask, dtype=torch.bool),
            "tgt_mask": torch.tensor(tgt_mask, dtype=torch.bool),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def batch_collate_fn(
    batch: dict[str, torch.Tensor | list[str]],
    src_pad_token_id: int,
    tgt_pad_token_id: int,
    pad_all_to_len: int = -1,
) -> dict[str, torch.Tensor | list[str]]:
    """Used to tell the Dataloader how to properly build a batch.

    In order to correctly build a batch every sequence in it has to have the same length,
    so it pads the small sequences to the longest one. It does it for `src`, `tgt`, `src_mask` and `tgt_mask`.

    Note:
        This function needs two `PAD` token ids since in this Trasformer implementation there are 2 distinct tokenizers
        with their own vocabulary.
        Each vocabulary is built independently and in parallel, so there's no guarantee that it will have the same ID in both.

    Tip:
        By padding all sequences in the dataset to the same length higher GPU usage can achieved.

        It has to be coupled with `torch.compile` usage and with the dynamic cudagraph compilation disabled (`torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = False`)

    Args:
        batch (dict[str, torch.Tensor  |  list[str]]): Batch of token ids and masks.
        src_pad_token_id (int): Pad token for source sequences.
        tgt_pad_token_id (int): Pad token for target sequences.
        pad_all_to_len (int): Sequence length to pad all sequences. If -1 it gets ignored. Defaults to -1.

    Returns:
        dict[str, torch.Tensor | list[str]]: Batch with padded sequences.
    """

    keys = batch[0].keys()
    result = {}

    for key in keys:
        field_data = [sample[key] for sample in batch]

        if isinstance(field_data[0], torch.Tensor):
            if key == "src":
                pad_token_id = src_pad_token_id
            elif key == "tgt":
                pad_token_id = tgt_pad_token_id
            else:  # mask tensors
                pad_token_id = 0  # will be replaced with False

            if pad_all_to_len != -1:
                num_padding_values = pad_all_to_len - field_data[0].shape[0]
                if num_padding_values >= 0:
                    field_data[0] = F.pad(field_data[0], (0, num_padding_values), value=pad_token_id)

            padded_seq = pad_sequence(field_data, batch_first=True, padding_value=pad_token_id)
            result[key] = padded_seq
        else:  # src_text and tgt_text
            result[key] = field_data

    return result


def build_data_utils(
    config: DictConfig | ListConfig, return_all: bool = False, **kwargs
) -> (
    tuple[DataLoader, DataLoader]
    | tuple[DataLoader, DataLoader, TranslationDataset, TranslationDataset, BaseTokenizer, BaseTokenizer]
):
    """Build tokenizers, datasets and dataloaders for Machine Translation.

    Args:
        config (DictConfig | ListConfig): Configuration object from omegaconf.
        return_all (bool, optional): Whether to return dataloaders, datasets and tokenizers. Defaults to False.

    Returns:
        tuple[DataLoader, DataLoader] | tuple[DataLoader, DataLoader, TranslationDataset, TranslationDataset, BaseTokenizer, BaseTokenizer]: Dataloaders or dataloaders, datasets and tokenizers.
    """

    data = load_dataset(config.dataset.dataset_id, config.dataset.dataset_name, cache_dir=config.cache_ds_path)["train"]

    # TODO If resuming tokenizers from pretrained add check to ensure src and tgt languages matched between the provided config and the checkpoint one
    src_lang = config.dataset.src_lang
    tgt_lang = config.dataset.tgt_lang

    # Downsample the dataset. Mainly for computational contraints and to make tests.
    if config.dataset.max_len != -1:
        data = data.select(range(config.dataset.max_len))

    print(f"Train test splitting - {config.dataset.train_split}/{1 - float(config.dataset.train_split):.2f}")
    split = data.train_test_split(train_size=config.dataset.train_split, seed=config.seed, shuffle=False)
    train_data = split["train"]
    test_data = split["test"]

    try:
        train_src_texts, train_tgt_texts = [], []
        for text in train_data["translation"]:
            train_src_texts.append(text[src_lang])
            train_tgt_texts.append(text[tgt_lang])
        test_src_texts, test_tgt_texts = [], []
        for text in test_data["translation"]:
            test_src_texts.append(text[src_lang])
            test_tgt_texts.append(text[tgt_lang])
    except KeyError as err:  # Mainly here to ensure consistency when resuming model checkpoint
        raise DatasetLanguagesNotAvailableError(src_lang, tgt_lang) from err

    # Build tokenizers and vocabs. Both src and tgt tokenizers vocabs are built using the training data
    special_tokens = {
        "sos_token": config.tokenizer.sos_token,
        "eos_token": config.tokenizer.eos_token,
        "pad_token": config.tokenizer.pad_token,
        "unk_token": config.tokenizer.unk_token,
    }

    # Get tokenizers from pretrained
    if "src_tokenizer" in kwargs and "tgt_tokenizer" in kwargs:
        print("Getting tokenizers from pretrained...")
        src_tokenizer, tgt_tokenizer = kwargs.get("src_tokenizer"), kwargs.get("tgt_tokenizer")

    else:
        print("Building tokenizers and vocabularies...")
        tok_types_dict = {"word": WordTokenizer, "bpe": BPETokenizer}
        src_tokenizer = tok_types_dict[config.tokenizer.type](
            special_tokens,
            tokenizer_max_len=config.tokenizer.max_seq_len,
            max_vocab_size=config.tokenizer.max_vocab_size,
        )
        tgt_tokenizer = tok_types_dict[config.tokenizer.type](
            special_tokens,
            tokenizer_max_len=config.tokenizer.max_seq_len,
            max_vocab_size=config.tokenizer.max_vocab_size,
        )

        # To build the vocabularies texts are not filtered based on tokenizer.max_sequence_length
        src_tokens = [token for text in train_src_texts for token in src_tokenizer.tokenize(text)]
        tgt_tokens = [token for text in train_tgt_texts for token in tgt_tokenizer.tokenize(text)]

        # BPE Tokenizer
        if config.tokenizer.type == "bpe":
            # TODO BPE should take all available text in order to extract statistics
            # and the tokenize method can be done only if the vocabulary is built
            src_tokenizer.build_vocab(src_tokens)
            tgt_tokenizer.build_vocab(tgt_tokens)
        else:
            src_tokenizer.build_vocab(
                src_tokens,
                min_freq=config.tokenizer.vocab_min_freq,
                extend_with_glove=bool(
                    src_lang == "en" and config.model_configs.pretrained_word_embeddings == "GloVe"
                ),  # GloVe is trained on english only datasets so it doesn't make sense to extend non english vocabs
                glove_version=config.model_configs[config.chosen_model_size].glove_version,
            )
            tgt_tokenizer.build_vocab(
                tgt_tokens,
                min_freq=config.tokenizer.vocab_min_freq,
                extend_with_glove=bool(tgt_lang == "en" and config.model_configs.pretrained_word_embeddings == "GloVe"),
                glove_version=config.model_configs[config.chosen_model_size].glove_version,
            )

    config.tokenizer.src_sos_token_idx = src_tokenizer.sos_token_idx
    config.tokenizer.src_eos_token_idx = src_tokenizer.eos_token_idx
    config.tokenizer.src_pad_token_idx = src_tokenizer.pad_token_idx
    config.tokenizer.src_unk_token_idx = src_tokenizer.unk_token_idx
    config.tokenizer.tgt_sos_token_idx = tgt_tokenizer.sos_token_idx
    config.tokenizer.tgt_eos_token_idx = tgt_tokenizer.eos_token_idx
    config.tokenizer.tgt_pad_token_idx = tgt_tokenizer.pad_token_idx
    config.tokenizer.tgt_unk_token_idx = tgt_tokenizer.unk_token_idx

    print("Building datasets...")
    train_dataset = TranslationDataset(
        train_src_texts,
        train_tgt_texts,
        src_tokenizer,
        tgt_tokenizer,
        src_lang=config.dataset.src_lang,
        tgt_lang=config.dataset.tgt_lang,
        max_sequence_length=config.tokenizer.max_seq_len,
    )
    test_dataset = TranslationDataset(
        test_src_texts,
        test_tgt_texts,
        src_tokenizer,
        tgt_tokenizer,
        src_lang=config.dataset.src_lang,
        tgt_lang=config.dataset.tgt_lang,
        max_sequence_length=config.tokenizer.max_seq_len,
    )

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_dataloader.batch_size,
        num_workers=config.train_dataloader.num_workers,
        collate_fn=partial(
            batch_collate_fn,
            src_pad_token_id=config.tokenizer.src_pad_token_idx,
            tgt_pad_token_id=config.tokenizer.tgt_pad_token_idx,
            pad_all_to_len=config.tokenizer.max_seq_len if config.train_dataloader.pad_all_to_max_len else -1,
        ),
        shuffle=config.train_dataloader.shuffle,
        drop_last=config.train_dataloader.drop_last,
        persistent_workers=True,  # If True, the dataloader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive
        pin_memory=torch.cuda.is_available(),  # Pinned memory is not used anyway by torch where no accelerator is employed.
        prefetch_factor=config.train_dataloader.prefetch_factor,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.test_dataloader.batch_size,
        num_workers=config.test_dataloader.num_workers,
        collate_fn=partial(
            batch_collate_fn,
            src_pad_token_id=config.tokenizer.src_pad_token_idx,
            tgt_pad_token_id=config.tokenizer.tgt_pad_token_idx,
            pad_all_to_len=config.tokenizer.max_seq_len if config.test_dataloader.pad_all_to_max_len else -1,
        ),
        shuffle=config.test_dataloader.shuffle,
        drop_last=config.test_dataloader.drop_last,
        persistent_workers=True,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=config.train_dataloader.prefetch_factor,
    )

    print(f"Train dataloader length: {len(train_dataloader)}")
    print(f"Test dataloader length: {len(test_dataloader)}")

    if return_all:
        return train_dataloader, test_dataloader, train_dataset, test_dataset, src_tokenizer, tgt_tokenizer
    return train_dataloader, test_dataloader
