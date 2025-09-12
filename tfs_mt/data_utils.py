import os
import zipfile
from collections import Counter
from functools import partial
from multiprocessing import Pool

import ignite.distributed as idist
import requests
import torch
from datasets import load_dataset
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class VocabNotBuiltError(Exception):
    def __init__(self, msg="Vocabulary not built. Call build_vocab first."):
        super().__init__(msg)


class GloVeVersionError(Exception):
    def __init__(self, glove_version, glove_available_versions):
        msg = f"GloVe version is not available, got {glove_version}, available versions: {glove_available_versions}."
        super().__init__(msg)


def chunkify(input_list: list[str], n: int) -> list[str]:
    """Split list into n nearly equal parts."""
    k, m = divmod(len(input_list), n)
    return [input_list[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def count_tokens_chunk(chunk: list[str]) -> Counter:
    """Count tokens in one chunk."""
    return Counter(chunk)


def parse_glove_tokens(lines: list[str]) -> list[str]:
    """Parse a chunk of GloVe embeddings file into a list of tokens.

    For each line it removes any extra spaces, splits it by spaces and take the first part building the tokens' list.

    It skips "malformed" tokens to avoid duplicates in vocabulary.
    The GloVe file contains tokens with spaces inside, eg. `103 3/4`, which are not handled by the `TrasformerTokenizer` for simplicity.
    """
    result = []
    for line in lines:
        parts = line.strip().split()
        try:
            float(parts[1])
        except ValueError:
            continue
        else:
            token = parts[0].lower()
            result.append(token)

    return result


class WordTokenizer:
    """Word tokenizer.
    Mainly used to let the model be compatible with pretrained GloVe embeddings.

    Args:
        special_tokens (dict[str, str] | None, optional): Special tokens to be considered, eg. BOS_TOKEN, EOS_TOKEN.
        contractions (dict[str, str] | None, optional): Contractions to be considered, eg. 's, 'll . Defaults to None.
        num_workers (int, optional): Number of CPU threads to use in parallel operations, eg. token counting or GloVe token extraction. Defaults to 4.
        tokenizer_max_len (int, optional): Tokenizer max sequence length. Mainly used to limit memory usage and performance impact during training and inference due to attention quadratic complexity. Defaults to 128.
    """

    def __init__(
        self,
        special_tokens: dict[str, str],
        contractions: dict[str, str] | None = None,
        num_workers: int = 4,
        tokenizer_max_len: int = 128,
    ):
        self.vocab: list[str] = []
        self.token_to_idx: dict[str, int] = {}
        self.idx_to_token: dict[int, str] = {}

        self.num_workers = num_workers
        self.tokenizer_max_len = tokenizer_max_len

        self.special_tokens = special_tokens

        self.contractions = contractions or {
            "'s": " 's",
            "'t": " 't",
            "'re": " 're",
            "'ve": " 've",
            "'m": " 'm",
            "'ll": " 'll",
            "'d": " 'd",
            "n't": " n't",
        }

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
        return len(self.vocab) if self.vocab else 0

    @property
    def sos_token_idx(self):
        return self.token_to_idx.get(self.special_tokens["sos_token"], 0)

    @property
    def eos_token_idx(self):
        return self.token_to_idx.get(self.special_tokens["eos_token"], 1)

    @property
    def pad_token_idx(self):
        return self.token_to_idx.get(self.special_tokens["pad_token"], 2)

    @property
    def unk_token_idx(self):
        return self.token_to_idx.get(self.special_tokens["unk_token"], 3)

    def tokenize(self, text: str) -> list[str]:
        """Tokenizer based on GloVe word tokenizer in order to let the model be compatible with GloVe pretrained embeddings.

        Max word length is 1000. Contractions are treated as distinct tokens, eg. `n't`, `'s`, `'ll`.

        Reference: [GloVe source code](https://github.com/stanfordnlp/GloVe/blob/master/src/common.c#L75)

        Args:
            text (str): text to be tokenized.

        Returns:
            List[str]: List of string tokens from text.
        """
        text = text.strip().lower()

        for contraction, replacement in self.contractions.items():
            text = text.replace(contraction, replacement)

        tokens = text.split()

        return [token[:1000] for token in tokens][: self.tokenizer_max_len]

    def from_pretrained(self):
        # TODO
        pass

    def build_vocab_parallel(
        self,
        tokens: list[str],
        min_freq: int = 2,
        extend_with_glove: bool = False,
        glove_version: str = "glove.2024.wikigiga.50d",
        **kwargs,
    ) -> None:
        """Build vocabulary method. Uses multithreading execution to speed up the computation.

        Args:
            tokens (list[str]): Tokens from dataset to build vocabulary on.
            min_freq (int, optional): Minimum number of times a token has to appear in the dataset to be included in the vocabulary. Defaults to 2.
            extend_with_glove (bool, optional): Enable vocabulary extension with GloVe tokens. Defaults to False.
            glove_version (str, optional): GloVe version to use if `extend_with_glove` is `True`. Defaults to "glove.2024.wikigiga.50d".

        Raises:
            GloVeVersionError: Raised when supplied glove_version is unavailable.
        """
        vocab = []
        vocab.extend(self.special_tokens.values())
        vocab_set = set(
            vocab
        )  # Used for quick O(1) insertion of new tokens, instead of searching in the vocab list for each new token (O(n))

        if min_freq > 1:
            # Split tokens in chunks and assign them to CPU thread for parallel counting
            with Pool(self.num_workers) as pool:
                counts_list = pool.map(count_tokens_chunk, chunkify(tokens, self.num_workers))

            # Merge counters
            token_counts = Counter()
            for c in counts_list:
                token_counts.update(c)

            for token, count in token_counts.items():
                if count >= min_freq:
                    vocab_set.add(token.lower())
        else:
            for token in tokens:
                vocab_set.add(token.lower())

        vocab = list(vocab_set)
        del vocab_set

        glove_tokens = []

        # Parallel GloVe loading
        if extend_with_glove:
            print("Extending vocab with GloVe tokens using parallel processing...")

            if glove_version not in self.glove_available_versions:
                raise GloVeVersionError(glove_version, self.glove_available_versions)

            data_path = os.getcwd() + "/data" if "data_path" not in kwargs else kwargs["data_path"]
            glove_folder_path = data_path + f"/{glove_version}"
            os.makedirs(glove_folder_path, exist_ok=True)

            url = f"https://nlp.stanford.edu/data/wordvecs/{glove_version}.zip"
            zip_path = data_path + f"/{glove_version}.zip"

            glove_tokens = []

            try:
                glove_filepath = None
                for file in os.listdir(glove_folder_path):
                    if file.endswith(".txt"):
                        glove_filepath = os.path.join(glove_folder_path, file)
                        break

                if glove_filepath is None:
                    print(f"GloVe not found in {glove_folder_path}. Downloading...")

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

                print(f"Loading GloVe {glove_version} embeddings in parallel...")

                with open(glove_filepath, encoding="utf-8") as f:
                    lines = f.readlines()

                # Split GloVe file lines into chunks and assign them to CPU threads
                line_chunks = list(chunkify(lines, self.num_workers))
                with Pool(self.num_workers) as pool:
                    lists = pool.map(parse_glove_tokens, line_chunks)

                # Merge results from each workers
                for lst in lists:
                    glove_tokens.extend(lst)

                initial_size = len(vocab)
                vocab.extend(glove_tokens)
                vocab = list(set(vocab))

                print(f"Added {len(vocab) - initial_size} tokens from GloVe")

                os.remove(glove_filepath)

            except Exception as e:
                print(f"Error with GloVe processing: {e}")
                raise

        # Create mappings
        self.token_to_idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx_to_token = dict(enumerate(vocab))
        self.vocab = vocab

        print(f"Built vocabulary with {len(vocab)} tokens.")

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
        vocab = []
        vocab.extend(self.special_tokens.values())
        vocab_set = set(vocab)

        if min_freq > 1:
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                if count >= min_freq:
                    vocab_set.add(token.lower())
        else:
            for token in tokens:
                vocab_set.add(token.lower())

        vocab = list(vocab_set)
        del vocab_set

        if extend_with_glove:
            print("Extending vocab with GloVe tokens...")

            if glove_version not in self.glove_available_versions:
                raise GloVeVersionError(glove_version, self.glove_available_versions)

            data_path = os.getcwd() + "/data" if "data_path" not in kwargs else kwargs["data_path"]
            glove_folder_path = data_path + f"/{glove_version}"
            os.makedirs(glove_folder_path, exist_ok=True)

            url = f"https://nlp.stanford.edu/data/wordvecs/{glove_version}.zip"
            zip_path = data_path + f"/{glove_version}.zip"

            glove_tokens = []

            try:
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

                    for file in os.listdir(glove_folder_path):
                        if file.endswith(".txt"):
                            glove_filepath = os.path.join(glove_folder_path, file)
                            break

                print(f"Loading GloVe {glove_version} tokens from file...")

                with open(glove_filepath, encoding="utf-8") as f:
                    lines = f.readlines()
                # Using parse_glove_tokens function in order to avoid code duplication
                glove_tokens = parse_glove_tokens(lines)

                initial_size = len(vocab)
                vocab.extend(glove_tokens)
                vocab = list(set(vocab))

                print(f"Added {len(vocab) - initial_size} tokens from GloVe")

                os.remove(glove_filepath)

            except Exception as e:
                print(f"Error with GloVe processing GloVe: {e}")

        # Create mappings
        self.token_to_idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx_to_token = dict(enumerate(vocab))
        self.vocab = vocab

        print(f"Built vocabulary with {len(vocab)} tokens.")

    def encode(self, input_sequence: str | list[str], pad_to_len: int | None = None) -> tuple[list[int], list[bool]]:
        """Tokenizer encode function.

        It also returns the mask to be used during attention in order not compute it with respect to PAD tokens.
        The mask is designed to True where there's a token with the model has to compute attention to, False otherwise.

        Args:
            input_sequence (str | list[str]): Sequence to be encoded.
            pad_to_len (int | None, optional): Sequence length to pad `input_sequence` with. Defaults to None.

        Raises:
            VocabNotBuiltError: Raised when vocabulary is not built.

        Returns:
            tuple[list[int], list[bool]]: List of token ids and mask.
        """

        if self.vocab_size == 0:
            raise VocabNotBuiltError()

        # Useful when building a TranslationDataset in order to execute tokenize method only once
        tokens = self.tokenize(input_sequence) if isinstance(input_sequence, str) else input_sequence

        # Add SOS and EOS tokens to given sequence
        if tokens[0] != self.special_tokens["sos_token"]:
            tokens.insert(0, self.special_tokens["sos_token"])
        if tokens[-1] != self.special_tokens["eos_token"]:
            tokens.append(self.special_tokens["eos_token"])

        token_ids = [
            self.token_to_idx[token]
            if token in self.token_to_idx
            else self.token_to_idx[self.special_tokens["unk_token"]]
            for token in tokens
        ]

        if pad_to_len is not None:  # Pad sequence to pad_to_len
            pad_to_len += 2  # Considering SOS and EOS tokens
            token_ids.extend([
                self.token_to_idx[self.special_tokens["pad_token"]] for _ in range(pad_to_len - len(tokens))
            ])

        # Disabling attention to pad tokens
        mask = [token != self.token_to_idx[self.special_tokens["pad_token"]] for token in token_ids]

        return token_ids, mask

    def decode(self, token_ids: list[int]) -> list[str]:
        """Decode token IDs.
        Returns the unknown token if the input token is not present in the vocabulary.

        Args:
            token_ids (list[int]): List of tokens ids to decode into text.

        Raises:
            VocabNotBuiltError: Vocabulary is not built.

        Returns:
            list[str]: Decoded text.
        """
        if self.vocab_size == 0:
            raise VocabNotBuiltError()
        return [self.idx_to_token.get(idx, self.special_tokens["unk_token"]) for idx in token_ids]


class TranslationDataset(Dataset):
    """Translation Dataset.

    Args:
        src_texts (list[str]): List of source texts.
        tgt_texts (list[str]): List of target texts.
        src_tokenizer (WordTokenizer): Tokenizer used to preprocess the source language text.
        tgt_tokenizer (WordTokenizer): Tokenizer used to preprocess the target language text.
        src_lang (str): Identifier for the source language, e.g., `"en"` for English.
        tgt_lang (str): Identifier for the target language, e.g., `"it"` for Italian.
        max_sequence_length (int | None, optional): Maximum sequence length for tokenization. If None, sequences are not truncated. Defaults to None.
    """

    def __init__(
        self,
        src_texts: list[str],
        tgt_texts: list[str],
        src_tokenizer: WordTokenizer,
        tgt_tokenizer: WordTokenizer,
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

        # Filter input data excluding texts longer than max_sequence_length
        if max_sequence_length > 0:
            print(f"Max sequence length set to {max_sequence_length}.")
            self.src_texts, self.tgt_texts = [], []
            for src_text, tgt_text in zip(src_texts, tgt_texts, strict=False):
                if (
                    len(src_tokenizer.tokenize(src_text)) > max_sequence_length - 2
                ):  # -2 accounts for SOS and EOS tokens that will be added in the __getitem__ method
                    continue
                if len(tgt_tokenizer.tokenize(tgt_text)) > max_sequence_length - 2:
                    continue
                self.src_texts.append(src_text)
                self.tgt_texts.append(tgt_text)
        else:
            self.src_texts = src_texts
            self.tgt_texts = tgt_texts

        if self.src_tokenizer.vocab_size == 0 or self.tgt_tokenizer.vocab_size == 0:
            if "extend_vocab_with_glove" in kwargs and "glove_version" in kwargs:
                self._build_vocabs(
                    vocab_min_freq=kwargs.get("vocab_min_freq", 2),
                    extend_with_glove=kwargs.get("extend_vocab_with_glove", True),
                    glove_version=kwargs.get("glove_version", "glove.2024.wikigiga.50d"),
                )
            else:
                self._build_vocabs(kwargs.get("vocab_min_freq", 2))

    def _build_vocabs(self, vocab_min_freq: int = 2, extend_with_glove: bool = False, **kwargs) -> None:
        """Build vocabularies for tokenizers."""

        print("Building vocabs, it may take a few minutes...")

        # Provides lists of tokens. Here the lists are not converted to sets cause the tokenizer may need the token frequencies
        src_tokens = [token for text in self.src_texts for token in self.src_tokenizer.tokenize(text)]
        tgt_tokens = [token for text in self.tgt_texts for token in self.tgt_tokenizer.tokenize(text)]

        self.src_tokenizer.build_vocab_parallel(
            src_tokens,
            min_freq=vocab_min_freq,
            extend_with_glove=bool(
                extend_with_glove and self.src_lang == "en"
            ),  # GloVe is trained on english only datasets so it doesn't make sense to extend non english vocabs
            glove_version=kwargs.get("glove_version", "glove.2024.wikigiga.50d"),
        )
        self.tgt_tokenizer.build_vocab_parallel(
            tgt_tokens,
            min_freq=vocab_min_freq,
            extend_with_glove=bool(extend_with_glove and self.tgt_lang == "en"),
            glove_version=kwargs.get("glove_version", "glove.2024.wikigiga.50d"),
        )

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        src_text_tokenized = self.src_tokenizer.tokenize(src_text)
        tgt_text_tokenized = self.tgt_tokenizer.tokenize(tgt_text)

        # src and tgt sequence lengths must be the same to properly compute cross attention
        # The smaller sequence will be padded to the length of the longer sequence
        # Attention mask ensure no attention is computed with pad tokens
        max_seq_len = max(len(src_text_tokenized), len(tgt_text_tokenized))

        # Tokenize texts
        src_tokens, src_mask = self.src_tokenizer.encode(src_text_tokenized, pad_to_len=max_seq_len)
        tgt_tokens, src_mask = self.tgt_tokenizer.encode(tgt_text_tokenized, pad_to_len=max_seq_len)

        return {
            "src": torch.tensor(src_tokens, dtype=torch.long),
            "tgt": torch.tensor(tgt_tokens, dtype=torch.long),
            "src_mask": torch.tensor(src_mask, dtype=torch.bool),
            "tgt_mask": torch.tensor(src_mask, dtype=torch.bool),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def batch_collate_fn(
    batch: dict[str, torch.Tensor | list[str]], src_pad_token_id: int, tgt_pad_token_id: int
) -> dict[str, torch.Tensor | list[str]]:
    """Used to tell the Dataloader how to properly build a batch.

    In order to correctly build a batch every sequence in it has to have the same length,
    so it pads the small sequences to the longest one. It does it for `src`, `tgt`, `src_mask` and `tgt_mask`.

    This function needs a two pad token ids since in this Trasformer implementation there are 2 distinct tokenizers
    with their own vocabulary.
    Each vocabulary is built independently and in parallel, so there's no guarantee that the `pad_token` will have the same ID in both.

    Args:
        batch (dict[str, torch.Tensor  |  list[str]]): Batch of token ids and masks.
        src_pad_token_id (int): Pad token for source sequences.
        tgt_pad_token_id (int): Pad token for target sequences.

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

            padded_seq = pad_sequence(field_data, batch_first=True, padding_value=pad_token_id)
            result[key] = padded_seq
        else:  # src_text and tgt_text
            result[key] = field_data

    return result


def build_data_utils(
    config: DictConfig | ListConfig, return_all: bool = False
) -> (
    tuple[DataLoader, DataLoader]
    | tuple[DataLoader, DataLoader, TranslationDataset, TranslationDataset, WordTokenizer, WordTokenizer]
):
    """Build tokenizers, datasets and dataloaders for Machine Translation.
    Designed to support torch ignite distributed training.

    Args:
        config (DictConfig | ListConfig): Configuration object from omegaconf.
        return_all (bool, optional): Whether to return dataloaders, datasets and tokenizers. Defaults to False.

    Returns:
        tuple[DataLoader, DataLoader] | tuple[DataLoader, DataLoader, TranslationDataset, TranslationDataset, WordTokenizer, WordTokenizer]: Dataloaders or dataloaders, datasets and tokenizers.
    """

    if config.training_hp.distributed_training:
        local_rank = idist.get_local_rank()
        if local_rank > 0:
            idist.barrier()

    data = load_dataset(config.dataset.dataset_id, config.dataset.dataset_name, cache_dir=config.cache_ds_path)["train"]

    src_lang = config.dataset.src_lang
    tgt_lang = config.dataset.tgt_lang

    # Downsample the dataset. Mainly for computational contraints and to make tests.
    if config.dataset.max_len != -1:
        data = data.select(range(config.dataset.max_len))

    split = data.train_test_split(train_size=config.dataset.train_split, seed=config.seed)
    train_data = split["train"]
    test_data = split["test"]

    train_src_texts = [text[src_lang] for text in train_data["translation"]]
    train_tgt_texts = [text[tgt_lang] for text in train_data["translation"]]
    test_src_texts = [text[src_lang] for text in test_data["translation"]]
    test_tgt_texts = [text[tgt_lang] for text in test_data["translation"]]

    # Build tokenizers and vocabs. Both src and tgt tokenizers vocabs are built using the training data
    special_tokens = {
        "sos_token": config.dataset.sos_token,
        "eos_token": config.dataset.eos_token,
        "pad_token": config.dataset.pad_token,
        "unk_token": config.dataset.unk_token,
    }

    src_tokenizer = WordTokenizer(special_tokens)
    tgt_tokenizer = WordTokenizer(special_tokens)

    # To build the vocabularies texts are not filtered based on tokenizer.max_sequence_length
    src_tokens = [token for text in train_src_texts for token in src_tokenizer.tokenize(text)]
    tgt_tokens = [token for text in train_tgt_texts for token in tgt_tokenizer.tokenize(text)]

    src_tokenizer.build_vocab_parallel(
        src_tokens,
        min_freq=config.dataset.vocab_min_freq,
        extend_with_glove=bool(
            src_lang == "en"
        ),  # GloVe is trained on english only datasets so it doesn't make sense to extend non english vocabs
        glove_version=config.model_configs[config.chosen_model_size].glove_version,
    )
    tgt_tokenizer.build_vocab_parallel(
        tgt_tokens,
        min_freq=config.dataset.vocab_min_freq,
        extend_with_glove=bool(tgt_lang == "en"),
        glove_version=config.model_configs[config.chosen_model_size].glove_version,
    )

    config.dataset.src_sos_token_id = src_tokenizer.sos_token_idx
    config.dataset.src_eos_token_id = src_tokenizer.eos_token_idx
    config.dataset.src_pad_token_id = src_tokenizer.pad_token_idx
    config.dataset.src_unk_token_id = src_tokenizer.unk_token_idx
    config.dataset.tgt_sos_token_id = tgt_tokenizer.sos_token_idx
    config.dataset.tgt_eos_token_id = tgt_tokenizer.eos_token_idx
    config.dataset.tgt_pad_token_id = tgt_tokenizer.pad_token_idx
    config.dataset.tgt_unk_token_id = tgt_tokenizer.unk_token_idx

    train_dataset = TranslationDataset(
        train_src_texts,
        train_tgt_texts,
        src_tokenizer,
        tgt_tokenizer,
        src_lang=config.dataset.src_lang,
        tgt_lang=config.dataset.tgt_lang,
        max_sequence_length=config.model_parameters.tokenizer_max_len,
    )
    test_dataset = TranslationDataset(
        test_src_texts,
        test_tgt_texts,
        src_tokenizer,
        tgt_tokenizer,
        src_lang=config.dataset.src_lang,
        tgt_lang=config.dataset.tgt_lang,
        max_sequence_length=config.model_parameters.tokenizer_max_len,
    )

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    if config.training_hp.distributed_training and local_rank == 0:
        idist.barrier()

    # Ignite autodataloader adapts for distributed and non-distributed configurations
    train_dataloader = idist.auto_dataloader(
        train_dataset,
        batch_size=config.train_dataloader.batch_size,
        num_workers=config.train_dataloader.num_workers,
        collate_fn=partial(
            batch_collate_fn,
            src_pad_token_id=config.dataset.src_pad_token_id,
            tgt_pad_token_id=config.dataset.tgt_pad_token_id,
        ),
        shuffle=config.train_dataloader.shuffle,
        drop_last=config.train_dataloader.drop_last,
        persistent_workers=True,  # If True, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive
    )
    test_dataloader = idist.auto_dataloader(
        test_dataset,
        batch_size=config.test_dataloader.batch_size,
        num_workers=config.test_dataloader.num_workers,
        collate_fn=partial(
            batch_collate_fn,
            src_pad_token_id=config.dataset.src_pad_token_id,
            tgt_pad_token_id=config.dataset.tgt_pad_token_id,
        ),
        shuffle=config.test_dataloader.shuffle,
        drop_last=config.test_dataloader.drop_last,
        persistent_workers=True,
    )

    print(f"Train dataloader length: {len(train_dataloader)}")
    print(f"Test dataloader length: {len(test_dataloader)}")

    if return_all:
        return train_dataloader, test_dataloader, train_dataset, test_dataset, src_tokenizer, tgt_tokenizer
    return train_dataloader, test_dataloader
