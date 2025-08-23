import os
import zipfile
from collections import Counter
from multiprocessing import Pool

import requests
import torch
from torch.utils.data import Dataset

from .configs.load_config import load_config

CONFIG = load_config()


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


class TransformerTokenizer:
    """Transformer tokenizer"""

    def __init__(
        self,
        special_tokens: dict[str, str] | None = None,
        contractions: dict[str, str] | None = None,
        num_workers: int = 4,
    ):
        self.vocab = []
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.special_tokens = special_tokens or {
            "sos_token": CONFIG["sos_token"],
            "eos_token": CONFIG["eos_token"],
            "pad_token": CONFIG["pad_token"],
            "unk_token": CONFIG["unk_token"],
        }
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
        self.num_workers = num_workers

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

        return [token[:1000] for token in tokens]

    def from_pretrained(self):
        # TODO
        pass

    def build_vocab_parallel(
        self,
        tokens: list[str],
        min_freq: int = 1,
        extend_with_glove: bool = False,
        glove_version: str = "glove.2024.wikigiga.50d",
        **kwargs,
    ) -> None:
        """_summary_ TODO

        Args:
            tokens (list[str]): _description_
            min_freq (int, optional): _description_. Defaults to 1.
            extend_with_glove (bool, optional): _description_. Defaults to False.
            glove_version (str, optional): _description_. Defaults to "glove.2024.wikigiga.50d".

        Raises:
            GloVeVersionError: _description_
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

            glove_available_versions = [
                "glove.2024.dolma.300d",
                "glove.2024.wikigiga.300d.zip",
                "glove.2024.wikigiga.200d.zip",
                "glove.2024.wikigiga.100d.zip",
                "glove.2024.wikigiga.50d",
                "glove.42B.300d",
                "glove.6B",
                "glove.840B.300d",
                "glove.twitter.27B.zip",
            ]
            if glove_version not in glove_available_versions:
                raise GloVeVersionError(glove_version, glove_available_versions)

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
                        zip_ref.extractall("data")

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
        min_freq: int = 1,
        extend_with_glove: bool = False,
        glove_version: str = "glove.2024.wikigiga.50d",
        **kwargs,
    ) -> None:
        """_summary_ TODO

        Args:
            tokens (list[str]): _description_
            min_freq (int, optional): _description_. Defaults to 1.
            extend_with_glove (bool, optional): _description_. Defaults to False.
            glove_version (str, optional): _description_. Defaults to "glove.2024.wikigiga.50d".

        Raises:
            GloVeVersionError: _description_
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

            glove_available_versions = [
                "glove.2024.dolma.300d",
                "glove.2024.wikigiga.300d.zip",
                "glove.2024.wikigiga.200d.zip",
                "glove.2024.wikigiga.100d.zip",
                "glove.2024.wikigiga.50d",
                "glove.42B.300d",
                "glove.6B",
                "glove.840B.300d",
                "glove.twitter.27B.zip",
            ]
            if glove_version not in glove_available_versions:
                raise GloVeVersionError(glove_version, glove_available_versions)

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
                        zip_ref.extractall("data")

                    for file in os.listdir(glove_folder_path):
                        if file.endswith(".txt"):
                            glove_filepath = os.path.join(glove_folder_path, file)
                            break

                print(f"Loading GloVe {glove_version} tokens from file...")

                with open(glove_filepath, encoding="utf-8") as f:
                    lines = f.readlines()
                glove_tokens = parse_glove_tokens(
                    lines
                )  # Using parse_glove_tokens function in order to avoid code duplication

                initial_size = len(vocab)
                vocab.extend(glove_tokens)
                vocab = list(set(vocab))

                print(f"Added {len(vocab) - initial_size} tokens from GloVe")

            except Exception as e:
                print(f"Error with GloVe processing GloVe: {e}")

        # Create mappings
        self.token_to_idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx_to_token = dict(enumerate(vocab))
        self.vocab = vocab

        print(f"Built vocabulary with {len(vocab)} tokens.")

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if self.vocab_size == 0:
            raise VocabNotBuiltError()

        tokens = self.tokenize(text)

        # Add SOS and EOS tokens for target sequences
        tokens.insert(0, self.special_tokens["sos_token"])
        tokens.append(self.special_tokens["eos_token"])

        token_ids = []
        for token in tokens:
            if token in self.token_to_idx:
                token_ids.append(self.token_to_idx[token])
            else:
                token_ids.append(self.token_to_idx[self.special_tokens["unk_token"]])

        return token_ids

    def decode(self, token_ids: list[int]) -> list[str]:
        """Decode token IDs.
        Returns the unknown token if the input token is not present in the vocabulary.
        """
        return [self.idx_to_token.get(idx, CONFIG["unk_token"]) for idx in token_ids]


class TranslationDataset(Dataset):
    def __init__(
        self,
        dataset,
        src_tokenizer: TransformerTokenizer,
        tgt_tokenizer: TransformerTokenizer,
        src_lang: str,
        tgt_lang: str,
        max_length: int | None = None,
        vocab_min_freq: int = 1,
        extend_vocab_with_glove: bool = False,
    ):
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

        # Downsample the dataset. Mainly for computational contraints and to make tests
        if self.max_length is not None:
            self.dataset = self.dataset.filter(
                lambda x: len(x["translation"][src_lang].split()) <= max_length
                and len(x["translation"][tgt_lang].split()) <= max_length
            )

        self._build_vocabs(vocab_min_freq=vocab_min_freq, extend_with_glove=extend_vocab_with_glove)

    def _build_vocabs(self, vocab_min_freq: int = 1, extend_with_glove: bool = False) -> None:
        """Build vocabularies for tokenizers."""

        print("Building vocabs, it may take a few minutes...")

        # Provides lists of tokens. Here the lists are not converted to sets cause the tokenizer may need the token frequencies
        src_tokens = [
            token for el in self.dataset for token in self.src_tokenizer.tokenize(el["translation"][self.src_lang])
        ]
        tgt_tokens = [
            token for el in self.dataset for token in self.tgt_tokenizer.tokenize(el["translation"][self.tgt_lang])
        ]

        self.src_tokenizer.build_vocab_parallel(
            src_tokens,
            min_freq=vocab_min_freq,
            extend_with_glove=bool(
                extend_with_glove and self.src_lang == "en"
            ),  # GloVe is trained on english only datasets so it doesn't make sense to extend non english vocabs
        )
        self.tgt_tokenizer.build_vocab_parallel(
            tgt_tokens,
            min_freq=vocab_min_freq,
            extend_with_glove=bool(extend_with_glove and self.tgt_lang == "en"),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        example = self.dataset[idx]
        src_text = example["translation"][self.src_lang]
        tgt_text = example["translation"][self.tgt_lang]

        # Tokenize texts
        src_tokens = self.src_tokenizer.encode(src_text)
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text)

        return {
            "src": torch.tensor(src_tokens, dtype=torch.long),
            "tgt": torch.tensor(tgt_tokens, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
