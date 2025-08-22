import os
import re
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


class TranslationDataset(Dataset):
    def __init__(
        self,
        dataset,
        src_tokenizer,
        tgt_tokenizer,
        src_lang,
        tgt_lang,
        max_length=None,
        vocab_min_freq=1,
        extend_vocab_with_glove=False,
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

    def _build_vocabs(self, vocab_min_freq: int = 1, extend_with_glove: bool = False):
        """Build vocabularies for tokenizers."""

        print("Building vocabs, it may take a few minutes...")

        # Provides lists of tokens
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

    def __getitem__(self, idx) -> dict[str, torch.Tensor | str]:
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


def chunkify(lst, n):
    """Split list into n nearly equal parts."""
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def count_tokens_chunk(chunk):
    """Count tokens in one chunk."""
    return Counter(chunk)


def parse_glove_chunk(lines):
    """Parse a list of lines into embeddings dict."""
    result = {}
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        token = parts[0].lower()
        try:
            embedding = [float(x) for x in parts[1:]]
        except Exception as e:
            print(e)
            continue
        result[token] = embedding

    return result


class TransformerTokenizer:
    def __init__(self, special_tokens=None, num_workers=4):
        self.special_tokens = special_tokens or {
            "sos_token": CONFIG["sos_token"],
            "eos_token": CONFIG["eos_token"],
            "pad_token": CONFIG["pad_token"],
            "unk_token": CONFIG["unk_token"],
        }
        self.vocab = []
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.num_workers = num_workers

    def tokenize(self, text: str) -> list[str]:
        """Tokenizer based on GloVe word tokenizer in order to let the model be compatible with GloVe pretrained embeddings.
        Max word length is 1000. Contractions are considered are separated tokens, eg. n't, 's, 'll
        Reference: https://github.com/stanfordnlp/GloVe/blob/master/src/common.c#L75

        Args:
            text (str): text to be tokenized.

        Returns:
            List[str]: List of string tokens from text.
        """

        text = text.lower()

        contractions = {
            "'s": " 's",
            "'t": " 't",
            "'re": " 're",
            "'ve": " 've",
            "'m": " 'm",
            "'ll": " 'll",
            "'d": " 'd",
            "n't": " n't",
        }
        for contraction, replacement in contractions.items():
            text = text.replace(contraction, replacement)

        tokens = re.findall(r"\b\w+\b", text)

        return [token[:1000] for token in tokens]

    def from_pretrained(self):
        # TODO
        pass

    def build_vocab_parallel(self, tokens, min_freq=1, extend_with_glove=False):
        vocab = []
        vocab.extend(self.special_tokens.values())

        if min_freq > 1:
            with Pool(self.num_workers) as pool:
                counts_list = pool.map(count_tokens_chunk, chunkify(tokens, self.num_workers))

            # Merge counters
            token_counts = Counter()
            for c in counts_list:
                token_counts.update(c)

            for token, count in token_counts.items():
                if count >= min_freq and token not in vocab:
                    vocab.append(token.lower())
        else:
            seen = set(vocab)
            for token in tokens:
                low = token.lower()
                if low not in seen:
                    vocab.append(low)
                    seen.add(low)

        glove_embeddings = {}

        # Parallel GloVe loading
        if extend_with_glove:
            print("Extending vocab with GloVe tokens...")

            glove_version = "glove.2024.wikigiga.50d"
            data_path = os.getcwd() + "/data"
            glove_folder_path = data_path + f"/{glove_version}"
            os.makedirs(glove_folder_path, exist_ok=True)

            url = f"https://nlp.stanford.edu/data/wordvecs/{glove_version}.zip"
            zip_path = data_path + f"/{glove_version}.zip"

            try:
                glove_file = None
                for file in os.listdir(glove_folder_path):
                    if file.endswith(".txt"):
                        glove_file = os.path.join(glove_folder_path, file)
                        break

                if glove_file is None:
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
                            glove_file = os.path.join(glove_folder_path, file)
                            break

                print(f"Loading GloVe {glove_version} embeddings in parallel...")

                with open(glove_file, encoding="utf-8") as f:
                    lines = f.readlines()

                line_chunks = list(chunkify(lines, self.num_workers))
                with Pool(self.num_workers) as pool:
                    dicts = pool.map(parse_glove_chunk, line_chunks)

                # Merge dicts
                for d in dicts:
                    glove_embeddings.update(d)

                initial_size = len(vocab)
                for token in glove_embeddings:
                    if token not in vocab:
                        vocab.append(token)

                print(f"Added {len(vocab) - initial_size} tokens from GloVe")

                if os.path.exists(zip_path):
                    os.remove(zip_path)

            except Exception as e:
                print(f"Error with GloVe processing: {e}")
                raise

        # Create mappings
        self.token_to_idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx_to_token = dict(enumerate(vocab))
        self.vocab = vocab

        if extend_with_glove:  # Create vocab file with embeddings
            print("Creating vocab + embedding file...")
            vocab_file_path = os.path.join(data_path, "vocab_with_embeddings.txt")
            with open(vocab_file_path, "w+", encoding="utf-8") as f:
                for idx, token in enumerate(vocab):
                    if token in glove_embeddings:
                        embedding_str = " ".join(map(str, glove_embeddings[token]))
                        f.write(f"{idx} {token} {embedding_str}\n")
                    else:
                        f.write(f"{idx} {token}\n")
            print(f"Saved to {vocab_file_path}.")

        print(f"Built vocabulary with {len(vocab)} tokens.")

    def build_vocab(self, tokens, min_freq=1, extend_with_glove=False):
        vocab = []
        vocab.extend(self.special_tokens.values())

        if min_freq > 1:
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                if count >= min_freq and token not in vocab:
                    vocab.append(token.lower())
        else:
            for token in tokens:
                if token not in vocab:
                    vocab.append(token.lower())

        glove_embeddings = {}

        if extend_with_glove:
            print("Extending vocab with GloVe tokens...")

            glove_version = "glove.2024.wikigiga.50d"  # glove.2024.dolma.300d
            data_path = os.getcwd() + "/data"
            glove_folder_path = data_path + f"/{glove_version}"
            os.makedirs(glove_folder_path, exist_ok=True)

            url = f"https://nlp.stanford.edu/data/wordvecs/{glove_version}.zip"
            zip_path = data_path + f"/{glove_version}.zip"

            try:
                glove_file = None
                for file in os.listdir(glove_folder_path):
                    if file.endswith(".txt"):
                        glove_file = os.path.join(glove_folder_path, file)
                        break

                if glove_file is None:
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
                            glove_file = os.path.join(glove_folder_path, file)
                            break

                else:
                    print(f"Loading GloVe {glove_version} embeddings from file...")

                    with open(glove_file, encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            token = parts[0].lower()
                            embedding = [float(x) for x in parts[1:]]
                            glove_embeddings[token] = embedding

                    initial_size = len(vocab)
                    for token in glove_embeddings:
                        if token not in vocab:
                            vocab.append(token)

                    print(f"Added {len(vocab) - initial_size} tokens from GloVe")

                os.remove(zip_path)

            except Exception as e:
                print(f"Error with GloVe processing GloVe: {e}")

        # Create mappings
        self.token_to_idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx_to_token = dict(enumerate(vocab))
        self.vocab = vocab

        if extend_with_glove:  # Create vocab file with embeddings
            print("Creating a vocab + embedding file. It'll be useful in model embeddings.")

            data_path = os.getcwd() + "/data"
            vocab_file_path = data_path + "/vocab_with_embeddings.txt"
            with open(vocab_file_path, "w+", encoding="utf-8") as f:
                for idx, token in enumerate(vocab):
                    if token in glove_embeddings:
                        embedding_str = " ".join(map(str, glove_embeddings[token]))
                        f.write(f"{idx} {token} {embedding_str}\n")
                    else:
                        f.write(f"{idx} {token}\n")

            print(f"Vocab_glove_embeddings file saved to {vocab_file_path}.")

        print(f"Built vocabulary with {len(vocab)} tokens.")

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""

        if self.vocab is None:
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
        """Decode token IDs"""
        tokens = [self.idx_to_token.get(idx, CONFIG["unk_token"]) for idx in token_ids]
        return tokens

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
