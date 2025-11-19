# Transformer inference script
#
# Author: Giovanni Spadaro - https://giovannispadaro.it
# Project: https://github.com/Giovo17/tfs-mt
# Documentation: https://giovo17.github.io/tfs-mt
#
# Copyright (c) Giovanni Spadaro.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import torch

from tfs_mt.architecture import build_model
from tfs_mt.data_utils import WordTokenizer
from tfs_mt.decoding_utils import greedy_decoding

base_url = "https://huggingface.co/giovo17/tfs-mt/resolve/main/"
src_tokenizer = WordTokenizer.from_pretrained(base_url + "src_tokenizer_word.json")
tgt_tokenizer = WordTokenizer.from_pretrained(base_url + "tgt_tokenizer_word.json")

model = build_model(
    config="https://huggingface.co/giovo17/tfs-mt/resolve/main/config-lock.yaml",
    from_pretrained=True,
    model_path="https://huggingface.co/giovo17/tfs-mt/resolve/main/model.safetensors",
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

input_tokens, input_mask = src_tokenizer.encode("Hi, how are you?")

output = greedy_decoding(model, tgt_tokenizer, input_tokens, input_mask)[0]
print(output)
