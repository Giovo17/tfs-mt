---
language:
- {{src_lang}}
- {{tgt_lang}}
license: mit
tags:
- pytorch
- nlp
- machine-translation
pipeline_tag: translation
{{datasets_section}}
---

<h1 align="center">tfs-mt<br>
Transformer from scratch for Machine Translation</h1>

<div align="center">
    <a href="https://img.shields.io/github/v/release/Giovo17/tfs-mt" alt="Release">
        <img src="https://img.shields.io/github/v/release/Giovo17/tfs-mt"/>
    </a>
    <a href="https://github.com/Giovo17/tfs-mt/actions/workflows/main.yml?query=branch%3Amain" alt="Build status">
      <img src="https://img.shields.io/github/actions/workflow/status/Giovo17/tfs-mt/main.yml?branch=main"/>
    </a>
    <a href="https://huggingface.co/giovo17/tfs-mt/blob/main/LICENSE" alt="License">
        <img src="https://img.shields.io/badge/license-MIT-green.svg"/>
    </a>
    <br>
    <a href="https://github.com/Giovo17/tfs-mt">
        🏠 Homepage
    </a>
    •
    <a href="https://giovo17.github.io/tfs-mt">
        📖 Documentation
    </a>
    •
    <a href="https://huggingface.co/spaces/giovo17/tfs-mt-demo">
        🎬 Demo
    </a>
    •
    <a href="https://pypi.org/project/tfs-mt">
        📦 PyPi
    </a>

</div>

---

This project implements the Transformer architecture from scratch considering Machine Translation as the usecase. It's mainly intended as an educational resource and a functional implementation of the architecture and the training/inference logic.

Here you can find the weights of the trained `small` size Transformer and the pretrained tokenizers.

## Quick Start

```bash
pip install tfs-mt
```

```python
{{inference_code}}
```

## Model Architecture

{{model_architecture}}

### Tokenizer

{{tokenizer_details}}

## Dataset

{{dataset_details}}

## Full training configuration

<details>
<summary>Click to expand complete config-lock.yaml</summary>

```yaml
{{full_config}}
```

</details>


## License

This model weights are licensed under the **MIT License**.

The base weights used for training were sourced from GloVe. Their are licensed under the
[ODC Public Domain Dedication and License (PDDL)](https://opendatacommons.org/licenses/pddl/1-0/).


## Citation

If you use `tfs-mt` in your research or project, please cite:

```bibtex
@software{Spadaro_tfs-mt,
author = {Spadaro, Giovanni},
licenses = {MIT, CC BY-SA 4.0},
title = {{tfs-mt}},
url = {https://github.com/Giovo17/tfs-mt}
}
```
