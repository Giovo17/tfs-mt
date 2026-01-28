<div align="center">

<h1>tfs-mt<br>
Transformer from scratch for Machine Translation</h1>

<a href="https://github.com/Giovo17/tfs-mt/releases" alt="Release">
    <img src="https://img.shields.io/github/v/release/Giovo17/tfs-mt"/></a>
<a href="https://github.com/Giovo17/tfs-mt/actions/workflows/main.yml?query=branch%3Amain" alt="Build status">
<img src="https://img.shields.io/github/actions/workflow/status/Giovo17/tfs-mt/main.yml?branch=main"/></a>
<!--a href="https://github.com/Giovo17/tfs-mt/blob/main/LICENSE" alt="License">
    <img src="https://img.shields.io/github/license/Giovo17/tfs-mt"/></a!-- -->
<a href="https://creativecommons.org/licenses/by-sa/4.0/" alt="License">
    <img src="https://img.shields.io/badge/license-CC_BY_SA_4.0-green.svg"/></a>
<a></a>

<a href="https://pypi.org/project/tfs-mt/" alt="PyPi monthly downloads">
    <img src="https://img.shields.io/pypi/dm/tfs-mt"/></a>

<br>

<a href="https://github.com/Giovo17/tfs-mt">
    🏠 Homepage
</a>
•
<a href="#getting-started">
    ▶️ Getting started
</a>
•
<a href="https://huggingface.co/giovo17/tfs-mt">
    🤗 Hugging Face
</a>
•
<a href="https://huggingface.co/spaces/giovo17/tfs-mt-demo">
    🎬 Demo
</a>

</div>

---

This project implements the Transformer architecture from scratch considering Machine Translation as the usecase. It's mainly intended as an educational resource and a functional implementation of the architecture and the training/inference logic.

## Getting started

### From PyPI

```bash
pip install tfs-mt
```

### From source

#### Prerequisites

- `uv` [[install](https://docs.astral.sh/uv/#installation)]

#### Steps
```bash
git clone https://github.com/Giovo17/tfs-mt.git
cd tfs-mt

uv sync

cp .env.example .env
# Edit .env file with your configuration
```

## Usage

### Training

To start training the model with the default configuration:

```bash
uv run src/train.py
```

### Inference

To run inference using the trained model from the [HuggingFace repo](https://huggingface.co/giovo17/tfs-mt):

```bash
uv run src/inference.py
```

### Configuration

The whole project parameters can be configured in `src/tfs_mt/configs/config.yml`. Key configurations include:

- **Model Architecture**: Config, dropout, GloVe embedding init, ...
- **Training**: Optimizer, Learning rate scheduler, number of epochs, ...
- **Data**: Dataset, Dataloader, Tokenizer, ...


## License

- **Source code**: licensed under the **MIT License**.
    - Note: This project includes modified code derived from PyTorch Ignite, which is licensed under the **BSD 3-Clause License**. See the `LICENSE` file for the full text of both licenses and original copyright notices.

- **Documentation**: located in the `docs/` directory, licensed under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**. See `docs/LICENSE`.

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
