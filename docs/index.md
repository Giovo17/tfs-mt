<h1 align="center">tfs-mt</h1>

<div align="center">
    <a href="https://img.shields.io/github/v/release/Giovo17/tfs-mt" alt="Release">
        <img src="https://img.shields.io/github/v/release/Giovo17/tfs-mt"/></a>
    <a href="https://github.com/Giovo17/tfs-mt/actions/workflows/main.yml?query=branch%3Amain" alt="Build status">
      <img src="https://img.shields.io/github/actions/workflow/status/Giovo17/tfs-mt/main.yml?branch=main" /></a>
    <a href="https://github.com/Giovo17/tfs-mt/blob/main/LICENSE" alt="License">
        <img src="https://img.shields.io/github/license/Giovo17/tfs-mt" /></a>
    <a href="https://github.com/Giovo17/tfs-mt" alt="GitHub">
        <img src="https://img.shields.io/badge/github-repo-blue?logo=github"/></a>
    <a href="https://huggingface.co/giovo17/tfs-mt" alt="Hugging Face">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-tfs--mt-ffc107?color=ffc107&logoColor=white"/></a>
    <a href="https://huggingface.co/spaces/giovo17/tfs-mt-demo" alt="Demo app">
        <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo%20app-tfs--mt--demo-ffc107?color=ffc107&logoColor=white"/></a>
    <a href="https://pypi.org/project/tfs-mt/" alt="PyPi monthly downloads">
        <img src="https://img.shields.io/pypi/dm/tfs-mt"/></a>

</div>


This project implements the Transformer architecture from scratch considering Machine Translation as the usecase. It's mainly intended as an educational resource and a functional implementation of the architecture and the training/inference logic.

## Getting Started

### From pip

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

## Architecture

For a detailed explanation of the architecture and design choices, please refer to the [Architecture Documentation](https://giovo17.github.io/tfs-mt/architecture_explain/).

### Model Sizes

The project supports various model configurations to suit different computational resources:

| Parameter                | Nano     | Small    | Base     | Original |
| :----------------------- | :------- | :------- | :------- | :------- |
| **Encoder Layers** | 4        | 6        | 8        | 6        |
| **Decoder Layers** | 4        | 6        | 8        | 6        |
| **d_model**        | 50       | 100      | 300      | 512      |
| **Num Heads**      | 4        | 6        | 8        | 8        |
| **d_ff**           | 200      | 400      | 800      | 2048     |
| **Norm Type**      | PostNorm | PostNorm | PostNorm | PostNorm |
| **Dropout**        | 0.1      | 0.1      | 0.1      | 0.1      |
| **GloVe Dim**      | 50d      | 100d     | 300d     | -        |
