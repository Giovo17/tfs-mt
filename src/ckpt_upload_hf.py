# Checkpoint uploader script
#
# Author: Giovanni Spadaro - https://giovannispadaro.it
# Project: https://github.com/Giovo17/tfs-mt
# Documentation: https://giovo17.github.io/tfs-mt
#
# Copyright (c) Giovanni Spadaro.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import argparse
import os
import tempfile
from typing import Any

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import RepositoryNotFoundError
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

from tfs_mt.architecture import Transformer
from tfs_mt.ignite_custom_utils.checkpoint import S3Saver

load_dotenv()


class CheckpointUploadError(Exception):
    """Base exception for checkpoint upload errors."""

    pass


class S3DownloadError(CheckpointUploadError):
    """Exception raised when S3 download fails."""

    def __init__(self, path: str, original_error: Exception):
        self.path = path
        self.original_error = original_error
        super().__init__(f"Failed to download from S3: {path}. Error: {original_error}")


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parse S3 path into bucket and key."""
    _, _, path = s3_path.partition("s3://")
    bucket, _, key = path.partition("/")

    if not bucket or not key:
        raise ValueError(f"Invalid S3 path: {s3_path}. Expected format: s3://bucket/key")

    return bucket, key


def download_from_s3(s3_client, bucket: str, key: str, local_path: str):
    """Download a file from S3."""
    try:
        s3_client.download_file(bucket, key, local_path)
    except Exception as e:
        raise S3DownloadError(f"s3://{bucket}/{key}", e) from e


def load_checkpoint_from_path(checkpoint_path: str, device: str = "cpu") -> dict[str, Any]:
    """
    Load checkpoint from local file or S3.

    Args:
        checkpoint_path: Path to checkpoint file (local or s3://)
        device: Device to load the checkpoint on

    Returns:
        Dictionary containing checkpoint data
    """
    is_s3 = checkpoint_path.startswith("s3://")

    if is_s3:
        bucket, key = parse_s3_path(checkpoint_path)
        s3 = S3Saver._make_s3_client()

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            tmp_path = tmp.name

            print(f"Downloading checkpoint from S3: {checkpoint_path}")
            download_from_s3(s3, bucket, key, tmp_path)
            checkpoint = torch.load(tmp_path, map_location=device, weights_only=True)
            return checkpoint

    else:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from local file: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        return checkpoint


def get_base_path(checkpoint_path: str) -> str:
    """Get the base directory path from checkpoint path."""
    if checkpoint_path.endswith((".pt", ".pth")):
        return "/".join(checkpoint_path.split("/")[:-1])
    return checkpoint_path


def load_config(config_path: str) -> DictConfig | None:
    """Load config-lock.yaml from local or S3 using OmegaConf."""
    is_s3 = config_path.startswith("s3://")

    if is_s3:
        bucket, key = parse_s3_path(config_path)
        s3 = S3Saver._make_s3_client()

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as tmp:
            tmp_path = tmp.name

            download_from_s3(s3, bucket, key, tmp_path)
            config = OmegaConf.load(tmp_path)
            return config

    else:
        if not os.path.isfile(config_path):
            print(f"Warning: Config file not found: {config_path}")
            return None

        config = OmegaConf.load(config_path)
        return config


def format_model_architecture(config: DictConfig, chosen_size: str) -> list[str]:
    """Format model architecture details for README."""
    lines = []

    if "model_configs" in config and chosen_size in config.model_configs:
        model_cfg = OmegaConf.to_container(config.model_configs[chosen_size], resolve=True)
        lines.append(f"**Model Size**: `{chosen_size}`")
        lines.append("")

        architecture_params = {
            "num_encoder_layers": "Encoder Layers",
            "num_decoder_layers": "Decoder Layers",
            "d_model": "Model Dimension",
            "num_heads": "Attention Heads",
            "d_ff": "FFN Dimension",
            "norm_type": "Normalization Type",
        }

        for key, label in architecture_params.items():
            if key in model_cfg:
                lines.append(f"- **{label}**: {model_cfg[key]}")

        if "dropout" in config.model_parameters:
            lines.append(f"- **Dropout**: {config.model_parameters.dropout}")

        # Embeddings info
        if "pretrained_word_embeddings" in config.model_configs:
            lines.append(f"- **Pretrained Embeddings**: {config.model_configs.pretrained_word_embeddings}")
        if "positional_embeddings" in config.model_configs:
            lines.append(f"- **Positional Embeddings**: {config.model_configs.positional_embeddings}")

        # GloVe details if available
        if "glove_version" in model_cfg:
            lines.append(f"- **GloVe Version**: {model_cfg['glove_version']}")

    return lines


def format_training_details(config: DictConfig) -> list[str]:
    """Format training hyperparameters for README."""
    lines = []

    hp = config.training_hp

    # Basic training settings
    lines.append(f"- **Epochs**: {hp.num_epochs}")
    amp_status = f"{hp.use_amp}"
    if hp.use_amp and "amp_dtype" in hp:
        amp_status += f" ({hp.amp_dtype})"
    lines.append(f"- **Mixed Precision**: {amp_status}")
    lines.append(f"- **Torch Compile Mode**: {hp.torch_compile_mode}")
    lines.append(f"- **Gradient Clipping**: {hp.max_gradient_norm}")

    # Loss configuration
    loss_cfg = OmegaConf.to_container(hp.loss, resolve=True)
    lines.append(f"- **Loss Type**: {loss_cfg.get('type', 'N/A')}")
    lines.append(f"- **Label Smoothing**: {loss_cfg['label_smoothing']}")

    # Optimizer
    opt_cfg = OmegaConf.to_container(hp.optimizer, resolve=True)
    lines.append(f"- **Optimizer**: {opt_cfg.get('type', 'N/A')}")
    lines.append(f"  - Weight Decay: {opt_cfg.get('weight_decay', 'N/A')}")
    lines.append(f"  - Beta1: {opt_cfg.get('beta1', 'N/A')}, Beta2: {opt_cfg.get('beta2', 'N/A')}")
    lines.append(f"  - Epsilon: {opt_cfg.get('eps', 'N/A')}")

    # Learning rate scheduler
    lr_cfg = OmegaConf.to_container(hp.lr_scheduler, resolve=True)
    lines.append(f"- **LR Scheduler**: {lr_cfg.get('type', 'N/A')}")
    lines.append(f"  - Min LR: {lr_cfg.get('min_lr', 'N/A')}")
    lines.append(f"  - Max LR: {lr_cfg.get('max_lr', 'N/A')}")
    lines.append(f"  - Warmup Iterations: {lr_cfg.get('warmup_iters', 'N/A')}")
    if "stable_iters_prop" in lr_cfg:
        lines.append(f"  - Stable Iterations Proportion: {lr_cfg['stable_iters_prop']}")

    return lines


def format_tokenizer_details(config: DictConfig) -> list[str]:
    """Format tokenizer configuration for README."""
    lines = []

    tok_cfg = OmegaConf.to_container(config.tokenizer, resolve=True)

    lines.append(f"- **Type**: {tok_cfg.get('type', 'N/A')}")
    lines.append(f"- **Max Sequence Length**: {tok_cfg.get('max_seq_len', 'N/A')}")
    lines.append(f"- **Max Vocabulary Size**: {tok_cfg.get('max_vocab_size', 'N/A')}")
    lines.append(f"- **Minimum Frequency**: {tok_cfg.get('vocab_min_freq', 'N/A')}")
    lines.append("")

    return lines


def format_dataset_details(config: DictConfig) -> list[str]:
    """Format dataset information for README."""
    lines = []

    ds_cfg = OmegaConf.to_container(config.dataset, resolve=True)

    lines.append(f"- **Task**: {ds_cfg.get('dataset_task', 'N/A')}")
    lines.append(f"- **Dataset ID**: `{ds_cfg.get('dataset_id', 'N/A')}`")
    lines.append(f"- **Dataset Name**: `{ds_cfg.get('dataset_name', 'N/A')}`")
    lines.append(f"- **Source Language**: {ds_cfg.get('src_lang', 'N/A')}")
    lines.append(f"- **Target Language**: {ds_cfg.get('tgt_lang', 'N/A')}")
    lines.append(f"- **Train Split**: {ds_cfg.get('train_split', 'N/A')}")

    return lines


def generate_readme(config: DictConfig, model_name: str) -> str:
    """Generate a comprehensive README for the Hugging Face repository."""

    src = config.dataset.get("src_lang", "")
    tgt = config.dataset.get("tgt_lang", "")

    readme_parts = [
        "---",
        "language:",
        f"- {src}",
        f"- {tgt}",
        "license: mit",
        "tags:",
        "- pytorch",
        "- nlp",
        "- machine-translation",
        "pipeline_tag: translation",
    ]

    # Add dataset tag if available
    if "dataset_id" in config.dataset:
        dataset_id = config.dataset.dataset_id
        readme_parts.append("datasets:")
        readme_parts.append(f"- {dataset_id}")

    readme_parts.extend([
        "---",
        "",
        f"# {model_name}",
        "",
    ])

    # Model description
    readme_parts.append("Transformer from scratch for Machine Translation.")

    inference_script_path = os.path.join(os.getcwd(), "src/inference.py")
    with open(inference_script_path) as f:
        inference_code = f.readlines()

    inference_code = [line.rstrip("\n") for line in inference_code if not line.startswith("#")]
    if not inference_code[0]:
        inference_code.pop(0)

    readme_parts.extend([
        "",
        "## Quick Start",
        "",
        "```bash",
        "pip install tfs-mt",
        "```",
        "",
        "```python",
        *inference_code,
        "```",
        "",
    ])

    # Model Architecture
    readme_parts.extend([
        "## Model Architecture",
        "",
    ])

    chosen_size = config.get("chosen_model_size", "unknown")
    arch_lines = format_model_architecture(config, chosen_size)
    readme_parts.extend(arch_lines)
    readme_parts.append("")

    # Tokenizer Configuration
    readme_parts.extend([
        "### Tokenizer",
        "",
    ])

    tok_lines = format_tokenizer_details(config)
    readme_parts.extend(tok_lines)
    readme_parts.append("")

    # Dataset Information
    readme_parts.extend([
        "## Dataset",
        "",
    ])

    ds_lines = format_dataset_details(config)
    readme_parts.extend(ds_lines)
    readme_parts.append("")

    readme_parts.extend([
        "## Full training configuration",
        "",
        "<details>",
        "<summary>Click to expand complete config-lock.yaml</summary>",
        "",
        "```yaml",
        OmegaConf.to_yaml(config, resolve=True).strip(),
        "```",
        "",
        "</details>",
    ])

    return "\n".join(readme_parts)


def upload_to_huggingface(
    checkpoint_path: str,
    repo_id: str,
    hf_token: str | None = None,
    private: bool = True,
):
    """
    Upload model checkpoint, tokenizers, config, and README to Hugging Face.

    Args:
        checkpoint_path: Path to checkpoint (local or s3://)
        repo_id: Hugging Face repository ID (e.g., "username/model-name")
        hf_token: Hugging Face API token
        private: Whether to create a private repository
    """
    api = HfApi(token=hf_token)

    try:
        repo_files = api.list_repo_files(repo_id=repo_id)
    except RepositoryNotFoundError:
        repo_files = []

    # Create repository
    print(f"Creating/accessing repository: {repo_id}")
    try:
        create_repo(repo_id, private=private, token=hf_token, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create repo (may already exist): {e}")

    # Get base path for tokenizers and config
    base_path = get_base_path(checkpoint_path)
    is_s3 = checkpoint_path.startswith("s3://")

    # Load checkpoint
    checkpoint = load_checkpoint_from_path(checkpoint_path, "cpu")

    # Save model to temporary file
    ckpt_dict = {
        "model": "model.pt",
        "optimizer": "optimizer.pt",
        "lr_scheduler": "lr_scheduler.pt",
        "trainer": "trainer.pt",
    }
    for key, filename in ckpt_dict.items():
        if key not in checkpoint:
            print(f"Skipping '{key}', not found in checkpoint.")
            continue

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            tmp_path = tmp.name

            print(f"Saving {key} checkpoint...")
            torch.save(checkpoint[key], tmp_path)

            if filename not in repo_files:
                print(f"Uploading {key} checkpoint to repo as {filename}...")
                api.upload_file(
                    path_or_fileobj=tmp_path,
                    path_in_repo=filename,
                    repo_id=repo_id,
                    token=hf_token,
                )

    config_path = f"{base_path}/config-lock.yaml"
    config = load_config(config_path)

    # Upload tokenizers
    if is_s3:
        src_tokenizer_filename = f"src_tokenizer_{config.tokenizer.type}.json"
        tgt_tokenizer_filename = f"tgt_tokenizer_{config.tokenizer.type}.json"

        bucket, base_key = parse_s3_path(base_path)
        src_tokenizer_key = f"{base_key}/{src_tokenizer_filename}"
        tgt_tokenizer_key = f"{base_key}/{tgt_tokenizer_filename}"
        s3 = S3Saver._make_s3_client()

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                src_tokenizer_tmp_path = os.path.join(tmp_dir, src_tokenizer_filename)
                tgt_tokenizer_tmp_path = os.path.join(tmp_dir, tgt_tokenizer_filename)

                print("Downloading tokenizers from S3...")
                download_from_s3(s3, bucket, src_tokenizer_key, src_tokenizer_tmp_path)
                download_from_s3(s3, bucket, tgt_tokenizer_key, tgt_tokenizer_tmp_path)

                if src_tokenizer_filename not in repo_files or tgt_tokenizer_filename not in repo_files:
                    print("Uploading tokenizers...")
                    api.upload_file(
                        path_or_fileobj=src_tokenizer_tmp_path,
                        path_in_repo=src_tokenizer_filename,
                        repo_id=repo_id,
                        token=hf_token,
                    )
                    api.upload_file(
                        path_or_fileobj=tgt_tokenizer_tmp_path,
                        path_in_repo=tgt_tokenizer_filename,
                        repo_id=repo_id,
                        token=hf_token,
                    )
            except S3DownloadError as e:
                print(f"Warning: Could not upload tokenizers: {e}")

    else:
        src_tokenizer_path = f"{base_path}/{src_tokenizer_filename}"
        tgt_tokenizer_path = f"{base_path}/{tgt_tokenizer_filename}"
        if os.path.isfile(src_tokenizer_path) and os.path.isfile(tgt_tokenizer_path):
            if src_tokenizer_filename not in repo_files and tgt_tokenizer_filename not in repo_files:
                print("Uploading tokenizers...")
                api.upload_file(
                    path_or_fileobj=src_tokenizer_path,
                    path_in_repo=src_tokenizer_filename,
                    repo_id=repo_id,
                    token=hf_token,
                )
                api.upload_file(
                    path_or_fileobj=tgt_tokenizer_path,
                    path_in_repo=tgt_tokenizer_filename,
                    repo_id=repo_id,
                    token=hf_token,
                )
        else:
            raise FileNotFoundError("One the tokenizers file is not present")

    # Load safetensor to provide a safer alternative to PyTorch pickle format
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_safetensor_path = os.path.join(tmp_dir, "model.safetensors")

        model = Transformer(
            src_vocab_size=config.src_tokenizer_vocab_size,
            tgt_vocab_size=config.tgt_tokenizer_vocab_size,
            num_encoder_blocks=config.model_configs[config.chosen_model_size].num_encoder_layers,
            num_decoder_blocks=config.model_configs[config.chosen_model_size].num_decoder_layers,
            d_model=config.model_configs[config.chosen_model_size].d_model,
            num_heads=config.model_configs[config.chosen_model_size].num_heads,
            d_ff=config.model_configs[config.chosen_model_size].d_ff,
            norm_type=config.model_configs[config.chosen_model_size].norm_type,
            dropout_prob=config.model_parameters.dropout,
            max_seq_len=config.tokenizer.max_seq_len,
        )

        model_state_dict = checkpoint["model"]
        first_layer_key = next(iter(model_state_dict.keys()))
        if first_layer_key.startswith("_orig_mod."):
            new_state_dict = {}
            prefix = "_orig_mod."
            for k, v in model_state_dict.items():
                new_k = k[len(prefix) :] if k.startswith(prefix) else k
                new_state_dict[new_k] = v
            model_state_dict = new_state_dict

        model.load_state_dict(model_state_dict, strict=True, assign=True)

        save_file(model.state_dict(), tmp_safetensor_path)

        if "model.safetensors" not in repo_files:
            api.upload_file(
                path_or_fileobj=tmp_safetensor_path,
                path_in_repo="model.safetensors",
                repo_id=repo_id,
                token=hf_token,
            )

        print("Successfully uploaded safetensor model.")

    # Clean config
    config.pop("wandb_organization", None)
    config.pop("output_dir", None)
    config.pop("base_path", None)
    config.pop("cache_ds_path", None)
    config.pop("s3_bucket_name", None)
    config.pop("ckpt_path_to_resume_from", None)
    config.pop("tokenizers_resume_path", None)

    # Upload config
    with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
        tmp_config_path = tmp.name

        with open(tmp_config_path, "w+") as f:
            OmegaConf.save(config, f)

        print("Uploading config...")
        api.upload_file(
            path_or_fileobj=tmp_config_path, path_in_repo="config-lock.yaml", repo_id=repo_id, token=hf_token
        )

    # Generate and upload README
    print("Generating README...")
    readme_content = generate_readme(config, repo_id.split("/")[-1])

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as tmp:
        tmp.write(readme_content)
    try:
        tmp_readme_path = tmp.name

        print("Uploading README...")
        api.upload_file(path_or_fileobj=tmp_readme_path, path_in_repo="README.md", repo_id=repo_id, token=hf_token)
    finally:
        os.remove(tmp.name)

    print(f"\nSuccessfully uploaded all files to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload PyTorch checkpoint to Hugging Face Hub")
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint file (local path or s3://bucket/key)")
    parser.add_argument(
        "--repo_id", type=str, default="giovo17/tfs-mt", help="Hugging Face repository ID (e.g., username/model-name)"
    )
    parser.add_argument("--private", default=True, help="Create a private repository")

    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")

    upload_to_huggingface(
        checkpoint_path=args.checkpoint_path,
        repo_id=args.repo_id,
        hf_token=hf_token,
        private=args.private,
    )
