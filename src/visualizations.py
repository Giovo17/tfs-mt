# Visualizations script
#
# Author: Giovanni Spadaro - https://giovannispadaro.it
# Project: https://github.com/Giovo17/tfs-mt
# Documentation: https://giovo17.github.io/tfs-mt
#
# Copyright (c) Giovanni Spadaro.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from tfs_mt.architecture import build_model
from tfs_mt.data_utils import WordTokenizer
from tfs_mt.embeddings import SinusoidalPositionalEncoding
from tfs_mt.training_utils import KLDivLabelSmoothingLoss, setup_lr_lambda_fn


def load_model_and_tokenizers():
    base_url = "https://huggingface.co/giovo17/tfs-mt/resolve/main/"
    src_tokenizer = WordTokenizer.from_pretrained(base_url + "src_tokenizer_word.json")
    tgt_tokenizer = WordTokenizer.from_pretrained(base_url + "tgt_tokenizer_word.json")

    model = build_model(
        config=base_url + "config-lock.yaml",
        from_pretrained=True,
        model_path=base_url + "model.safetensors",
    )
    model.eval()

    return model, src_tokenizer, tgt_tokenizer


def display_attention_matrix(
    selected_layer: int = -1, selected_attention_head: int = 0, pad_len: int = 9, annot_threshold: float = 0.05
):
    model, src_tokenizer, _ = load_model_and_tokenizers()

    example_en_string = "The Transformer architecture was introduced in 2017"

    tokens = src_tokenizer.tokenize(example_en_string)

    token_ids, mask = src_tokenizer.encode(tokens, pad_to_len=pad_len)

    # Reconstruct display tokens to include the padding tokens.
    # NOTE pad_to_len in tokenizer.encode() doesn't consider the SOS and EOS tokens
    display_tokens = tokens + ["<pad>"] * (pad_len - len(tokens) + 2)

    # Run the encode method just to populate attention weights in the encoder
    src_sequence = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    src_mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
    with torch.inference_mode():
        _ = model.encode(src_sequence, src_mask)

    # Select the attention weights of the last encoder layer
    attn_weights = model.encoder[selected_layer].self_attention.attn_weights
    # Select the head and the first and only element in the batch
    attn_weights = attn_weights[0, selected_attention_head, :, :]
    # The last rows of the attention matrix associated with padding results in nan values.
    # Replacing them with zeros for better visualization
    attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

    # Attention mask has shape [B, 1, 1, S] cause the mask is the across every head
    attn_mask = model.attention_mask[0, 0, :, :]

    attn_weights = pd.DataFrame(attn_weights.detach().numpy())

    # Display only values higher than annot_threshold for better visualization
    annot = attn_weights.map(lambda v: f"{v:.2f}" if v > annot_threshold else "")

    _, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        attn_weights,
        cmap="viridis",
        annot=annot,
        square=True,
        vmin=0,
        vmax=1,
        fmt="",
        xticklabels=display_tokens,
        yticklabels=display_tokens,
    )

    # Add red borders where mask is True
    for i in range(attn_mask.shape[0]):
        for j in range(attn_mask.shape[1]):
            if not attn_mask[i, j]:  # The mask values are true where attention has to be computed
                ax.plot([j, j + 1], [i, i + 1], color="red", linewidth=1)
                ax.plot([j + 1, j], [i, i + 1], color="red", linewidth=1)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "docs/assets/img/attention_matrix.png"))
    plt.show()


def display_causal_attention_matrix(
    selected_layer: int = -1, selected_attention_head: int = 0, pad_len: int = 9, annot_threshold: float = 0.05
):
    model, src_tokenizer, tgt_tokenizer = load_model_and_tokenizers()

    example_en_string = "The Transformer architecture was introduced in 2017"
    example_it_string = "L'architettura Transformer è stata introdotta nel 2017"

    src_tokens = src_tokenizer.tokenize(example_en_string)
    src_token_ids, src_mask = src_tokenizer.encode(src_tokens, pad_to_len=pad_len)

    tgt_tokens = tgt_tokenizer.tokenize(example_it_string)

    tgt_token_ids, tgt_mask = tgt_tokenizer.encode(tgt_tokens, pad_to_len=pad_len)  # pad_to_len=len(tgt_tokens_list)+2)

    display_tokens = tgt_tokens + ["<pad>"] * (pad_len - len(tgt_tokens) + 2)

    # Run the decode method just to populate attention weights in the encoder
    src_sequence = torch.tensor(src_token_ids, dtype=torch.long).unsqueeze(0)
    src_mask_tensor = torch.tensor(src_mask, dtype=torch.bool).unsqueeze(0)
    tgt_sequence = torch.tensor(tgt_token_ids, dtype=torch.long).unsqueeze(0)
    tgt_mask_tensor = torch.tensor(tgt_mask, dtype=torch.bool).unsqueeze(0)
    with torch.inference_mode():
        encoder_representation = model.encode(src_sequence, src_mask_tensor)
        _ = model.decode(tgt_sequence, encoder_representation, tgt_mask_tensor, src_mask_tensor)

    # Select the attention weights of the last decoder layer
    attn_weights = model.decoder[selected_layer].self_attention.attn_weights
    # Select the head and the first and only element in the batch
    attn_weights = attn_weights[0, selected_attention_head, :, :]
    attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

    # Attention mask has shape [B, 1, 1, S] cause the mask is the across every head
    causal_attn_mask = model.causal_attention_mask[0, 0, :, :]

    attn_weights = pd.DataFrame(attn_weights.detach().numpy())

    annot = attn_weights.map(lambda v: f"{v:.2f}" if v > annot_threshold else "")

    _, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        attn_weights,
        cmap="viridis",
        annot=annot,
        square=True,
        vmin=0,
        vmax=1,
        fmt="",
        xticklabels=display_tokens,
        yticklabels=display_tokens,
    )

    # Add red borders where mask is True
    for i in range(causal_attn_mask.shape[0]):
        for j in range(causal_attn_mask.shape[1]):
            if not causal_attn_mask[i, j]:  # The mask values are true where attention has to be computed
                ax.plot([j, j + 1], [i, i + 1], color="red", linewidth=1)
                ax.plot([j + 1, j], [i, i + 1], color="red", linewidth=1)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "docs/assets/img/causal_attention_matrix.png"))
    plt.show()


def display_cross_attention_matrix(
    selected_layer: int = -1,
    selected_attention_head: int = 0,
    src_text: str = "The Transformer architecture was introduced in 2017",
    tgt_text: str = "L'architettura Transformer è stata introdotta nel 2017",
    pad_len: int = 9,
    annot_threshold: float = 0.05,
    save_path: str = "cross_attention_matrix.png",
):
    model, src_tokenizer, tgt_tokenizer = load_model_and_tokenizers()

    src_tokens = src_tokenizer.tokenize(src_text)
    src_token_ids, src_mask = src_tokenizer.encode(src_tokens, pad_to_len=pad_len)

    tgt_tokens = tgt_tokenizer.tokenize(tgt_text)

    tgt_token_ids, tgt_mask = tgt_tokenizer.encode(tgt_tokens, pad_to_len=pad_len)

    src_display_tokens = src_tokens + ["<pad>"] * (pad_len - len(src_tokens) + 2)
    tgt_display_tokens = tgt_tokens + ["<pad>"] * (pad_len - len(tgt_tokens) + 2)

    # Run the decode method just to populate attention weights in the encoder
    src_sequence = torch.tensor(src_token_ids, dtype=torch.long).unsqueeze(0)
    src_mask_tensor = torch.tensor(src_mask, dtype=torch.bool).unsqueeze(0)
    tgt_sequence = torch.tensor(tgt_token_ids, dtype=torch.long).unsqueeze(0)
    tgt_mask_tensor = torch.tensor(tgt_mask, dtype=torch.bool).unsqueeze(0)
    with torch.inference_mode():
        encoder_representation = model.encode(src_sequence, src_mask_tensor)
        _ = model.decode(tgt_sequence, encoder_representation, tgt_mask_tensor, src_mask_tensor)

    # Select the attention weights of the last decoder layer
    attn_weights = model.decoder[selected_layer].cross_attention.attn_weights
    # Select the head and the first and only element in the batch
    attn_weights = attn_weights[0, selected_attention_head, :, :]
    attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

    # Attention mask has shape [B, 1, 1, S] cause the mask is the across every head
    cross_attn_mask = model.cross_attention_mask[0, 0, :, :]

    attn_weights = pd.DataFrame(attn_weights.detach().numpy())

    annot = attn_weights.map(lambda v: f"{v:.2f}" if v > annot_threshold else "")

    _, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        attn_weights,
        cmap="viridis",
        annot=annot,
        square=True,
        vmin=0,
        vmax=1,
        fmt="",
        xticklabels=src_display_tokens,
        yticklabels=tgt_display_tokens,
    )

    # Add red borders where mask is True
    for i in range(cross_attn_mask.shape[0]):
        for j in range(cross_attn_mask.shape[1]):
            if not cross_attn_mask[i, j]:  # The mask values are true where attention has to be computed
                ax.plot([j, j + 1], [i, i + 1], color="red", linewidth=1)
                ax.plot([j + 1, j], [i, i + 1], color="red", linewidth=1)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "docs/assets/img/", save_path))
    plt.show()


def display_positional_encoding():
    pos_encoder = SinusoidalPositionalEncoding(
        d_model=16,
        max_sequence_length=500,
    )

    pe_lut: torch.Tensor = pos_encoder.pe_lut
    pe = pe_lut.numpy().transpose()

    plt.figure(figsize=(10, 6))
    sns.heatmap(pe, cmap="RdBu", center=0, annot=False, yticklabels=True, xticklabels=100)
    plt.xlabel("Sequence position")
    plt.ylabel("Embedding dimension")
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "docs/assets/img/positional_encoding.png"))
    plt.show()


def display_sinusoidal_signals():
    fig, axes = plt.subplots(8, 1, figsize=(10, 6), sharex=True)

    t = np.linspace(0, 1024, num=1024)

    pos_encoder = SinusoidalPositionalEncoding(
        d_model=8,
        max_sequence_length=1024,
    )

    x_pin_index = 407

    for _, (ax, i) in enumerate(zip(axes, range(8), strict=False)):
        signal = pos_encoder.pe_lut[:, i]

        ax.plot(t, signal)

        y_label = round(signal.numpy()[x_pin_index], 2)
        ax.set_ylabel(y_label, rotation=0, labelpad=20, fontsize=20, color="black", weight="bold")
        ax.yaxis.set_label_coords(-0.05, 0.10)

        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(0)

        ax.grid(False)

    # Add vertical line to show the values of this pseudo positional embedding at example index
    for ax in axes:
        ax.axvline(x=x_pin_index, color="#000000", linewidth=3, linestyle="-", alpha=0.8)

    plt.xlim(0, 1024)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "docs/assets/img/sinusoidal_signals.png"))
    plt.show()


def display_square_wave_signals():
    fig, axes = plt.subplots(8, 1, figsize=(10, 6), sharex=True)

    periods = [8, 16, 32, 64, 128, 256, 512, 1024]
    t = np.linspace(0, 1024, num=1024)
    x_pin_index = 407

    for _, (ax, period) in enumerate(zip(axes, periods, strict=False)):
        # Square wave signal
        signal = 1 - (np.mod(t, period) < period / 2).astype(int)

        ax.step(t, signal, where="post", linewidth=1.5)
        ax.fill_between(t, 0, signal, step="post", alpha=0.1)

        ax.set_ylabel(signal[x_pin_index], rotation=0, labelpad=20, fontsize=20, color="black", weight="bold")
        ax.yaxis.set_label_coords(-0.05, 0.10)

        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(0)

        ax.grid(False)

    # Add vertical line to show the values of this pseudo positional embedding at example index
    for ax in axes:
        ax.axvline(x=x_pin_index, color="#000000", linewidth=3, linestyle="-", alpha=0.8)

    plt.xlim(0, 1024)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "docs/assets/img/square_wave_signals.png"))
    plt.show()


def display_lr_schedule_wsd(config):
    config.training_hp.lr_scheduler.type = "wsd"

    batch_size = 16
    input_dim = 10

    def get_batch():
        x = torch.randn(batch_size, input_dim)
        y = torch.randint(0, 2, (batch_size,)).long()
        return x, y

    # Dummy model
    model = nn.Linear(input_dim, 2)
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1)

    lr_lambda = setup_lr_lambda_fn(config)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    lrs = []
    global_step = 0
    for _epoch in range(config.training_hp.num_epochs):
        for _ in range(config.num_train_iters_per_epoch):
            x, y = get_batch()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            lrs.append(optimizer.param_groups[0]["lr"])

            scheduler.step()
            global_step += 1

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=list(range(len(lrs))), y=lrs)
    plt.xlabel("Step")
    plt.ylabel("Learning rate")
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), f"/docs/assets/img/lr_scheduling_{config.training_hp.lr_scheduler.type}.png"))
    plt.show()


def display_lr_schedule_original(config):
    config.training_hp.lr_scheduler.type = "original"

    batch_size = 16
    input_dim = 5

    def get_batch():
        x = torch.randn(batch_size, input_dim)
        y = torch.randint(0, 2, (batch_size,)).long()
        return x, y

    # Dummy model
    model = nn.Linear(input_dim, 2)
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1)

    plt.figure(figsize=(10, 5))

    graph_configs = [(100, 10000), (256, 10000), (256, 15000)]
    for d_model, warmup_iters in graph_configs:
        config.model_configs[config.chosen_model_size].d_model = d_model
        config.training_hp.lr_scheduler.warmup_iters = warmup_iters

        lr_lambda = setup_lr_lambda_fn(config)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        lrs = []
        global_step = 0
        for _epoch in range(config.training_hp.num_epochs):
            for _ in range(config.num_train_iters_per_epoch):
                x, y = get_batch()
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                lrs.append(optimizer.param_groups[0]["lr"])

                scheduler.step()
                global_step += 1

        sns.lineplot(x=list(range(len(lrs))), y=lrs, label=f"d_model={d_model}, warmup={warmup_iters}")

    plt.xlabel("Step")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), f"/docs/assets/img/lr_scheduling_{config.training_hp.lr_scheduler.type}.png"))
    plt.show()


def display_label_smoothing(config):
    criterion = KLDivLabelSmoothingLoss(vocab_size=5, padding_idx=1, smoothing=config.training_hp.loss.label_smoothing)

    outputs = torch.FloatTensor([
        [0, 0.2, 0.7, 0.1, 0],
        [0, 0.2, 0.7, 0.1, 0],
        [0, 0.2, 0.7, 0.1, 0],
        [0, 0.2, 0.7, 0.1, 0],
        [0, 0.2, 0.7, 0.1, 0],
    ])

    pred = outputs.log_softmax(-1)
    target = torch.LongTensor([2, 1, 0, 3, 3])

    _, true_dist = criterion(pred, target, return_true_dist=True)

    LS_data = pd.concat([
        pd.DataFrame({
            "target distribution": true_dist[x, y].flatten(),
            "columns": y,
            "rows": x,
        })
        for y in range(5)
        for x in range(5)
    ])

    heatmap_data = LS_data.pivot(index="rows", columns="columns", values="target distribution")

    plt.figure(figsize=(6, 6))
    sns.heatmap(heatmap_data, cmap="viridis", cbar_kws={"label": "target distribution"}, square=True)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "/docs/assets/img/label_smoothing.png"))
    plt.show()


def display_training_loss(config, csv_path):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))

    # Smoothed loss with Exponential Moving Average
    df["smoothed_loss"] = df["training/train_loss"].ewm(span=20).mean()

    sns.lineplot(data=df, x="_step", y="training/train_loss", label="Loss", color="blue")
    # Smoothed loss
    sns.lineplot(data=df, x="_step", y="smoothed_loss", label="Loss (EMA)", color="blue", alpha=0.5)

    # Add vertical lines for epoch boundaries
    iters_per_epoch = config.num_train_iters_per_epoch
    max_step = df["_step"].max()
    for i in range(1, int(max_step // iters_per_epoch) + 1):
        plt.axvline(x=i * iters_per_epoch, color="red", linestyle="--", alpha=0.2)

    plt.title("Training loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "/docs/assets/img/training_loss.png"))
    plt.show()


def display_performance_metrics(csv_path):
    df = pd.read_csv(csv_path)

    metrics_map = {
        "test_eval/Bleu": "BLEU",
        # "test_eval/Bleu_smooth_2": "BLEU_smooth_2",
        "test_eval/Rouge/Rouge-L-F": "ROUGE-L-F",
        # "test_eval/Rouge/Rouge-L-P": "ROUGE-L-P",
        # "test_eval/Rouge/Rouge-L-R": "ROUGE-L-R",
        "test_eval/Rouge/Rouge-2-F": "ROUGE-2-F",
        # "test_eval/Rouge/Rouge-2-P": "ROUGE-2-P",
        # "test_eval/Rouge/Rouge-2-R": "ROUGE-2-R",
    }

    available_metrics = list(metrics_map)

    # Melt the dataframe to have a metric column that later will be used as hue
    melted_df = df.melt(id_vars=["_step"], value_vars=available_metrics, var_name="Metric", value_name="Score")
    melted_df = melted_df.dropna(subset=["Score"])

    melted_df["Metric"] = melted_df["Metric"].map(metrics_map)  # Map original column names to display labels

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=melted_df, x="_step", y="Score", hue="Metric", marker="o")

    plt.title("Performance metrics")
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.xlim(left=-2500)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), "/docs/assets/img/performance_metrics.png"))
    plt.show()


if __name__ == "__main__":
    config_path = os.path.join(os.getcwd(), "src/tfs_mt/configs/config.yml")
    config = OmegaConf.load(config_path)

    sns.set_context("talk", font_scale=0.95, rc={"font.size": 14})

    os.makedirs(os.path.join(os.getcwd(), "docs/assets/img"), exist_ok=True)

    # Attention matrix ------------------------------------------------------------------------------------------------

    config.chosen_model_size = "small"

    selected_layer = -1
    selected_attention_head = 1

    display_attention_matrix(selected_layer, selected_attention_head)
    display_causal_attention_matrix(selected_layer, selected_attention_head)
    display_cross_attention_matrix(selected_layer, selected_attention_head)

    # Positional Encoding ---------------------------------------------------------------------------------------------

    display_positional_encoding()
    display_sinusoidal_signals()
    display_square_wave_signals()

    # LR scheduler ----------------------------------------------------------------------------------------------------

    config.num_train_iters_per_epoch = 28889
    config.training_hp.num_epochs = 2

    display_lr_schedule_original(config)
    display_lr_schedule_wsd(config)

    # Loss ------------------------------------------------------------------------------------------------------------

    config.training_hp.loss.label_smoothing = 0.2

    display_label_smoothing(config)

    # Experiments results graphs --------------------------------------------------------------------------------------

    results_csv = os.path.join(os.getcwd(), "/data/eval/tfs_mt_small_251104-1748.csv")

    display_training_loss(config, results_csv)
    display_performance_metrics(results_csv)
