import logging
import os
from collections.abc import Mapping
from datetime import datetime
from logging import Logger

import ignite.distributed as idist
import torch
from ignite.contrib.engines import common
from ignite.engine import DeterministicEngine, Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.handlers.early_stopping import EarlyStopping
from ignite.handlers.time_limit import TimeLimit
from ignite.handlers.wandb_logger import WandBLogger
from ignite.metrics.metric import Metric
from ignite.utils import setup_logger
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.optimizer import Optimizer
from torch.utils.data import DistributedSampler, Sampler

from .data_utils import WordTokenizer


class CheckpointNotFoundError(Exception):
    def __init__(self, checkpoint_filepath):
        msg = f"Given {checkpoint_filepath!s} does not exist."
        super().__init__(msg)


def log_metrics(engine: Engine, tag: str) -> None:
    """Log `engine.state.metrics` with given `engine` and `tag`.

    Args:
        engine (Engine): Instance of `Engine` which metrics to log.
        tag (str): A string to add at the start of output.
    """

    metrics_format = f"{tag} [{engine.state.epoch}/{engine.state.iteration}]: {engine.state.metrics}"
    engine.logger.info(metrics_format)


def resume_from(
    to_load: Mapping,
    checkpoint_filepath: str,
    logger: Logger,
    strict: bool = True,
) -> None:
    """Loads state dict from a checkpoint file to resume the training.

    Args:
        to_load (Mapping): A dictionary with objects, e.g. {“model”: model, “optimizer”: optimizer, ...}
        checkpoint_filepath (str): Path to the checkpoint file.
        logger (Logger): To log info about resuming from a checkpoint.
        strict (bool, optional): Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's `state_dict()` function. Defaults to True.

    Raises:
        CheckpointNotFoundError: Raised when checkpoint file doesn't exist.
    """
    if not os.path.isfile(checkpoint_filepath):
        raise CheckpointNotFoundError(checkpoint_filepath)
    checkpoint = torch.load(checkpoint_filepath, map_location="cpu")

    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint, strict=strict)
    logger.info("Successfully resumed from a checkpoint: %s", checkpoint_filepath)


def setup_output_dir(config: DictConfig | ListConfig, rank: int) -> str:
    """Create output folder."""
    output_dir = config.output_dir
    if rank == 0:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{now}-backend-{config.backend}-lr-{config.training_hp.optimizer_args.learning_rate}"
        output_dir = os.path.join(config.output_dir, name)
        os.makedirs(output_dir, exist_ok=True)

    return idist.broadcast(output_dir, src=0)  # Path(idist.broadcast(output_dir, src=0))


def save_config(config: DictConfig | ListConfig, output_dir: str):
    """Save configuration to config-lock.yaml for result reproducibility."""
    with open(f"{output_dir}/config-lock.yaml", "w+") as f:
        OmegaConf.save(config, f)


def setup_logging(config: DictConfig | ListConfig) -> Logger:
    """Setup logger with `ignite.utils.setup_logger()`.

    Args:
        config (DictConfig | ListConfig): Config object. config has to contain `verbose` and `output_dir` attribute.

    Returns:
        Logger: n instance of `Logger`.
    """

    green = "\033[32m"
    reset = "\033[0m"
    logger = setup_logger(
        name=f"{green}[ignite]{reset}",
        level=logging.INFO,
        filepath=os.path.join(config.output_dir, "training-info.log"),
    )
    return logger


def setup_exp_logging(
    config: DictConfig | ListConfig,
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer],
    evaluators: Engine | dict[str, Engine],
) -> WandBLogger:
    """Setup Experiment Tracking WandB logger from Ignite.

    Using common.setup_wandb_logging which setup an ignite's Engine compatible WandB logger.
    It takes as kwargs wandb.init compatible arguments.

    References:
    1. setup_wandb_logging documentation [[link](https://docs.pytorch.org/ignite/contrib/engines.html#ignite.contrib.engines.common.setup_wandb_logging)]
    2. WandBLogger documentation [[link](https://docs.pytorch.org/ignite/generated/ignite.handlers.wandb_logger.html#ignite.handlers.wandb_logger.WandBLogger)]
    3. wandb.init documentation [[link](https://docs.wandb.ai/ref/python/sdk/functions/init/)]
    """
    logger = common.setup_wandb_logging(
        trainer,
        optimizers,
        evaluators,
        config.log_every_iters,
        # wandb.init kwargs
        entity=config.wandb_organization,
        project=config.model_base_name,
        name=config.model_name,
        config=config._content,
        tags=["pytorch", "pytorch", "nlp", "machine-translation"],
    )
    return logger


def setup_handlers(
    trainer: Engine,
    evaluator: Engine,
    config: DictConfig | ListConfig,
    to_save_train: dict | None = None,
    to_save_test: dict | None = None,
) -> tuple[Checkpoint, Checkpoint]:
    """Setup Ignite handlers."""

    ckpt_handler_train = ckpt_handler_test = None

    if idist.get_rank() == 0:  # Setup checkpointing only on rank 0
        # Checkpointing
        saver = DiskSaver(os.path.join(config.output_dir, "checkpoints"), require_empty=False)
        ckpt_handler_train = Checkpoint(
            to_save_train,
            saver,
            filename_prefix=config.model_base_name,
            n_saved=config.checkpoints_retain_n,
        )
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=config.save_every_iters),
            ckpt_handler_train,
        )

        global_step_transform = None
        if to_save_train.get("trainer", None) is not None:
            global_step_transform = global_step_from_engine(to_save_train["trainer"])
        ckpt_handler_test = Checkpoint(
            to_save_test,
            saver,
            filename_prefix="best",
            n_saved=config.checkpoints_retain_n,
            global_step_transform=global_step_transform,
            score_name="test_bleu",
            score_function=Checkpoint.get_default_score_fn("Bleu"),
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), ckpt_handler_test)

    # Early stopping
    def score_fn(engine: Engine):
        return engine.state.metrics["Bleu"]

    es = EarlyStopping(config.training_hp.early_stopping_patience, score_fn, trainer)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, es)

    # Time limit reached policy to stop training. Mainly used in Kaggle due to 12 hours run limit.
    if config.time_limit_sec != -1:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TimeLimit(config.time_limit_sec))

    return ckpt_handler_train, ckpt_handler_test


def nlp_metric_transform(
    output: tuple[torch.Tensor, torch.Tensor], tgt_tokenizer: WordTokenizer
) -> tuple[list[list[str]], list[list[list[str]]]]:
    """Transform `eval_one_iter` output to be compliant with ignite nlp metrics.

    References:
    1. Bleu documentation page [[link](https://docs.pytorch.org/ignite/generated/ignite.metrics.Bleu.html)]
    2. Rouge documentation page [[link](https://docs.pytorch.org/ignite/generated/ignite.metrics.Rouge.html)]

    Args:
        output (tuple[torch.Tensor, torch.Tensor]): Output of `eval_one_iter`.
        tgt_tokenizer (WordTokenizer): Target tokenizer used to decode tokens.

    Returns:
        tuple[list[list[str]], list[list[list[str]]]]: Metrics complatible output.
    """

    output_logits, tgt_output_label = output

    # Get predicted tokens from logits
    output_logits = output_logits.detach()
    output_probs = torch.softmax(output_logits, dim=-1)
    output_tokens = torch.argmax(output_probs, dim=-1)

    # Move to list of int for tokenizer.decode compatibility
    output_tokens = output_tokens.cpu().numpy().tolist()
    tgt_output_label = tgt_output_label.cpu().numpy().tolist()

    # Decode batched token sequences to lists of lists of vocab token sequences
    y_pred = [tgt_tokenizer.decode(sample) for sample in output_tokens]
    y = [tgt_tokenizer.decode(sample) for sample in tgt_output_label]

    # Adjust shape. Ignite wants a corpus of lists of target label sentences for each hypotheses.
    # Since the dataset proposes only one target translation for a given input, y is wrapped in a list.
    y = [y]

    return y_pred, y


def loss_metric_transform(output: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform `eval_one_iter` output to be compliant with torch loss computation.

    Args:
        output (tuple[torch.Tensor, torch.Tensor]): Output of `eval_one_iter`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Loss compatible output.
    """

    output_logits, tgt_output_label = output
    return output_logits.reshape(-1, output_logits.size(-1)), tgt_output_label.reshape(-1)


def setup_trainer(
    config: DictConfig | ListConfig,
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: str | torch.device,
    train_sampler: Sampler,
) -> Engine | DeterministicEngine:
    """Setup a trainer with multigpu and mixed precision training support.

    Args:
        config (DictConfig | ListConfig): Project config file.
        model (nn.Module): Transformer model.
        optimizer (Optimizer): Optimizer.
        loss_fn (nn.Module): Loss function.
        device (str | torch.device): Device.
        train_sampler (Sampler): Torch data sampler. Use for multigpu training.

    Returns:
        Engine | DeterministicEngine: Trainer object.
    """

    # Gradient scaler for mixed precision training.
    # It helps prevent gradients with small magnitudes from underflowing when training with mixed precision.
    scaler = GradScaler(device, enabled=config.training_hp.use_amp)

    def train_one_iter(
        engine: Engine | DeterministicEngine, batch: dict[str, torch.Tensor | str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # non_blocking asynchronously transfers tensor from CPU to device. More here: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
        src_sequence = batch["src"].to(device, non_blocking=True, dtype=torch.long)
        tgt_sequence = batch["tgt"].to(device, non_blocking=True, dtype=torch.long)
        src_mask = batch["src_mask"].to(device, non_blocking=True, dtype=torch.long)
        # Mask is not shrinked accordingly to tgt_sequence here. It will be handled during attention processing.
        tgt_mask = batch["tgt_mask"].to(device, non_blocking=True, dtype=torch.long)  # [:, :-1]

        # Shifted target sequence as label for teacher forcing. Reshape to 1D tensor to later compute loss
        tgt_output_label = tgt_sequence[:, 1:]

        tgt_input_sequence = tgt_sequence[:, :-1]

        model.train()

        optimizer.zero_grad()

        # Mixed precision training if enabled in config. Reference: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        # autocast will automatically manage which operations to run in FP16 and which ones to run in FP32.
        # eg. matmul will cast to FP16 and it's a crucial part of the whole pipeline.
        # Here the complete list of FP16 supported modules: https://docs.pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16
        with autocast(device.type, dtype=torch.float16, enabled=config.training_hp.use_amp):
            output_logits = model(src_sequence, tgt_input_sequence, src_mask, tgt_mask)
            # output_logits shape: [B*S, V]  (B: batch size, S: sequence length, V: vocabulary size)
            # tgt_output_label shape: [B*S]
            loss = loss_fn(output_logits.reshape(-1, output_logits.size(-1)), tgt_output_label.reshape(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metric = {"train_loss": loss.item()}
        engine.state.metrics = metric

        return metric

    trainer = Engine(train_one_iter)

    # Set epoch for distributed sampler
    @trainer.on(Events.EPOCH_STARTED)
    def set_epoch():
        if idist.get_world_size() > 1 and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(trainer.state.epoch - 1)

    return trainer


def setup_evaluator(
    config: DictConfig | ListConfig,
    model: nn.Module,
    loss_fn: nn.Module,
    metrics: dict[str, Metric],
    device: str | torch.device,
) -> tuple[Engine | DeterministicEngine, Engine | DeterministicEngine]:
    """Setup an evaluator with mixed precision training support.

    Args:
        config (DictConfig | ListConfig): Project config file.
        model (nn.Module): Transformer model.
        metrics (dict[str, Metric]): Metrics to be used.
        device (str | torch.device): Device.

    Returns:
        tuple[Engine | DeterministicEngine, Engine | DeterministicEngine]: Evaluator objects.
    """

    # Gradient scaler is not required during evaluation.
    # https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html#inference-evaluation

    @torch.no_grad()
    def eval_one_iter(engine: Engine, batch: dict[str, torch.Tensor | str]) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE See train_one_iter function for code explanation

        src_sequence = batch["src"].to(device, non_blocking=True, dtype=torch.long)
        tgt_sequence = batch["tgt"].to(device, non_blocking=True, dtype=torch.long)
        src_mask = batch["src_mask"].to(device, non_blocking=True, dtype=torch.long)
        tgt_mask = batch["tgt_mask"].to(device, non_blocking=True, dtype=torch.long)

        tgt_output_label = tgt_sequence[:, 1:]

        tgt_input_sequence = tgt_sequence[:, :-1]

        model.eval()

        with autocast(device.type, enabled=config.training_hp.use_amp):
            output_logits = model(src_sequence, tgt_input_sequence, src_mask, tgt_mask)

        return output_logits, tgt_output_label

    train_evaluator = Engine(eval_one_iter)
    test_evaluator = Engine(eval_one_iter)

    for name, metric in metrics.items():
        metric.attach(train_evaluator, name)
    for name, metric in metrics.items():
        metric.attach(test_evaluator, name)

    return train_evaluator, test_evaluator
