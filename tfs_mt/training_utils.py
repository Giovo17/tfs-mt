import math
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime
from logging import Logger

import botocore
import ignite.distributed as idist
import torch
from ignite.engine import DeterministicEngine, Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, ProgressBar, global_step_from_engine
from ignite.handlers.early_stopping import EarlyStopping
from ignite.handlers.time_limit import TimeLimit
from ignite.metrics import GpuInfo
from ignite.metrics.metric import Metric
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DistributedSampler, Sampler

from .data_utils import WordTokenizer
from .ignite_custom_utils.checkpoint import BucketNotFoundError, S3Saver
from .ignite_custom_utils.common import setup_trackio_logging, setup_wandb_logging
from .ignite_custom_utils.trackio_logger import TrackioLogger
from .ignite_custom_utils.wandb_logger import WandBLogger


class CheckpointNotFoundError(Exception):
    def __init__(self, checkpoint_path):
        msg = f"Given {checkpoint_path!s} does not exist."
        super().__init__(msg)


class InvalidCheckpointS3PathError(Exception):
    def __init__(self, msg="checkpoint_s3_path must be of form s3://bucket/key"):
        super().__init__(msg)


class S3FailedDownloadError(Exception):
    def __init__(self, checkpoint_path, e):
        msg = f"Failed to download checkpoint from S3 ({checkpoint_path!s}): {e!s}"
        super().__init__(msg)


def resume_from_ckpt(
    checkpoint_path: str,
    to_load: Mapping | None = None,
    device: torch.device | None = None,
    logger: Logger | None = None,
    strict: bool = True,
    resume_tokenizers: bool = False,
    tokenizers_type: str | None = None,
) -> None | tuple[WordTokenizer, WordTokenizer]:
    """Loads state dict from a checkpoint file to resume the training or loads tokenizers.
    It supports loading from local or bucket s3 checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file (or folder) in S3 bucket or in filesystem.
        to_load (Mapping | None, optional): A dictionary with objects.. Defaults to None.
        device (torch.device | None, optional): Device. Defaults to None.
        logger (Logger | None, optional): To log info about resuming from a checkpoint. Defaults to None.
        strict (bool, optional): Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's `state_dict()` function. Defaults to True.
        resume_tokenizers (bool, optional): Whether to load only tokenizers. Defaults to False.
        tokenizers_type (str, optional): Tokenizers type (Word, BPE).

    Raises:
        CheckpointNotFoundError: Raised when checkpoint file doesn't exist.
        InvalidCheckpointS3PathError: Raised when bucket and file key are not correctly extracted from provided url.
        S3FailedDownloadError: Raised when download fails.

    Returns:
        None | tuple[WordTokenizer, WordTokenizer]: Pretrained tokenizers if resume_tokenizers. Otherwise None.
    """

    resume_method = "local"
    if checkpoint_path.startswith("s3://"):
        resume_method = "bucket-s3"

    if resume_method == "local":
        if not os.path.isfile(checkpoint_path):
            raise CheckpointNotFoundError(checkpoint_path)

        if resume_tokenizers:
            # Detect if the function checkpoint_path is the folder/bucket path or the pt export filepath
            ckpt_basepath = (
                "/".join(checkpoint_path.split("/")[:-1])
                if checkpoint_path.endswith((".pt", ".pth"))
                else checkpoint_path
            )
            src_tokenizer = WordTokenizer.from_pretrained(ckpt_basepath + f"/src_tokenizer_{tokenizers_type}.json")
            tgt_tokenizer = WordTokenizer.from_pretrained(ckpt_basepath + f"/tgt_tokenizer_{tokenizers_type}.json")
            return src_tokenizer, tgt_tokenizer

        checkpoint = torch.load(checkpoint_path, map_location=device)

        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint, strict=strict)
        logger.info("Successfully resumed from a local checkpoint: %s", checkpoint_path)

    else:  # resume_method == "bucket-s3":
        _, _, path = checkpoint_path.partition("s3://")
        bucket, _, key = path.partition("/")

        if bucket == "" or key == "":
            raise InvalidCheckpointS3PathError()

        src_tokenizer_key = key.split("/")[0] + f"/src_tokenizer_{tokenizers_type}.json" if resume_tokenizers else None
        tgt_tokenizer_key = key.split("/")[0] + f"/tgt_tokenizer_{tokenizers_type}.json" if resume_tokenizers else None

        s3 = S3Saver._make_s3_client()

        if resume_tokenizers:  # Download tokenizers to temperary files
            with (
                tempfile.NamedTemporaryFile(delete=False) as src_tmp,
                tempfile.NamedTemporaryFile(delete=False) as tgt_tmp,
            ):
                src_tmp_path = src_tmp.name
                tgt_tmp_path = tgt_tmp.name
            try:
                try:
                    s3.download_file(bucket, src_tokenizer_key, src_tmp_path)
                    s3.download_file(bucket, tgt_tokenizer_key, tgt_tmp_path)
                except Exception as e:
                    raise S3FailedDownloadError(checkpoint_path, e) from e

                src_tokenizer = WordTokenizer.from_pretrained(src_tmp_path)
                tgt_tokenizer = WordTokenizer.from_pretrained(tgt_tmp_path)
                logger.info(f"Successfully resumed tokenizers from a bucket s3 checkpoint: {bucket}")
                return src_tokenizer, tgt_tokenizer
            finally:
                with suppress(OSError):
                    os.remove(src_tmp_path)
                    os.remove(tgt_tmp_path)

        else:  # Download to_load to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
            try:
                try:
                    s3.download_file(bucket, key, tmp_path)
                except Exception as e:
                    raise S3FailedDownloadError(checkpoint_path, e) from e

                checkpoint = torch.load(tmp_path, map_location=device)

                Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint, strict=strict)
                logger.info(f"Successfully resumed training objects from a bucket s3 checkpoint: {checkpoint_path}")
            finally:
                with suppress(OSError):
                    os.remove(tmp_path)


def setup_output_dir(config: DictConfig | ListConfig, rank: int) -> str:
    """Create output folder."""
    output_dir = config.output_dir
    if rank == 0:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{now}-min_lr-{config.training_hp.lr_scheduler.min_lr}-max_lr-{config.training_hp.lr_scheduler.max_lr}"
        output_dir = os.path.join(config.output_dir, name)
        os.makedirs(output_dir, exist_ok=True)

    return idist.broadcast(output_dir, src=0)  # Path(idist.broadcast(output_dir, src=0))


def s3_upload(filepath: str, bucket: str, s3_key: str | None = None):
    """Upload a file on S3 bucket."""
    s3 = S3Saver._make_s3_client()
    s3_key = s3_key or filepath.split("/")[-1]  # Default key = filename

    # Verify bucket existence
    try:
        s3.head_bucket(Bucket=bucket)
    except botocore.exceptions.ClientError as exc:
        if exc.response["Error"]["Code"] == "404":
            raise BucketNotFoundError(bucket) from exec
        else:
            raise

    try:
        s3.upload_file(filepath, bucket, s3_key)
        print(f"Uploaded '{filepath}' to s3://{bucket}/{s3_key}")
    except Exception as exc:
        print(f"Failed to upload '{filepath}' to S3: {exc}")
        raise


def save_config(
    config: DictConfig | ListConfig,
    output_dir: str,
    enable_ckpt: bool = True,
):
    """Save configuration to config-lock.yaml for result reproducibility."""
    with open(f"{output_dir}/config-lock.yaml", "w+") as f:
        OmegaConf.save(config, f)
    # Upload to S3 endpoint
    if config.s3_bucket_name is not None and enable_ckpt:
        s3_upload(
            filepath=f"{output_dir}/config-lock.yaml",
            bucket=config.s3_bucket_name,
            s3_key=f"{config.model_name}/config-lock.yaml",
        )


def log_metrics(engine: Engine, tag: str) -> None:
    """Log `engine.state.metrics` with given `engine` and `tag`.

    Args:
        engine (Engine): Instance of `Engine` which metrics to log.
        tag (str): A string to add at the start of output.
    """

    metrics_format = f"{tag} [{engine.state.epoch}/{engine.state.iteration}]: {engine.state.metrics}"
    engine.logger.info(metrics_format)


def setup_exp_logging(
    config: DictConfig | ListConfig,
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer],
    evaluators: Engine | dict[str, Engine],
    return_all_loggers: bool = False,
) -> WandBLogger | tuple[WandBLogger, TrackioLogger]:
    """Setup Experiment Tracking with WandB and Trackio loggers.

    Using common.setup_wandb_logging which setup an ignite's Engine compatible WandB logger.
    It takes as kwargs wandb.init compatible arguments.
    Same for trackio.

    References:
    1. setup_wandb_logging documentation [[link](https://docs.pytorch.org/ignite/contrib/engines.html#ignite.contrib.engines.common.setup_wandb_logging)]
    2. WandBLogger documentation [[link](https://docs.pytorch.org/ignite/generated/ignite.handlers.wandb_logger.html#ignite.handlers.wandb_logger.WandBLogger)]
    3. wandb.init documentation [[link](https://docs.wandb.ai/ref/python/sdk/functions/init/)]
    4. trackio.init documentation [[link](https://huggingface.co/docs/trackio/en/api#trackio.init)]
    """
    wandb_logger = setup_wandb_logging(
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
    if not return_all_loggers:
        return wandb_logger

    os.environ["TRACKIO_DIR"] = (
        config.output_dir + "/trackio"
    )  # https://huggingface.co/docs/trackio/en/environment_variables#trackiodir

    trackio_logger = setup_trackio_logging(
        trainer,
        optimizers,
        evaluators,
        config.log_every_iters,
        # trackio.init kwargs
        project=config.model_base_name,
        name=config.model_name,
        config=config._content,
    )
    return wandb_logger, trackio_logger


def setup_handlers(
    trainer: Engine,
    evaluator: Engine,
    config: DictConfig | ListConfig,
    to_save_train: dict | None = None,
    to_save_test: dict | None = None,
    enable_ckpt: bool = True,
) -> None:
    """Setup Ignite handlers."""

    if enable_ckpt and idist.get_rank() == 0:  # Setup checkpointing only on rank 0
        # Setup checkpoints savers
        disk_saver = DiskSaver(os.path.join(config.output_dir, "checkpoints"), require_empty=False)
        s3_saver = (
            S3Saver(bucket=config.s3_bucket_name, prefix=config.model_name + "/")
            if config.s3_bucket_name is not None
            else None
        )

        # Training checkpointing.
        # Do it locally only if s3 checkpointing is disabled to save disk space in cloud instance.
        ckpt_handler_train = Checkpoint(
            to_save_train,
            save_handler=s3_saver if config.s3_bucket_name is not None else disk_saver,
            filename_prefix=config.model_base_name,
            n_saved=config.checkpoints_retain_n,
        )
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=config.save_every_iters),
            ckpt_handler_train,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), ckpt_handler_train)

        # Evaluation checkpointing.
        # Do it locally only if s3 checkpointing is disabled to save disk space in cloud instance.
        global_step_transform = None
        if to_save_train.get("trainer", None) is not None:
            global_step_transform = global_step_from_engine(to_save_train["trainer"])
        ckpt_handler_test = Checkpoint(
            to_save_test,
            save_handler=s3_saver if config.s3_bucket_name is not None else disk_saver,
            filename_prefix="best",
            n_saved=config.checkpoints_retain_n,
            global_step_transform=global_step_transform,
            score_name="test_bleu",
            score_function=Checkpoint.get_default_score_fn("Bleu"),
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), ckpt_handler_test)

    # Time limit reached policy to stop training. Mainly used in Kaggle due to 12 hours run limit.
    if config.time_limit_sec != -1:
        print(f"Setting up training time limit to {int(config.time_limit_sec) / 3600} hours.")
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TimeLimit(config.time_limit_sec))

    # Iterations and epochs progress bars
    ProgressBar(persist=True, bar_format="").attach(
        trainer, event_name=Events.EPOCH_STARTED, closing_event_name=Events.COMPLETED
    )
    ProgressBar(persist=False).attach(
        trainer, metric_names="all", event_name=Events.ITERATION_COMPLETED(every=config.update_pbar_every_iters)
    )

    if torch.cuda.is_available():
        GpuInfo().attach(trainer, name="gpu")


def setup_early_stopping(
    trainer: Engine,
    evaluator: Engine,
    config: DictConfig | ListConfig,
) -> None:
    """Setup early stopping."""

    def score_fn(engine: Engine):
        return engine.state.metrics["Bleu"]

    es = EarlyStopping(
        patience=config.training_hp.early_stopping.patience,  # Considered in number of iterations
        score_function=score_fn,
        trainer=trainer,
        min_delta=config.training_hp.early_stopping.min_delta,
    )
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, es)


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


def get_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, torch.nn.Parameter | float]]:
    """TODO Doc"""

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude biases, LayerNorm from weight decay regularization
        if name.endswith(".bias") or "layer_norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def setup_lr_lambda_fn(config: DictConfig | ListConfig):
    """Setup function that will govern learning rate scheduling.
    Following the DeepSeek V3 [implementation](https://arxiv.org/abs/2412.19437) there have been considered 3 phases:
    1. Warmup
    2. Stable
    3. Decay

    # TODO Continue

    Args:
        config (DictConfig | ListConfig): Project config file.
    """

    min_lr = config.training_hp.lr_scheduler.min_lr
    max_lr = config.training_hp.lr_scheduler.max_lr

    total_iters = config.num_train_iters_per_epoch * config.training_hp.num_epochs
    warmup_iters = config.training_hp.lr_scheduler.warmup_iters
    stable_iters = config.training_hp.lr_scheduler.stable_iters_prop * total_iters - warmup_iters
    decay_iters = total_iters - warmup_iters - stable_iters

    def lr_lambda(step):
        # Warmup phase
        if step < warmup_iters:
            return max_lr * ((step + 1) / warmup_iters)

        # Stable phase
        elif step < stable_iters + warmup_iters:
            return max_lr

        # Decay phase
        else:
            step = step - warmup_iters - stable_iters
            cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_iters))
            return min_lr + (max_lr - min_lr) * cosine_decay

    return lr_lambda


def setup_trainer(
    config: DictConfig | ListConfig,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    loss_fn: nn.Module,
    device: torch.device,
    train_sampler: Sampler,
) -> Engine | DeterministicEngine:
    """Setup a trainer with multigpu and mixed precision training support.

    Args:
        config (DictConfig | ListConfig): Project config file.
        model (nn.Module): Transformer model.
        optimizer (Optimizer): Optimizer.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device.
        train_sampler (Sampler): Torch data sampler. Use for multigpu training.

    Returns:
        Engine | DeterministicEngine: Trainer object.
    """

    # Gradient scaler for mixed precision training. Not required for bfloat16 training, cause it has the same range of float32.
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

        # Count how many tokens encoder and decoder see during training, excluding SOS and EOS tokens
        num_src_tokens = src_mask.to(torch.int8).sum().item() - 2 * src_mask.size(0)
        num_tgt_tokens = tgt_mask.to(torch.int8).sum().item() - 2 * tgt_mask.size(0)
        engine.state.tokens_seen_src = getattr(engine.state, "tokens_seen_src", 0) + num_src_tokens
        engine.state.tokens_seen_tgt = getattr(engine.state, "tokens_seen_tgt", 0) + num_tgt_tokens

        model.train()

        optimizer.zero_grad()

        # Mixed precision training if enabled in config. Reference: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        # autocast will automatically manage which operations to run in FP16 and which ones to run in FP32.
        # eg. matmul will cast to FP16 and it's a crucial part of the whole pipeline.
        # Here the complete list of FP16 supported modules: https://docs.pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16
        # NOTE Switching to torch.bfloat16 for better accuracy and same efficiency of float16, more on this here: https://www.cerebras.ai/blog/to-bfloat-or-not-to-bfloat-that-is-the-question
        with autocast(device.type, dtype=torch.bfloat16, enabled=config.training_hp.use_amp):
            output_logits = model(src_sequence, tgt_input_sequence, src_mask, tgt_mask)
            # output_logits shape: [B*S, V]  (B: batch size, S: sequence length, V: vocabulary size)
            # tgt_output_label shape: [B*S]
            loss = loss_fn(output_logits.reshape(-1, output_logits.size(-1)), tgt_output_label.reshape(-1))

        scaler.scale(loss).backward()

        if config.training_hp.max_gradient_norm > 0.0:  # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training_hp.max_gradient_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        metric = {
            "train_loss": loss.item(),
            "tokens_seen_src_cum": getattr(engine.state, "tokens_seen_src", 0),
            "tokens_seen_tgt_cum": getattr(engine.state, "tokens_seen_tgt", 0),
            "tokens_seen_tot_cum": getattr(engine.state, "tokens_seen_src", 0)
            + getattr(engine.state, "tokens_seen_tgt", 0),
            # "gradient_norm": getattr(engine.state, "gradient_norm", 0)
        }
        engine.state.metrics = metric

        return metric

    trainer = Engine(train_one_iter)

    # Set epoch for distributed sampler
    @trainer.on(Events.EPOCH_STARTED)
    def set_epoch():
        if idist.get_world_size() > 1 and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(trainer.state.epoch - 1)
        # Initialize token seen during training counters
        trainer.state.tokens_seen_src = 0
        trainer.state.tokens_seen_tgt = 0

    return trainer


def setup_evaluator(
    config: DictConfig | ListConfig,
    model: nn.Module,
    metrics: dict[str, Metric],
    device: torch.device,
) -> tuple[Engine | DeterministicEngine, Engine | DeterministicEngine]:
    """Setup an evaluator with mixed precision training support.

    Args:
        config (DictConfig | ListConfig): Project config file.
        model (nn.Module): Transformer model.
        metrics (dict[str, Metric]): Metrics to be used.
        device (torch.device): Device.

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

        with autocast(device.type, dtype=torch.bfloat16, enabled=config.training_hp.use_amp):
            output_logits = model(src_sequence, tgt_input_sequence, src_mask, tgt_mask)

        return output_logits, tgt_output_label

    train_evaluator = Engine(eval_one_iter)
    test_evaluator = Engine(eval_one_iter)

    for name, metric in metrics.items():
        metric.attach(train_evaluator, name)
    for name, metric in metrics.items():
        metric.attach(test_evaluator, name)

    return train_evaluator, test_evaluator
