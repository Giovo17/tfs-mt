# Transformer train script
#
# Author: Giovanni Spadaro - https://giovannispadaro.it
# Project: https://github.com/Giovo17/tfs-mt
# Documentation: https://giovo17.github.io/tfs-mt
#
# This script is licensed under the license found in the LICENSE file in the root directory of this source tree.

import argparse
import gc
import logging
import os
from datetime import datetime
from functools import partial
from pprint import pformat

import ignite.distributed as idist
import torch
from dotenv import load_dotenv
from ignite.engine import Events
from ignite.metrics import Bleu, Loss, Rouge
from ignite.utils import manual_seed, setup_logger
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profile, record_function
from torchinfo import summary

from tfs_mt.architecture import build_model
from tfs_mt.data_utils import build_data_utils
from tfs_mt.decoding_utils import greedy_decoding
from tfs_mt.training_utils import (
    get_param_groups,
    log_metrics,
    loss_metric_transform,
    nlp_metric_transform,
    resume_from_ckpt,
    s3_upload,
    save_config,
    setup_evaluator,
    setup_exp_logging,
    setup_handlers,
    setup_lr_lambda_fn,
    setup_output_dir,
    setup_trainer,
)

load_dotenv()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # remove tokenizer parallelism warning


# Torch profiler activities
activities = [ProfilerActivity.CPU]

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    activities += [ProfilerActivity.CUDA]


wandb_api_key = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = wandb_api_key

config_path = os.path.join(os.getcwd(), "tfs_mt/configs/config.yml")
base_path = os.getcwd()
output_dir = os.path.join(base_path, "data/output")
cache_ds_path = os.path.join(base_path, "data")
time_limit_sec = -1


class TooManyWarmupItersError(Exception):
    def __init__(self, warmup_iters, total_iters):
        msg = f"The number of warmup iterations cannot be greater than 50% the total number of iterations, \
        got warmup_iters: {warmup_iters}, total_iterations: {total_iters}"
        super().__init__(msg)


def run(local_rank, config, distributed=False, enable_log_ckpt=True):
    if distributed:
        rank = idist.get_rank()
        manual_seed(config.seed + rank)
        output_dir = setup_output_dir(config, rank)
        config.output_dir = output_dir

        idist.barrier()

        if rank == 0:
            save_config(config, output_dir, enable_ckpt=enable_log_ckpt)
    else:
        rank = 0
        manual_seed(config.seed)
        output_dir = setup_output_dir(config, 0)
        config.output_dir = output_dir
        save_config(config, config.output_dir, enable_ckpt=enable_log_ckpt)

    # Setup logger
    logger = setup_logger(level=logging.INFO, filepath=os.path.join(config.output_dir, "training-info.log"))
    logger.info("Configuration: \n%s", pformat(config))

    # Resume tokenizer from pretrained and build data utils
    if config.ckpt_path_to_resume_from is not None:
        src_tokenizer, tgt_tokenizer = resume_from_ckpt(
            config.ckpt_path_to_resume_from, logger=logger, resume_tokenizers=True
        )
        print("Tokenizers restored from pretrained.")
        train_dataloader, test_dataloader = build_data_utils(
            config, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer
        )

    else:  # Build data utils from scratch
        train_dataloader, test_dataloader, _, _, src_tokenizer, tgt_tokenizer = build_data_utils(
            config, return_all=True
        )

    # Save tokenizers
    src_tokenizer.to_json(config.output_dir + "/src_tokenizer.json")
    tgt_tokenizer.to_json(config.output_dir + "/tgt_tokenizer.json")
    if config.s3_bucket_name is not None and enable_log_ckpt:
        s3_upload(
            filepath=config.output_dir + "/src_tokenizer.json",
            bucket=config.s3_bucket_name,
            s3_key=f"{config.model_name}/src_tokenizer.json",
        )
        s3_upload(
            filepath=config.output_dir + "/tgt_tokenizer.json",
            bucket=config.s3_bucket_name,
            s3_key=f"{config.model_name}/tgt_tokenizer.json",
        )
        print(f"Uploaded tokenizers to s3://{config.s3_bucket_name}")

    config.num_train_iters_per_epoch = len(train_dataloader)

    # Raise exception if warmup iterations are more than 50% of total iterations
    if (
        config.training_hp.lr_scheduler.warmup_iters
        > 0.5 * config.training_hp.num_epochs * config.num_train_iters_per_epoch
    ):
        raise TooManyWarmupItersError(
            config.training_hp.lr_scheduler.warmup_iters,
            config.training_hp.num_epochs * config.num_train_iters_per_epoch,
        )

    # Initialize model, optimizer, loss function, device or resume from checkpoint
    device = idist.device()
    init_model = build_model(config, src_tokenizer, tgt_tokenizer)
    if torch.cuda.is_available():
        init_model = torch.compile(init_model, mode="max-autotune")
    try:
        logger.info(
            summary(
                init_model,
                [(16, 128), (16, 128), (16, 128), (16, 128)],
                dtypes=[torch.long, torch.long, torch.bool, torch.bool],
            )
        )
    except RuntimeError:
        logger.info(f"Total number of parameters: {sum(p.numel() for p in init_model.parameters())}")

    # Get model ready for multigpu training if available and move the model to current device
    model = idist.auto_model(init_model)

    if distributed:
        config.training_hp.lr_scheduler.min_lr *= idist.get_world_size()
        config.training_hp.lr_scheduler.max_lr *= idist.get_world_size()

    model_param_groups = get_param_groups(model, config.training_hp.optimizer.weight_decay)
    init_optimizer = AdamW(
        model_param_groups,
        lr=config.training_hp.lr_scheduler.min_lr,
        betas=(config.training_hp.optimizer.beta1, config.training_hp.optimizer.beta2),
    )
    optimizer = idist.auto_optim(init_optimizer)  # Get optimizer ready for multigpu training if available

    loss_fn = CrossEntropyLoss(
        # Ignore padding tokens in loss computation
        ignore_index=tgt_tokenizer.pad_token_idx,
        # This will average the loss over batch_size * sequence_length (pad tokens don't contribute to sequence_length)
        reduction="mean",
        # During training, we employed label smoothing of value 0.1 (Attention is all you need page 8)
        # This reduces overconfidence and improves generalization
        label_smoothing=config.training_hp.loss_label_smoothing,
    ).to(device=device)

    # Initialize learning rate scheduler
    lr_lambda = setup_lr_lambda_fn(config)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Setup metrics
    metrics = {
        "Bleu": Bleu(
            ngram=4, smooth="smooth1", output_transform=partial(nlp_metric_transform, tgt_tokenizer=tgt_tokenizer)
        ),
        "Bleu_smooth_2": Bleu(
            ngram=4, smooth="smooth2", output_transform=partial(nlp_metric_transform, tgt_tokenizer=tgt_tokenizer)
        ),
        "Rouge": Rouge(
            variants=["L", 2],
            multiref="best",
            output_transform=partial(nlp_metric_transform, tgt_tokenizer=tgt_tokenizer),
        ),
        "Loss": Loss(loss_fn, output_transform=loss_metric_transform),
    }

    # Setup trainer and evaluator
    trainer = setup_trainer(config, model, optimizer, lr_scheduler, loss_fn, device, train_dataloader.sampler)
    train_evaluator, test_evaluator = setup_evaluator(config, model, metrics, device)

    # Setup engines logger with python logging print training configurations
    trainer.logger = train_evaluator.logger = test_evaluator.logger = logger

    # Setup ignite handlers
    to_save_train = {
        "model": model,
        "optimizer": optimizer,
        "trainer": trainer,
        "lr_scheduler": lr_scheduler,
    }
    to_save_test = {"model": model}
    # When enable_log_ckpt is False the two returned ckpt handlers will be None
    setup_handlers(trainer, test_evaluator, config, to_save_train, to_save_test, enable_ckpt=enable_log_ckpt)

    # Time profiling
    # profiler = HandlersTimeProfiler()
    # profiler.attach(trainer)

    # Experiment tracking
    if enable_log_ckpt and rank == 0:
        print(config.model_name)
        exp_wandb_logger, exp_trackio_logger = setup_exp_logging(
            config,
            trainer,
            optimizer,
            evaluators={"train_eval": train_evaluator, "test_eval": test_evaluator},  # evaluator
            return_all_loggers=True,
        )

    # Print metrics to the stderr with "add_event_handler" method for training stats
    # More on ignite Events: https://docs.pytorch.org/ignite/generated/ignite.engine.events.Events.html
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.log_every_iters),
        log_metrics,
        tag="train",
    )

    # Run evaluators at every training epoch end using "on" decorator method and print metrics to the stderr
    # @trainer.on(Events.EPOCH_COMPLETED(every=1))
    # def run_train_eval():
    #     train_evaluator.run(train_dataloader)
    #     log_metrics(train_evaluator, "train")
    @trainer.on(Events.ITERATION_COMPLETED(every=config.eval_every_iters))
    def run_test_eval():
        test_evaluator.run(test_dataloader)
        log_metrics(test_evaluator, "test")

    # Run evaluator when trainer starts to make sure it works.
    @trainer.on(Events.STARTED)
    def run_test_eval_on_start():
        test_evaluator.run(test_dataloader)

    # Decode a sequence to debug training
    @test_evaluator.on(Events.EPOCH_COMPLETED(every=1))
    def run_decoding_debug():
        nr_sequences = 3
        sample_batch = next(test_dataloader.__iter__())
        decoded_seq_batch = greedy_decoding(
            config=config,
            model=model,
            encoder_representation=model.encode(
                sample_batch["src"][:nr_sequences].to(device), sample_batch["src_mask"][:nr_sequences].to(device)
            ),
            src_mask=sample_batch["src_mask"][:nr_sequences].to(device),
            tgt_tokenizer=tgt_tokenizer,
            max_target_tokens=config.tokenizer.max_seq_len,
        )
        test_evaluator.logger.info(f"Source sequences: \n{sample_batch['src_text'][:nr_sequences]}")
        test_evaluator.logger.info(f"Decoded sequences: \n{decoded_seq_batch}")

    # Clean GPU cache
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def cleanup_memory(engine):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # Log time profiling
    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_intermediate_results():
    #    profiler.print_results(profiler.get_results())

    # Resume from checkpoint if option available in config
    if config.ckpt_path_to_resume_from is not None:
        resume_from_ckpt(
            config.ckpt_path_to_resume_from,
            to_load=to_save_train,
            device=device,
            logger=logger,
            strict=True,
        )

    # Run and profile training (PyTorch official tutorial on profiling: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
    with (
        profile(activities=activities, profile_memory=True, record_shapes=True) as prof,
        record_function("model_training"),
    ):
        trainer.run(
            train_dataloader,
            max_epochs=config.training_hp.num_epochs,
        )
    prof.export_chrome_trace(os.path.join(config.output_dir, "trace.json"))
    if config.s3_bucket_name is not None and enable_log_ckpt:
        s3_upload(
            filepath=os.path.join(config.output_dir, "trace.json"),
            bucket=config.s3_bucket_name,
            s3_key=f"{config.model_name}/trace.json",
        )

    # Close loggers and upload local log file to s3 if configured
    if enable_log_ckpt and rank == 0:
        exp_wandb_logger.close()
        exp_trackio_logger.close()
        # profiler.write_results(config.output_dir + "/time_profiling.csv")
    if config.s3_bucket_name is not None and enable_log_ckpt:
        s3_upload(
            filepath=os.path.join(config.output_dir, "training-info.log"),
            bucket=config.s3_bucket_name,
            s3_key=f"{config.model_name}/training-info.log",
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer training arguments")
    parser.add_argument(
        "-e",
        "--exec_mode",
        choices=["dev", "dummy"],
        default="dev",
        help="Execution mode: 'dev' or 'dummy' (default: dev)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Execution mode: {args.exec_mode}")
    if args.exec_mode == "dummy":
        print("Distributed training, experiment logging and checkpointing are disabled!")

    config = OmegaConf.load(config_path)

    if config.dataset.max_len > 0 and config.train_dataloader.batch_size > int(
        config.dataset.max_len * config.dataset.train_split
    ):
        config.train_dataloader.batch_size = int(config.dataset.max_len * config.dataset.train_split)
    if config.dataset.max_len > 0 and config.test_dataloader.batch_size > int(
        config.dataset.max_len * (1 - config.dataset.train_split)
    ):
        config.test_dataloader.batch_size = int(config.dataset.max_len * (1 - config.dataset.train_split))

    # Enable/disable distributed training
    config.training_hp.distributed_training = bool(torch.cuda.is_available() and torch.cuda.device_count() >= 2)
    print(f"Distributed training availability: {config.training_hp.distributed_training}")
    config.backend = "nccl" if config.training_hp.distributed_training else "none"

    config.base_path = base_path
    config.output_dir = output_dir
    config.cache_ds_path = cache_ds_path
    config.chosen_model_size = "nano"  # nano, small, base
    config.time_limit_sec = time_limit_sec
    config.wandb_organization = os.getenv("WANDB_ORGANIZATION")

    config.model_name = f"{config.model_base_name}_{config.chosen_model_size}_{datetime.now().strftime('%y%m%d-%H%M')}"

    config.exec_mode = args.exec_mode

    if config.training_hp.distributed_training and args.exec_mode != "dummy":
        with idist.Parallel(config.backend) as p:
            p.run(run, config, distributed=True, enable_log_ckpt=True)
    elif args.exec_mode == "dummy":
        run(0, config, distributed=False, enable_log_ckpt=False)
    elif args.exec_mode == "dev":
        run(0, config, distributed=False, enable_log_ckpt=True)
