# Transformer train script
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
import gc
import logging
import os
from datetime import datetime
from functools import partial
from pprint import pformat

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
    KLDivLabelSmoothingLoss,
    get_device,
    get_param_groups,
    log_metrics,
    loss_metric_transform,
    nlp_metric_transform,
    resume_from_ckpt,
    s3_upload,
    save_config,
    setup_early_stopping,
    setup_evaluator,
    setup_exp_logging,
    setup_handlers,
    setup_lr_lambda_fn,
    setup_output_dir,
    setup_trainer,
)

load_dotenv()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # remove tokenizer parallelism warning


# Torch profiler activities
activities = [ProfilerActivity.CPU]

if torch.cuda.is_available():
    # Consider if the batches aren't aligned with the same sequence length.
    # If so, when compiling the model CUDA creates separate cudagraphs for each different input shape.
    # Set to True to avoid this overhead when having batched of different input lengths.
    # Set to False to create cudagraphs and get speedup, but ensure the batches have all the same sequence length to avoid cudagraph recomputation overhead.
    # NOTE The best solution may be in the middle:
    #   First of all extract K different sequence lengths such that a good samples clusterization is achieved.
    #   (good is referred in terms of how similar, in terms of how long they are, sequences inside the cluster and how dissimilar the clusters are)
    #   When collating batches the samples inside have to be in the same previously assigned cluster and the batch-assigned padding sequence length has to be the cluster max sequence length.
    #   This way the number of different sequence length reduces drastically and few cudagraphs can be created (so the following option has to be set to False).
    #   This approach is demanded to future development.
    # Current approach:
    #   Set tokenizer max sequence length to the max sequence length encountered in training data
    #   Pad to tokenizer max sequence length
    #   Compile unique cudagraph
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = False

    # Allow matmul with FP32 precision to leverage NVIDIA tensor cores. More here: https://docs.pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
    torch.backends.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = True
    activities += [ProfilerActivity.CUDA]


wandb_api_key = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = wandb_api_key


class TooManyWarmupItersError(Exception):
    def __init__(self, warmup_iters, total_iters):
        msg = f"The number of warmup iterations cannot be greater than 50% the total number of iterations, \
        got warmup_iters: {warmup_iters}, total_iterations: {total_iters}"
        super().__init__(msg)


def run(config, enable_log_ckpt=True):
    manual_seed(config.seed)
    output_dir = setup_output_dir(config)
    config.output_dir = output_dir
    save_config(config, config.output_dir, enable_ckpt=enable_log_ckpt)

    # Setup logger
    logger = setup_logger(level=logging.INFO, filepath=os.path.join(config.output_dir, "training-info.log"))
    logger.info("Configuration: \n%s", pformat(config))

    # Resume tokenizer from pretrained and build data utils
    if config.ckpt_path_to_resume_from is not None or config.tokenizers_resume_path is not None:
        src_tokenizer, tgt_tokenizer = resume_from_ckpt(
            checkpoint_path=config.ckpt_path_to_resume_from
            if config.ckpt_path_to_resume_from is not None
            else config.tokenizers_resume_path,
            logger=logger,
            resume_tokenizers=True,
            tokenizers_type=config.tokenizer.type,
        )
        train_dataloader, test_dataloader = build_data_utils(
            config, src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer
        )

    else:  # Build data utils from scratch
        train_dataloader, test_dataloader, _, _, src_tokenizer, tgt_tokenizer = build_data_utils(
            config, return_all=True
        )

    # Save tokenizers
    src_tokenizer.to_json(config.output_dir + f"/src_tokenizer_{config.tokenizer.type}.json")
    tgt_tokenizer.to_json(config.output_dir + f"/tgt_tokenizer_{config.tokenizer.type}.json")
    if config.s3_bucket_name is not None and enable_log_ckpt:
        s3_upload(
            filepath=config.output_dir + f"/src_tokenizer_{config.tokenizer.type}.json",
            bucket=config.s3_bucket_name,
            s3_key=f"{config.model_name}/src_tokenizer_{config.tokenizer.type}.json",
        )
        s3_upload(
            filepath=config.output_dir + f"/tgt_tokenizer_{config.tokenizer.type}.json",
            bucket=config.s3_bucket_name,
            s3_key=f"{config.model_name}/tgt_tokenizer_{config.tokenizer.type}.json",
        )
        logger.info(f"Uploaded tokenizers to s3://{config.s3_bucket_name}")

    config.src_tokenizer_vocab_size = src_tokenizer.vocab_size
    config.tgt_tokenizer_vocab_size = tgt_tokenizer.vocab_size
    logger.info(f"Vocabulary size of source tokenizer: {config.src_tokenizer_vocab_size}")
    logger.info(f"Vocabulary size of target tokenizer: {config.tgt_tokenizer_vocab_size}")

    config.num_train_iters_per_epoch = len(train_dataloader)
    config.num_test_iters_per_epoch = len(test_dataloader)
    logger.info(f"Number of train iterations per epoch: {config.num_train_iters_per_epoch}")
    logger.info(f"Number of test iterations per epoch: {config.num_test_iters_per_epoch}")

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
    device = get_device()
    logger.info(f"Using device: {device}")

    model = build_model(config, src_tokenizer, tgt_tokenizer)
    model.to(device)
    if torch.cuda.is_available():
        model = torch.compile(model, mode=config.training_hp.torch_compile_mode)

    logger.info(
        summary(
            model,
            [(16, 128), (16, 128), (16, 128), (16, 128)],
            dtypes=[torch.long, torch.long, torch.bool, torch.bool],
        )
    )
    logger.info(f"Total number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    model_param_groups = get_param_groups(model.named_parameters(), config.training_hp.optimizer.weight_decay)

    optim_type = {"AdamW": AdamW}
    optimizer = optim_type[config.training_hp.optimizer.type](
        model_param_groups,
        lr=1,  # This way the learning rate values will be entirely managed by the learning rate scheduler
        betas=(config.training_hp.optimizer.beta1, config.training_hp.optimizer.beta2),
    )

    if config.training_hp.loss.type == "crossentropy":
        loss_fn = CrossEntropyLoss(
            # Ignore padding tokens in loss computation
            ignore_index=tgt_tokenizer.pad_token_idx,
            # This will average the loss over batch_size * sequence_length (pad tokens don't contribute to sequence_length)
            reduction="sum",  # The loss will be manually averaged by the number of processed tokens
            # During training, we employed label smoothing of value 0.1 (Attention is all you need page 8)
            # This reduces overconfidence and improves generalization
            label_smoothing=config.training_hp.loss.label_smoothing,
        ).to(device=device)

    elif config.training_hp.loss.type == "KLdiv-labelsmoothing":
        loss_fn = KLDivLabelSmoothingLoss(
            vocab_size=tgt_tokenizer.vocab_size,
            padding_idx=tgt_tokenizer.pad_token_idx,
            smoothing=config.training_hp.loss.label_smoothing,
        ).to(device=device)

    else:
        raise ValueError(f"Invalid loss type, got {config.training_hp.loss.type}")

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
        "Loss": Loss(loss_fn, output_transform=partial(loss_metric_transform, loss_type=config.training_hp.loss.type)),
    }

    # Setup trainer and evaluator
    trainer = setup_trainer(config, model, optimizer, lr_scheduler, loss_fn, device)
    test_evaluator = setup_evaluator(config, model, metrics, device)

    # Setup engines logger with python logging print training configurations
    trainer.logger = logger
    test_evaluator.logger = logger

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
    if enable_log_ckpt:
        exp_wandb_logger, exp_trackio_logger = setup_exp_logging(
            config,
            trainer,
            optimizer,
            evaluator=test_evaluator,
            metrics=metrics,
            model=model,
            return_all_loggers=True,
        )

    # Print metrics to the stderr with "add_event_handler" method for training stats
    # More on ignite Events: https://docs.pytorch.org/ignite/generated/ignite.engine.events.Events.html
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.log_every_iters),
        log_metrics,
        tag="train",
    )

    # Clean GPU cache
    @test_evaluator.on(Events.EPOCH_STARTED | Events.EPOCH_COMPLETED)
    def cleanup_memory(engine):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # Run evaluators at every training epoch end and print metrics to the stderr
    @trainer.on(Events.EPOCH_COMPLETED)
    def run_test_eval():
        test_evaluator.run(test_dataloader)
        log_metrics(test_evaluator, "test")

    # Run evaluator when trainer starts to make sure it works.
    @trainer.on(Events.STARTED)
    def run_test_eval_on_start():
        test_evaluator.run(test_dataloader)
        # Attach early stopping to test_evaluator here cause it relies on evaluator metrics dict to be defined and it is non-empty only after first run
        if config.training_hp.early_stopping.enabled:
            logger.info("Enabled early stopping.")
            setup_early_stopping(trainer, test_evaluator, config)

    # Decode a sequence to debug training
    @test_evaluator.on(Events.EPOCH_COMPLETED(every=1))
    def run_decoding_debug():
        nr_sequences = 3
        sample_batch = next(test_dataloader.__iter__())
        decoded_seq_batch = greedy_decoding(
            model=model,
            tgt_tokenizer=tgt_tokenizer,
            src_tokens=sample_batch["src"][:nr_sequences],
            src_mask=sample_batch["src_mask"][:nr_sequences],
            max_target_tokens=config.tokenizer.max_seq_len,
            output_mode="str",
        )
        test_evaluator.logger.info(f"Source sequences: \n{sample_batch['src_text'][:nr_sequences]}")
        test_evaluator.logger.info(f"Decoded sequences: \n{decoded_seq_batch}")

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

    # Save config-lock
    save_config(config, output_dir, enable_ckpt=enable_log_ckpt)

    # Run training
    trainer.run(train_dataloader, max_epochs=config.training_hp.num_epochs)

    # Close loggers and upload local log file to s3 if configured
    if enable_log_ckpt:
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
        "-p",
        "--path",
        type=str,
        default="/".join(os.path.abspath(__file__).split("/")[:-1]),
        help="Path to the project folder (default: this script's directory).",
    )
    parser.add_argument(
        "-e",
        "--exec-mode",
        choices=["dev", "dummy"],
        default="dev",
        help="""
        Execution mode: 'dev' or 'dummy' (default: dev).
        Dev: train the model in developer way, all training features are enabled.
        Dummy: debug model training, this disables experiment tracking and checkpointing.
        """,
    )
    parser.add_argument(
        "-s",
        "--size",
        choices=["nano", "small", "base", "original"],
        default="nano",
        help="""
        Model size: 'nano', 'small' or 'base' (default: nano).
        Refer to configs/config.yml file for more info on sizes.
        """,
    )
    parser.add_argument(
        "-tl",
        "--time-limit",
        type=int,
        default=-1,
        help="Execution time limit defined in seconds (default: -1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    base_path = args.path if os.path.isdir(args.path) else "/".join(os.path.abspath(__file__).split("/")[:-1])
    config_path = os.path.join(base_path, "tfs_mt/configs/config.yml")

    config = OmegaConf.load(config_path)

    config.base_path = base_path
    config.output_dir = os.path.join(base_path, "data/output")
    config.cache_ds_path = os.path.join(base_path, "data")
    config.chosen_model_size = args.size
    config.time_limit_sec = args.time_limit if args.time_limit > 0 else -1
    config.wandb_organization = os.getenv("WANDB_ORGANIZATION")

    # Detect Kaggle enviroment and make adjustments
    if os.path.exists("/kaggle"):
        print("Kaggle environment detected. Overriding some config options...")
        config.time_limit_sec = (
            args.time_limit if (args.time_limit <= 11.5 * 3600 and args.time_limit > 0) else 11.5 * 3600
        )
        config.training_hp.amp_dtype = "float16"  # Neither the T4 nor the P100 support bfloat16
        config.train_dataloader.num_workers = 4
        config.test_dataloader.num_workers = 4
        config.training_hp.torch_compile_mode = None

    # Setup dummy training. Mainly used for debugging
    print(f"Execution mode: {args.exec_mode}")
    if args.exec_mode == "dummy":
        print("Experiment logging and checkpointing are disabled!")
        config.dataset.max_len = 1_000
        config.training_hp.lr_scheduler.warmup_iters = (
            5  # if it's too high it raises TooManyWarmupItersError during training setup
        )
        config.log_every_iters = 1
        config.training_hp.num_epochs = 2
        config.training_hp.torch_compile_mode = None

    # Fix batch sizes if they are larger than train/test data (dataset_max_len * split_proportion)
    if config.dataset.max_len > 0 and config.train_dataloader.batch_size > int(
        config.dataset.max_len * config.dataset.train_split
    ):
        config.train_dataloader.batch_size = int(config.dataset.max_len * config.dataset.train_split)
    if config.dataset.max_len > 0 and config.test_dataloader.batch_size > int(
        config.dataset.max_len * (1 - config.dataset.train_split)
    ):
        config.test_dataloader.batch_size = int(config.dataset.max_len * (1 - config.dataset.train_split))

    config.model_name = f"{config.model_base_name}_{config.chosen_model_size}_{datetime.now().strftime('%y%m%d-%H%M')}"

    config.exec_mode = args.exec_mode

    # Run and profile training (PyTorch official tutorial on profiling: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
    with (
        profile(activities=activities, profile_memory=True, record_shapes=True) as prof,
        record_function("model_training"),
    ):
        if args.exec_mode == "dummy":
            run(config, enable_log_ckpt=False)
        elif args.exec_mode == "dev":
            run(config, enable_log_ckpt=True)

    prof.export_chrome_trace(os.path.join(config.output_dir, "trace.json"))
    if config.s3_bucket_name is not None and args.exec_mode != "dummy":
        s3_upload(
            filepath=os.path.join(config.output_dir, "trace.json"),
            bucket=config.s3_bucket_name,
            s3_key=f"{config.model_name}/trace.json",
        )
