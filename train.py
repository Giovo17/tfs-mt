# Transformer train script
#
# Author: Giovanni Spadaro - https://giovannispadaro.it
# Project: https://github.com/Giovo17/tfs-mt
# Documentation: https://giovo17.github.io/tfs-mt
#
# This script is licensed under the license found in the LICENSE file in the root directory of this source tree.

import argparse
import gc
import os
from datetime import datetime
from functools import partial
from pprint import pformat

import ignite.distributed as idist
import torch
from ignite.engine import Events
from ignite.handlers import FastaiLRFinder, PiecewiseLinear
from ignite.metrics import Bleu, Loss, Rouge
from ignite.utils import manual_seed
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torchinfo import summary

from tfs_mt.architecture import build_model
from tfs_mt.data_utils import build_data_utils
from tfs_mt.decoding_utils import greedy_decoding
from tfs_mt.training_utils import (
    log_metrics,
    loss_metric_transform,
    nlp_metric_transform,
    resume_from_ckpt,
    save_config,
    setup_evaluator,
    setup_exp_logging,
    setup_handlers,
    setup_logging,
    setup_output_dir,
    setup_trainer,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # remove tokenizer parallelism warning


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


from dotenv import load_dotenv

load_dotenv()

wandb_api_key = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = wandb_api_key

config_path = os.path.join(os.getcwd(), "tfs_mt/configs/config.yml")
base_path = os.getcwd()
output_dir = os.path.join(base_path, "data/output")
cache_ds_path = os.path.join(base_path, "data")
time_limit_sec = -1


class EpochConfigError(Exception):
    def __init__(self, num_warmup_epochs, num_epochs):
        msg = f"num_warmup_epochs cannot be greater than num_epochs, \
        got num_warmup_epochs={num_warmup_epochs}, num_epochs={num_epochs}"
        super().__init__(msg)


def run(local_rank, config, distributed=False, enable_log_ckpt=True):
    if distributed:
        rank = idist.get_rank()
        manual_seed(config.seed + rank)
        output_dir = setup_output_dir(config, rank)
        config.output_dir = output_dir

        idist.barrier()

        if rank == 0:
            save_config(config, output_dir)
    else:
        rank = 0
        manual_seed(config.seed)
        output_dir = setup_output_dir(config, 0)
        config.output_dir = output_dir
        save_config(config, config.output_dir)

    train_dataloader, test_dataloader, _, _, src_tokenizer, tgt_tokenizer = build_data_utils(config, return_all=True)

    config.num_iters_per_epoch = len(train_dataloader)

    # Initialize model, optimizer, loss function, device or resume from checkpoint
    device = idist.device()
    init_model = build_model(config, src_tokenizer, tgt_tokenizer)
    init_model = torch.compile(init_model)
    print(
        summary(
            init_model,
            [(16, 128), (16, 128), (16, 128), (16, 128)],
            dtypes=[torch.long, torch.long, torch.bool, torch.bool],
        )
    )

    # Get model ready for multigpu training if available and move the model to current device
    model = idist.auto_model(init_model)

    if distributed:
        config.training_hp.optimizer_args.learning_rate *= idist.get_world_size()
    init_optimizer = AdamW(
        model.parameters(),
        lr=config.training_hp.optimizer_args.learning_rate,
        weight_decay=config.training_hp.optimizer_args.weight_decay,
    )
    # Get model ready for multigpu training if available
    optimizer = idist.auto_optim(init_optimizer)
    loss_fn = CrossEntropyLoss(label_smoothing=config.training_hp.loss_label_smoothing).to(device=device)

    le = config.num_iters_per_epoch
    milestones_values = [
        (0, 0.0),
        (le * config.training_hp.num_warmup_epochs, config.training_hp.optimizer_args.learning_rate),
        (le * config.training_hp.num_epochs, 0.0),
    ]
    lr_scheduler = PiecewiseLinear(optimizer, param_name="lr", milestones_values=milestones_values)

    # Setup metrics to attach to evaluator
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
    trainer = setup_trainer(config, model, optimizer, loss_fn, device, train_dataloader.sampler)
    train_evaluator, test_evaluator = setup_evaluator(config, model, metrics, device)

    # Setup engines logger with python logging print training configurations
    logger = setup_logging(config)
    logger.info("Configuration: \n%s", pformat(config))
    trainer.logger = train_evaluator.logger = test_evaluator.logger = logger

    trainer.add_event_handler(Events.ITERATION_COMPLETED, lr_scheduler)

    # Setup ignite handlers
    to_save_train = {
        "model": model,
        "optimizer": optimizer,
        "trainer": trainer,
        "lr_scheduler": lr_scheduler,
    }
    to_save_test = {"model": model}
    # When enable_log_ckpt is False the two returned ckpt handlers will be None
    ckpt_handler_train, ckpt_handler_test = setup_handlers(
        trainer, test_evaluator, config, to_save_train, to_save_test, enable_ckpt=enable_log_ckpt
    )

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
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
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
    @test_evaluator.on(Events.EPOCH_COMPLETED(every=1))
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
            to_load=to_save_train,
            checkpoint_filepath=config.ckpt_path_to_resume_from,
            device=device,
            logger=logger,
            strict=True,
        )

    # Cyclical learning rate scheduling according to https://arxiv.org/abs/1506.01186
    lr_finder = FastaiLRFinder()
    to_save = {"model": model, "optimizer": optimizer}

    with lr_finder.attach(
        trainer,
        to_save=to_save,
        output_transform=lambda x: x["train_loss"],
        start_lr=1e-5,  # config.training_hp.optimizer_args.learning_rate
        end_lr=1e-2,
        step_mode="exp",
    ) as trainer_with_lr_finder:
        trainer_with_lr_finder.run(
            train_dataloader,
            max_epochs=config.training_hp.num_epochs,
        )

    # Plot lr_finder results and get lr_finder suggestion
    try:
        ax = lr_finder.plot(skip_end=0)
        ax.figure.savefig(config.output_dir + "/lr_finder_results.png")
        print(lr_finder.lr_suggestion())
    except Exception:
        print("Unable to plot lr_finder results")

    # trainer.run(
    #     train_dataloader,
    #     max_epochs=config.training_hp.num_epochs,
    # )

    if enable_log_ckpt and rank == 0:
        exp_wandb_logger.close()
        exp_trackio_logger.close()
        # profiler.write_results(config.output_dir + "/time_profiling.csv")

    if ckpt_handler_train is not None:
        logger.info(f"Last training checkpoint name - {ckpt_handler_train.last_checkpoint}")
    if ckpt_handler_test is not None:
        logger.info(f"Last testing checkpoint name - {ckpt_handler_test.last_checkpoint}")


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

    if config.training_hp.num_warmup_epochs > config.training_hp.num_epochs:
        raise EpochConfigError(
            f"num_warmup_epochs cannot be greater than num_epochs, \
        got num_warmup_epochs={config.training_hp.num_warmup_epochs}, num_epochs={config.training_hp.num_epochs}"
        )

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
