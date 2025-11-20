# Ignite common utils
#
# Author: Giovanni Spadaro - https://giovannispadaro.it
# Project: https://github.com/Giovo17/tfs-mt
# Documentation: https://giovo17.github.io/tfs-mt
#
# Copyright (c) Giovanni Spadaro.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

from typing import Any

from ignite.contrib.engines import common
from ignite.engine import Engine
from torch.optim.optimizer import Optimizer

from .trackio_logger import TrackioLogger
from .wandb_logger import WandBLogger


def setup_wandb_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> WandBLogger:
    logger = WandBLogger(**kwargs)
    common._setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_trackio_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> TrackioLogger:
    logger = TrackioLogger(**kwargs)
    common._setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger
