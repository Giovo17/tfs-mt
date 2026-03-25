from collections.abc import Callable

import torch
import torch.nn as nn
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from jaxtyping import Float


class Perplexity(Metric):
    """Computes Perplexity metric: exp( Sum(Loss) / Total_Tokens ).

    If the criterion is KLDivLabelSmoothingLoss, it returns the average loss (KL divergence) without exponentiation.

    Args:
        criterion (nn.Module): Loss criterion. The criterion must return the sum of losses.
        pad_idx (int): The padding index to ignore.
        output_transform (callable, optional): function that transforms 'x' input to update method.
        device (Union[str, torch.device], optional): device specification in case of distributed computation usage.
    """

    def __init__(
        self,
        criterion: nn.Module,
        pad_idx: int,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = "cpu",
    ):
        self.criterion = (
            criterion if criterion is not None else nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
        )
        self.pad_idx = pad_idx
        super().__init__(output_transform=output_transform, device=device)

    def _check_shape_and_type_consistency(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Check shape and type consistency of input vectors."""
        if preds.dim() not in [2, 3]:
            raise ValueError(
                "Input tensor `preds` is expected to have 2 dimensions [batch_size*seq_len, vocab_size] "
                "or 3 dimensions [batch_size, seq_len, vocab_size], "
                f"but got {preds.dim()}."
            )

        if target.dim() not in [1, 2]:
            raise ValueError(
                "Input tensor `target` is expected to have 1 dimension [batch_size*seq_len] "
                "or 2 dimensions [batch_size, seq_len], "
                f"but got {target.dim()}."
            )

        if preds.dim() == 2 and target.dim() != 1:
            raise ValueError(f"If preds is 2D, target must be 1D. Got preds: {preds.shape}, target: {target.shape}")

        if preds.dim() == 3 and target.dim() != 2:
            raise ValueError(f"If preds is 3D, target must be 2D. Got preds: {preds.shape}, target: {target.shape}")

        if not preds.is_floating_point():
            raise TypeError(f"Input tensor `preds` is expected to be of floating point type but got {preds.dtype}.")

        if target.dtype != torch.int64:
            raise TypeError(f"Input tensor `target` is expected to be of a type {torch.int64} but got {target.dtype}.")

    @reinit__is_reduced
    def reset(self):
        self._sum_loss = torch.tensor(0.0, device=self._device)
        self._num_tokens = torch.tensor(0.0, device=self._device)
        super().reset()

    @reinit__is_reduced
    def update(self, output: tuple[Float[torch.Tensor, "B S D"], Float[torch.Tensor, "B S D"]]):
        y_pred, y = output

        self._check_shape_and_type_consistency(y_pred, y)

        # Reshape to (N, C) and (N,) if necessary
        if y_pred.dim() == 3:
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            y = y.reshape(-1)

        loss = self.criterion(y_pred, y)

        # Count tokens excluding PAD tokens
        mask = y != self.pad_idx
        num_tokens = mask.sum()

        self._sum_loss += loss.item()
        self._num_tokens += num_tokens.item()

    @sync_all_reduce("_sum_loss", "_num_tokens")
    def compute(self):
        if self._num_tokens == 0:
            raise NotComputableError("Perplexity must have at least one example before it can be computed.")

        mean_loss = self._sum_loss / self._num_tokens

        if self.criterion.__class__.__name__ == "KLDivLabelSmoothingLoss":
            return mean_loss.item()

        return torch.exp(mean_loss).item()
