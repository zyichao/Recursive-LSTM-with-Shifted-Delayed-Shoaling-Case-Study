# training/losses.py
#
# Custom zero-crossing-weighted MSE loss from Zeng et al. 2026 (Eq. 36-37):
# standard MSE weights every prediction equally, which can underemphasize
# transition points where CIR crosses zero (a local peak or valley in the
# sediment trend) -- exactly the points most relevant to dredging decisions.
# This loss assigns a higher weight (w_high) to windows whose target
# contains a zero crossing, and a lower weight (w_low) to the rest.

import torch
import torch.nn as nn


class PlainMSELoss(nn.Module):
    """Standard (unweighted) MSE loss. Accepts and ignores the zero-crossing
    flag argument so it shares a call signature with
    WeightedZeroCrossingMSELoss -- the training loop can use either
    interchangeably without branching."""

    def forward(self, y_pred, y_true, zero_crossing_flags=None):
        return torch.mean((y_pred - y_true) ** 2)


class WeightedZeroCrossingMSELoss(nn.Module):
    """Weighted MSE loss (Eq. 37): each sample b in the batch is weighted

        w(b) = w_high   if its target window contains a CIR zero crossing
             = w_low     otherwise

    and the loss is the mean, over the batch and the prediction horizon, of
    w(b) * (y_pred - y_true)^2. The paper found w_high = 10 (with w_low = 1)
    optimal via grid search (Table 9).
    """

    def __init__(self, w_low: float = 1.0, w_high: float = 10.0):
        super().__init__()
        self.w_low = w_low
        self.w_high = w_high

    def forward(self, y_pred, y_true, zero_crossing_flags):
        weight = torch.where(
            zero_crossing_flags.bool(),
            torch.full_like(zero_crossing_flags, self.w_high),
            torch.full_like(zero_crossing_flags, self.w_low),
        )
        per_sample_mse = ((y_pred - y_true) ** 2).mean(dim=1)
        return (weight * per_sample_mse).mean()


def build_loss_fn(loss_cfg: dict) -> nn.Module:
    """Build the training loss from config.yaml's `loss` block.

    loss_cfg keys: `weighted_zero_crossing` (bool), `w_low`, `w_high`.
    """
    if loss_cfg.get("weighted_zero_crossing", False):
        return WeightedZeroCrossingMSELoss(
            w_low=loss_cfg.get("w_low", 1.0),
            w_high=loss_cfg.get("w_high", 10.0),
        )
    return PlainMSELoss()
