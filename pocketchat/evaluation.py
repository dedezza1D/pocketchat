"""Validation loss on fixed windows, shared by training and standalone evaluation."""

from __future__ import annotations

import torch

from pocketchat.data import TokenStream


@torch.no_grad()
def validation_loss(model, stream: TokenStream, batch_size: int, device: str, max_windows: int = 0) -> float:
    """Mean loss over the same non-overlapping windows on every call (all of them when max_windows is 0)."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for x, y in stream.windows(batch_size, device, max_windows):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            _, loss = model(x, y)
        total += loss.item() * x.shape[0]
        count += x.shape[0]
    model.train(was_training)
    return total / count
