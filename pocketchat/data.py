"""Token datasets stored as flat uint16 arrays on disk, read through a memory map."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenStream:
    """Samples random (input, target) windows from one long sequence of token ids."""

    def __init__(self, path: str | Path, seq_len: int, seed: int | None = None):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        if len(self.tokens) <= seq_len + 1:
            raise ValueError(f"{path} has only {len(self.tokens)} tokens")

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def num_windows(self) -> int:
        """Number of non-overlapping (input, target) windows in the file."""
        return (len(self.tokens) - 1) // self.seq_len

    def _to_device(self, chunk: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(chunk[:, :-1])
        y = torch.from_numpy(chunk[:, 1:])
        if device.startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    def batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Random windows, for training."""
        starts = self.rng.integers(0, len(self.tokens) - self.seq_len - 1, size=batch_size)
        chunk = np.stack([self.tokens[s : s + self.seq_len + 1] for s in starts]).astype(np.int64)
        return self._to_device(chunk, device)

    def windows(self, batch_size: int, device: str, max_windows: int = 0):
        """Fixed, non-overlapping windows from the start of the file, for evaluation.

        Every call yields the same windows, so losses from different checkpoints
        or different evaluation steps are directly comparable.
        """
        total = self.num_windows if max_windows <= 0 else min(max_windows, self.num_windows)
        for first in range(0, total, batch_size):
            idx = range(first, min(first + batch_size, total))
            chunk = np.stack(
                [self.tokens[i * self.seq_len : i * self.seq_len + self.seq_len + 1] for i in idx]
            ).astype(np.int64)
            yield self._to_device(chunk, device)


def write_tokens(path: str | Path, ids: np.ndarray) -> None:
    if ids.max(initial=0) >= 2**16:
        raise ValueError("token ids do not fit in uint16; use a vocabulary below 65536")
    ids.astype(np.uint16).tofile(path)
