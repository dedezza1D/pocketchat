"""Token datasets stored as flat uint16 arrays on disk, read through a memory map."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenStream:
    """Samples random (input, target) windows from one long sequence of token ids."""

    def __init__(self, path: str | Path, seq_len: int):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        if len(self.tokens) <= seq_len + 1:
            raise ValueError(f"{path} has only {len(self.tokens)} tokens")

    def __len__(self) -> int:
        return len(self.tokens)

    def batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        starts = np.random.randint(0, len(self.tokens) - self.seq_len - 1, size=batch_size)
        chunk = np.stack([self.tokens[s : s + self.seq_len + 1] for s in starts]).astype(np.int64)
        x = torch.from_numpy(chunk[:, :-1])
        y = torch.from_numpy(chunk[:, 1:])
        if device.startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def write_tokens(path: str | Path, ids: np.ndarray) -> None:
    if ids.max(initial=0) >= 2**16:
        raise ValueError("token ids do not fit in uint16; use a vocabulary below 65536")
    ids.astype(np.uint16).tofile(path)
