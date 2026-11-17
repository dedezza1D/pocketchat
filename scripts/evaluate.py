"""Measure checkpoints on the full validation set (fixed windows).

    uv run python scripts/evaluate.py --ckpt checkpoints/pretrain.pt checkpoints/pretrain.last.pt

Works with the best checkpoint (config + model) and with <out>.last.pt (model only;
the config is then read from --config-from).
"""

import argparse
import math
from pathlib import Path

import torch

from pocketchat.data import TokenStream
from pocketchat.evaluation import validation_loss
from pocketchat.model import GPT, GPTConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--config-from", default="checkpoints/pretrain.pt")
    ap.add_argument("--data-dir", default="data/full")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for path in args.ckpt:
        state = torch.load(path, map_location=device)
        config = state.get("config") or torch.load(args.config_from, map_location="cpu")["config"]
        model = GPT(GPTConfig(**config)).to(device)
        model.load_state_dict(state["model"])
        val = TokenStream(Path(args.data_dir) / "val.bin", model.cfg.seq_len)
        loss = validation_loss(model, val, args.batch_size, device)
        print(
            f"{path}: step {state.get('step')} | full val loss {loss:.4f} | "
            f"perplexity {math.exp(loss):.2f} | {val.num_windows} windows"
        )


if __name__ == "__main__":
    main()
