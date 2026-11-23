"""Supervised fine-tuning: teach the pretrained model to answer in the chat format.

    uv run python scripts/sft.py --out checkpoints/sft_reinit.pt --reinit-chat-tokens

The loss only counts assistant tokens (targets of every other position are -100).
Validation uses all held-out conversations, weighted by trained tokens, and the
checkpoint with the lowest validation loss is kept.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from pocketchat.chat_format import ASSISTANT, END, IGNORE, USER
from pocketchat.checkpoint import resolve_tokenizer, tokenizer_record
from pocketchat.model import GPT, GPTConfig
from pocketchat.tokenizer import BPETokenizer


class ChatSet:
    def __init__(self, path: Path):
        d = np.load(path)
        self.x, self.y, self.off = d["x"], d["y"], d["offsets"]

    def __len__(self) -> int:
        return len(self.off) - 1

    def example(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.off[i], self.off[i + 1]
        return self.x[a:b].astype(np.int64), self.y[a:b].astype(np.int64)

    def batch(self, idx, pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        exs = [self.example(i) for i in idx]
        n = max(len(x) for x, _ in exs)
        x = np.full((len(exs), n), pad_id, dtype=np.int64)
        y = np.full((len(exs), n), IGNORE, dtype=np.int64)
        for r, (xi, yi) in enumerate(exs):
            x[r, : len(xi)] = xi
            y[r, : len(yi)] = yi
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def lr_at(step, total, warmup, peak, floor):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def val_loss(model, data: ChatSet, batch_size: int, pad_id: int, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(data), batch_size):
        x, y = data.batch(range(i, min(i + batch_size, len(data))), pad_id, device)
        n = int((y != IGNORE).sum())
        with torch.autocast("cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y)
        total += loss.item() * n
        count += n
    model.train()
    return total / count


def reinit_chat_rows(model: GPT, tok: BPETokenizer, seed: int) -> None:
    """Replace the chat-token rows with fresh rows that look like an ordinary token.

    After pretraining these rows collapse to nearly the same vector (they were the
    wrong answer at every step through the tied output head). New rows start at the
    mean of the ordinary rows plus independent noise with the ordinary per-dimension spread.
    """
    W = model.tok_emb.weight.data
    special = set(tok.special_tokens.values())
    normal = torch.tensor([i for i in range(W.shape[0]) if i not in special], device=W.device)
    mean, std = W[normal].mean(0), W[normal].std(0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for name in (USER, ASSISTANT, END):
        noise = torch.randn(W.shape[1], generator=g).to(W.device, W.dtype)
        W[tok.special_tokens[name]] = mean + noise * std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/pretrain_final.pt")
    ap.add_argument("--tokenizer", default=None, help="default: the tokenizer recorded in --init")
    ap.add_argument("--data-dir", default="data/sft")
    ap.add_argument("--out", default="checkpoints/sft.pt")
    ap.add_argument("--reinit-chat-tokens", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    init = torch.load(args.init, map_location=device)
    tok = resolve_tokenizer(init, args.tokenizer)
    tok_rec = init.get("tokenizer") or tokenizer_record(args.tokenizer)
    pad_id = tok.special_tokens["<|endoftext|>"]
    cfg = GPTConfig(**{**init["config"], "dropout": args.dropout})
    model = GPT(cfg).to(device)
    model.load_state_dict(init["model"])
    if args.reinit_chat_tokens:
        reinit_chat_rows(model, tok, args.seed)

    train = ChatSet(Path(args.data_dir) / "train.npz")
    val = ChatSet(Path(args.data_dir) / "val.npz")
    per_step = args.batch_size * args.grad_accum
    total_steps = int(args.epochs * len(train) / per_step)
    print(f"{len(train)} train / {len(val)} val conversations, {total_steps} steps, reinit={args.reinit_chat_tokens}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=device == "cuda",
    )

    v0 = val_loss(model, val, 32, pad_id, device)
    log = [{"step": 0, "val_loss": v0}]
    best = v0
    print(f"step     0 | val {v0:.3f}")

    order = rng.permutation(len(train))
    cursor, t0 = 0, time.time()
    for step in range(1, total_steps + 1):
        lr = lr_at(step - 1, total_steps, args.warmup, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        micro = []
        for _ in range(args.grad_accum):
            if cursor + args.batch_size > len(order):
                order, cursor = rng.permutation(len(train)), 0
            micro.append(train.batch(order[cursor : cursor + args.batch_size], pad_id, device))
            cursor += args.batch_size
        n_total = sum(int((y != IGNORE).sum()) for _, y in micro)
        step_loss = 0.0
        for x, y in micro:
            n = int((y != IGNORE).sum())
            with torch.autocast("cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            (loss * n / n_total).backward()  # token-weighted mean over the whole step
            step_loss += loss.item() * n / n_total
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 20 == 0:
            print(f"step {step:5d} | loss {step_loss:.3f} | lr {lr:.2e} | grad {norm:.2f} | {time.time() - t0:.0f}s")
        if step % args.eval_every == 0 or step == total_steps:
            v = val_loss(model, val, 32, pad_id, device)
            log.append({"step": step, "train_loss": step_loss, "val_loss": v, "lr": lr})
            mark = ""
            if v < best:
                best, mark = v, " (best)"
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"config": {**cfg.to_dict(), "dropout": 0.0}, "model": model.state_dict(),
                            "step": step, "val_loss": v, "tokenizer": tok_rec, "args": vars(args)}, args.out)
            print(f"  eval: val {v:.3f}{mark}")

    Path(args.out).with_suffix(".log.json").write_text(json.dumps(log, indent=1))
    print(f"best val {best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
