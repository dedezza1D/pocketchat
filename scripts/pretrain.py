"""Pretrain the GPT on next-token prediction.

    uv run python scripts/pretrain.py --max-steps 200 --out checkpoints/smoke.pt

Uses bf16 autocast, gradient accumulation, AdamW with a warmup + cosine schedule,
and reports throughput so a full run can be sized for the available GPU.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from pocketchat.data import TokenStream
from pocketchat.model import GPT, GPTConfig
from pocketchat.tokenizer import BPETokenizer


def lr_at(step: int, max_steps: int, warmup: int, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, stream: TokenStream, batches: int, batch_size: int, device: str) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = stream.batch(batch_size, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out", default="checkpoints/pretrain.pt")
    ap.add_argument("--n-layer", type=int, default=8)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16, help="sequences per micro-batch")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    tok = BPETokenizer.load(args.tokenizer)
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        seq_len=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(cfg).to(device)
    print(f"model: {model.num_params() / 1e6:.1f}M parameters, device={device}")

    train = TokenStream(Path(args.data_dir) / "train.bin", cfg.seq_len)
    val = TokenStream(Path(args.data_dir) / "val.bin", cfg.seq_len)
    tokens_per_step = args.batch_size * args.grad_accum * cfg.seq_len
    print(f"train tokens: {len(train) / 1e6:.1f}M, tokens per step: {tokens_per_step}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=device == "cuda",
    )

    log = []
    best_val = float("inf")
    t_last = time.time()
    tokens_since = 0
    for step in range(args.max_steps):
        lr = lr_at(step, args.max_steps, args.warmup, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        for _ in range(args.grad_accum):
            x, y = train.batch(args.batch_size, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens_since += tokens_per_step

        if step % 10 == 0 or step == args.max_steps - 1:
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t_last
            tps = tokens_since / dt
            t_last, tokens_since = time.time(), 0
            print(f"step {step:5d} | loss {loss.item():.3f} | lr {lr:.2e} | grad {norm:.2f} | {tps / 1e3:.1f}k tok/s")

        if (step > 0 and step % args.eval_every == 0) or step == args.max_steps - 1:
            v = evaluate(model, val, args.eval_batches, args.batch_size, device)
            log.append({"step": step, "train_loss": loss.item(), "val_loss": v})
            print(f"  eval: val loss {v:.3f} (perplexity {math.exp(v):.1f})")
            if v < best_val:
                best_val = v
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step, "val_loss": v}, args.out)

    Path(args.out).with_suffix(".log.json").write_text(json.dumps(log, indent=1))
    if device == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"best val loss {best_val:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
