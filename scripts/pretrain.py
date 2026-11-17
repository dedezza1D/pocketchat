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
from pocketchat.evaluation import validation_loss
from pocketchat.model import GPT, GPTConfig
from pocketchat.tokenizer import BPETokenizer


def lr_at(step: int, schedule_steps: int, warmup: int, peak: float, floor: float) -> float:
    """Linear warmup, cosine decay to `floor` at `schedule_steps`, then constant at `floor`."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, schedule_steps - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


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
    ap.add_argument(
        "--schedule-steps",
        type=int,
        default=0,
        help="length of the warmup + cosine schedule (default: --max-steps). Keep the original value "
        "when extending a finished run so the learning rate stays at --min-lr instead of rising again.",
    )
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-windows", type=int, default=0, help="fixed validation windows per eval (0 = all)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", action="store_true", help="continue from <out>.last.pt if it exists")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    schedule_steps = args.schedule_steps or args.max_steps
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

    train = TokenStream(Path(args.data_dir) / "train.bin", cfg.seq_len, seed=args.seed)
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
    start_step = 0
    last_path = Path(args.out).with_suffix(".last.pt")
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_step, best_val, log = state["step"] + 1, state["best_val"], state["log"]
        if "data_rng" in state:
            train.rng.bit_generator.state = state["data_rng"]
        print(f"resumed from {last_path} at step {start_step}")

    t_last = time.time()
    tokens_since = 0
    for step in range(start_step, args.max_steps):
        lr = lr_at(step, schedule_steps, args.warmup, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        step_loss = torch.zeros((), device=device)
        for _ in range(args.grad_accum):
            x, y = train.batch(args.batch_size, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            step_loss += loss.detach() / args.grad_accum
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
            print(f"step {step:5d} | loss {step_loss.item():.3f} | lr {lr:.2e} | grad {norm:.2f} | {tps / 1e3:.1f}k tok/s")

        if (step > 0 and step % args.eval_every == 0) or step == args.max_steps - 1:
            v = validation_loss(model, val, args.batch_size, device, args.eval_windows)
            log.append({"step": step, "train_loss": step_loss.item(), "val_loss": v, "lr": lr})
            print(f"  eval: val loss {v:.3f} (perplexity {math.exp(v):.1f})")
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            if v < best_val:
                best_val = v
                torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step, "val_loss": v}, args.out)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step,
                    "best_val": best_val,
                    "log": log,
                    "data_rng": train.rng.bit_generator.state,
                },
                last_path,
            )
            Path(args.out).with_suffix(".log.json").write_text(json.dumps(log, indent=1))

    Path(args.out).with_suffix(".log.json").write_text(json.dumps(log, indent=1))
    if device == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"best val loss {best_val:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
