"""Generate text from a checkpoint.

    uv run python scripts/sample.py --ckpt checkpoints/smoke.pt --prompt "The water cycle" --seed 0
"""

import argparse

import torch

from pocketchat.checkpoint import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pretrain_final.pt")
    ap.add_argument("--tokenizer", default=None, help="default: the tokenizer recorded in the checkpoint")
    ap.add_argument("--prompt", default="The")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None, help="fix the draws to reproduce a sample")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load_model(args.ckpt, device, args.tokenizer)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    stop = {tok.special_tokens["<|endoftext|>"]}

    for i in range(args.n):
        idx = torch.tensor([tok.encode(args.prompt)], device=device)
        with torch.autocast("cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model.generate(idx, args.tokens, args.temperature, args.top_k, stop)
        print(f"--- sample {i + 1} ---")
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
