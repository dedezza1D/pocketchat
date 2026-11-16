"""Generate text from a checkpoint.

    uv run python scripts/sample.py --ckpt checkpoints/smoke.pt --prompt "The water cycle"
"""

import argparse

import torch

from pocketchat.model import GPT, GPTConfig
from pocketchat.tokenizer import BPETokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pretrain.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default="The")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(args.tokenizer)
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    stop = {tok.special_tokens["<|endoftext|>"]}

    for i in range(args.n):
        idx = torch.tensor([tok.encode(args.prompt)], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model.generate(idx, args.tokens, args.temperature, args.top_k, stop)
        print(f"--- sample {i + 1} ---")
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
