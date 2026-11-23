"""Chat with a fine-tuned checkpoint in the terminal.

    uv run python scripts/chat.py --ckpt checkpoints/sft_reinit.pt

The model keeps no state between messages: the whole conversation is re-encoded
every turn. When it no longer fits in the context window, the oldest turns are dropped.
Type /reset to start over, /quit to leave.
"""

import argparse

import torch

from pocketchat.chat_format import END, USER, prompt_ids
from pocketchat.checkpoint import load_model


def fit_history(tok, turns, seq_len, reserve):
    """Drop the oldest user/assistant pairs until the prompt leaves `reserve` tokens free."""
    while len(turns) > 1 and len(prompt_ids(tok, turns)) > seq_len - reserve:
        turns = turns[2:]
    return turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft_reinit.pt")
    ap.add_argument("--tokenizer", default=None, help="default: the tokenizer recorded in the checkpoint")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, state = load_model(args.ckpt, device, args.tokenizer)
    stop = {tok.special_tokens[END], tok.special_tokens[USER]}

    turns: list[tuple[str, str]] = []
    print("pocketchat — /reset to clear, /quit to exit")
    while True:
        try:
            msg = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg == "/quit":
            break
        if msg == "/reset":
            turns = []
            continue
        if not msg:
            continue
        turns = fit_history(tok, turns + [("user", msg)], model.cfg.seq_len, args.max_new)
        prompt = torch.tensor([prompt_ids(tok, turns)], device=device)
        with torch.autocast("cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model.generate(prompt, args.max_new, args.temperature, top_k=None, stop_ids=stop,
                                 top_p=args.top_p, repetition_penalty=args.repetition_penalty)
        reply = tok.decode([t for t in out[0, prompt.shape[1]:].tolist() if t not in stop])
        print(f"bot> {reply.strip()}")
        turns.append(("assistant", reply.strip()))


if __name__ == "__main__":
    main()
