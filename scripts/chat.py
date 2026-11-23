"""Chat with a fine-tuned checkpoint in the terminal.

    uv run python scripts/chat.py --ckpt checkpoints/sft_reinit.pt

The model keeps no state between messages: the whole conversation is re-encoded
every turn. When it no longer fits in the context window, the oldest turns are dropped.
Type /reset to start over, /quit to leave.

With --seed, every reply is drawn with a generator seeded with that number, so the same
conversation gives the same replies on the same hardware and PyTorch version. Replies are drawn
as in eval_chat.py, so on the GPU an evaluation ran on, --seed 1337 gives for a first question
the answer that evaluation wrote to evals/.
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


def reply(model, tok, turns, device, max_new, temperature, top_p, repetition_penalty, seed=None) -> str:
    """The assistant's next message, sampled as eval_chat.py does (a fresh generator per reply when seeded)."""
    stop = {tok.special_tokens[END], tok.special_tokens[USER]}
    g = None if seed is None else torch.Generator(device=device).manual_seed(seed)
    prompt = torch.tensor([prompt_ids(tok, turns)], device=device)
    with torch.autocast("cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model.generate(prompt, max_new, temperature, top_k=None, stop_ids=stop, top_p=top_p,
                             repetition_penalty=repetition_penalty, generator=g)
    return tok.decode([t for t in out[0, prompt.shape[1]:].tolist() if t not in stop]).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft_reinit.pt")
    ap.add_argument("--tokenizer", default=None, help="default: the tokenizer recorded in the checkpoint")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=None, help="repeat replies; 1337 is the seed of evals/")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, state = load_model(args.ckpt, device, args.tokenizer)

    turns: list[tuple[str, str]] = []
    print("pocketchat — /reset to clear, /quit to exit")
    while True:
        try:
            # Windows PowerShell starts text piped into a program with a byte order mark
            msg = input("\nyou> ").replace("\ufeff", "").strip()
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
        text = reply(model, tok, turns, device, args.max_new, args.temperature, args.top_p,
                     args.repetition_penalty, args.seed)
        print(f"bot> {text}")
        turns.append(("assistant", text))


if __name__ == "__main__":
    main()
