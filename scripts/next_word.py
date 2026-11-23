"""Show the chance the model gives to each possible next token, as in the tables of the blog series.

    uv run python scripts/next_word.py --prompt "Water is made of hydrogen and"
    uv run python scripts/next_word.py --prompt "Water is made of hydrogen and" --temperature 0.5 --word table
    uv run python scripts/next_word.py --ckpt checkpoints/sft_reinit.pt --prompt "Water is made of hydrogen and"

Special tokens written in the prompt (<|user|>, <|assistant|>, <|end|>) become control tokens:

    uv run python scripts/next_word.py --ckpt checkpoints/sft_reinit.pt \
        --prompt "<|user|>What is water made of?<|end|><|assistant|>Water is made of hydrogen and oxygen." --word "<|end|>"

--word looks a word up as it would follow the prompt, with a space before it ("table" -> " table");
a word that starts with a space or is a special token is used as written. When a word is split
into several tokens, its chance is the product of the chances of its tokens, one after the other.
"""

import argparse
import math

import torch
import torch.nn.functional as F

from pocketchat.checkpoint import load_model


@torch.no_grad()
def next_token_scores(model, ids: list[int], device: str = "cpu") -> torch.Tensor:
    """The model's score for every vocabulary token after `ids` (the last seq_len tokens are used)."""
    context = torch.tensor([ids[-model.cfg.seq_len :]], device=device)
    logits, _ = model(context)
    return logits[0, -1].float()


def word_ids(tok, word: str) -> list[int]:
    if word in tok.special_tokens:
        return [tok.special_tokens[word]]
    return tok.encode(word if word.startswith(" ") else " " + word, allow_special=False)


@torch.no_grad()
def word_chance(model, ids: list[int], target: list[int], temperature: float = 1.0, device: str = "cpu") -> float:
    """Chance that the tokens of `target` come next, one after the other."""
    chance = 1.0
    for t in target:
        chance *= F.softmax(next_token_scores(model, ids, device) / temperature, dim=-1)[t].item()
        ids = ids + [t]
    return chance


def percent(p: float) -> str:
    """Two significant digits for small chances (0.00014%), one decimal place otherwise (52.9%)."""
    pct = 100 * p
    if pct == 0:
        return "0%"
    decimals = 1 if pct >= 1 else min(12, 1 - math.floor(math.log10(pct)))
    return f"{pct:.{decimals}f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pretrain_final.pt")
    ap.add_argument("--tokenizer", default=None, help="default: the tokenizer recorded in the checkpoint")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--top", type=int, default=10, help="how many of the most likely tokens to list")
    ap.add_argument("--temperature", type=float, default=1.0, help="scores are divided by it before softmax")
    ap.add_argument("--word", action="append", default=[], help="also show this word's chance; can be repeated")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load_model(args.ckpt, device, args.tokenizer)
    ids = tok.encode(args.prompt)
    scores = next_token_scores(model, ids, device)
    probs = F.softmax(scores / args.temperature, dim=-1)

    print(f"{args.ckpt} | temperature {args.temperature}")
    print("prompt tokens: " + " ".join(repr(tok.decode([i])) for i in ids) + f" ({len(ids)})")
    print()
    print(f"{'':>6}  {'next token':<18}{'score':>8}  chance")
    top = torch.topk(probs, min(args.top, probs.numel()))
    for rank, (p, i) in enumerate(zip(top.values.tolist(), top.indices.tolist()), 1):
        print(f"{rank:>6}  {tok.decode([i])!r:<18}{scores[i].item():>8.2f}  {percent(p)}")
    for word in args.word:
        target = word_ids(tok, word)
        chance = word_chance(model, ids, target, args.temperature, device)
        label = repr(word if word in tok.special_tokens or word.startswith(" ") else " " + word)
        if len(target) == 1:
            print(f"{'word':>6}  {label:<18}{scores[target[0]].item():>8.2f}  {percent(chance)}")
        else:
            print(f"{'word':>6}  {label:<18}{'':>8}  {percent(chance)} ({len(target)} tokens)")


if __name__ == "__main__":
    main()
