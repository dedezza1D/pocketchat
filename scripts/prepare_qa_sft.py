"""Fine-tuning data that teaches the model to answer from a passage: SQuAD v1.1 train (CC BY-SA 4.0).

    uv run python scripts/prepare_qa_sft.py
    uv run python scripts/sft.py --init checkpoints/sft_reinit.pt --data-dir data/sft_read --out checkpoints/sft_read.pt --epochs 1

Each SQuAD question becomes one conversation in the open-book format of eval_qa.py: the user message
is the paragraph followed by the question, and the assistant replies with the short reference answer
and <|end|>, so the model learns to answer with the span and stop. The chat conversations from
data/sft are added (--mix-chat) so the model keeps answering ordinary prompts.

The validation set holds out whole Wikipedia articles (titles) of the train split, so validation
paragraphs never share an article with training ones; eval_qa.py uses the SQuAD validation split,
whose articles are separate from both.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from pocketchat.chat_format import encode_conversation, to_example
from pocketchat.qa import user_message
from pocketchat.tokenizer import BPETokenizer

# the dataset revision used for evaluation as well (scripts/eval_qa.py)
SQUAD_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
SQUAD, CHAT = 2, 3  # source ids, after oasst1 (0) and dolly (1) in data/sft


def squad_examples(rows, tok, max_len: int) -> list[tuple[str, tuple[np.ndarray, np.ndarray]]]:
    """(title, example) for every question whose prompt and answer fit in max_len tokens."""
    out = []
    for r in rows:
        turns = [("user", user_message(r["question"], r["context"])), ("assistant", r["answers"]["text"][0].strip())]
        ids, train = encode_conversation(tok, turns)
        if len(ids) > max_len + 1:
            continue  # a cut would drop the answer, the only trained part
        ex = to_example(ids, train, max_len)
        if ex is not None:
            out.append((r["title"], ex))
    return out


def split_by_title(examples, val_fraction: float, seed: int):
    titles = sorted({t for t, _ in examples})
    rng = np.random.default_rng(seed)
    held = set(rng.choice(titles, size=max(1, int(len(titles) * val_fraction)), replace=False))
    train = [ex for t, ex in examples if t not in held]
    val = [ex for t, ex in examples if t in held]
    return train, val, len(held)


def chat_examples(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    d = np.load(path)
    x, y, off = d["x"], d["y"], d["offsets"]
    return [(x[off[i] : off[i + 1]].astype(np.int64), y[off[i] : off[i + 1]].astype(np.int64)) for i in range(len(off) - 1)]


def save(path: Path, examples: list[tuple[int, tuple[np.ndarray, np.ndarray]]]) -> None:
    x = np.concatenate([e[1][0] for e in examples]).astype(np.uint16)
    y = np.concatenate([e[1][1] for e in examples]).astype(np.int32)
    offsets = np.cumsum([0] + [len(e[1][0]) for e in examples]).astype(np.int64)
    source = np.asarray([e[0] for e in examples], dtype=np.int8)
    np.savez(path, x=x, y=y, offsets=offsets, source=source)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="data/tokenizer_full.json")
    ap.add_argument("--out-dir", default="data/sft_read")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--val-fraction", type=float, default=0.02, help="share of train articles held out")
    ap.add_argument("--mix-chat", default="data/sft", help="directory with chat train.npz/val.npz; empty to skip")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    rows = list(load_dataset("rajpurkar/squad", split="train", revision=SQUAD_REVISION))
    squad = squad_examples(rows, tok, args.max_len)
    train_q, val_q, n_held = split_by_title(squad, args.val_fraction, args.seed)
    train = [(SQUAD, ex) for ex in train_q]
    val = [(SQUAD, ex) for ex in val_q]
    stats = {"squad_questions": len(rows), "squad_kept": len(squad), "held_out_articles": n_held,
             "squad_train": len(train_q), "squad_val": len(val_q)}
    if args.mix_chat:
        for split, dest in (("train", train), ("val", val)):
            chat = chat_examples(Path(args.mix_chat) / f"{split}.npz")
            dest += [(CHAT, ex) for ex in chat]
            stats[f"chat_{split}"] = len(chat)

    rng = np.random.default_rng(args.seed)
    train = [train[i] for i in rng.permutation(len(train))]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "train.npz", train)
    save(out / "val.npz", val)
    stats.update(train_examples=len(train), val_examples=len(val),
                 train_tokens=int(sum(len(e[1][0]) for e in train)),
                 trained_tokens=int(sum((e[1][1] != -100).sum() for e in train)))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
