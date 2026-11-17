"""Build the supervised fine-tuning set from two human-written English datasets.

- OpenAssistant oasst1 (Apache-2.0): conversation trees; we follow the best-ranked
  assistant reply at every level, giving one multi-turn conversation per tree.
- databricks-dolly-15k (CC BY-SA 3.0): single-turn instruction/response pairs,
  written by people who were told not to use generative AI.

    uv run python scripts/prepare_sft.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

from pocketchat.chat_format import encode_conversation, to_example
from pocketchat.tokenizer import BPETokenizer


def oasst_conversations(max_turns: int) -> list[list[tuple[str, str]]]:
    rows = []
    for split in ("train", "validation"):
        rows += list(load_dataset("OpenAssistant/oasst1", split=split))
    msgs = {r["message_id"]: r for r in rows if r["lang"] == "en" and not r["deleted"]}
    children = defaultdict(list)
    for r in msgs.values():
        if r["parent_id"] in msgs:
            children[r["parent_id"]].append(r)

    def best(kids, role):
        kids = [k for k in kids if k["role"] == role and k.get("review_result") is not False]
        if not kids:
            return None
        return min(kids, key=lambda k: (k["rank"] is None, k["rank"] if k["rank"] is not None else 0))

    convos = []
    for root in msgs.values():
        if root["parent_id"] is not None or root["role"] != "prompter":
            continue
        turns, node = [], root
        while node is not None and len(turns) < 2 * max_turns:
            turns.append(("user", node["text"].strip()))
            reply = best(children[node["message_id"]], "assistant")
            if reply is None:
                turns.pop()
                break
            turns.append(("assistant", reply["text"].strip()))
            node = best(children[reply["message_id"]], "prompter")
        if turns:
            convos.append(turns)
    return convos


def dolly_conversations() -> list[list[tuple[str, str]]]:
    convos = []
    for r in load_dataset("databricks/databricks-dolly-15k", split="train"):
        user = r["instruction"].strip()
        if r["context"].strip():
            user += "\n\n" + r["context"].strip()
        convos.append([("user", user), ("assistant", r["response"].strip())])
    return convos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="data/tokenizer_full.json")
    ap.add_argument("--out-dir", default="data/sft")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-turns", type=int, default=3, help="user/assistant pairs per oasst conversation")
    ap.add_argument("--val-fraction", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    sources = {"oasst1": oasst_conversations(args.max_turns), "dolly": dolly_conversations()}

    examples, stats = [], {}
    for name, convos in sources.items():
        kept = trained = total = truncated = 0
        for turns in convos:
            ids, train = encode_conversation(tok, turns)
            ex = to_example(ids, train, args.max_len)
            if ex is None:
                continue
            kept += 1
            truncated += len(ids) > args.max_len + 1
            total += len(ex[0])
            trained += int((ex[1] != -100).sum())
            examples.append((name, ex))
        stats[name] = {
            "conversations": len(convos),
            "kept": kept,
            "truncated": truncated,
            "tokens": total,
            "trained_tokens": trained,
        }

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(examples))
    n_val = int(len(examples) * args.val_fraction)
    splits = {"val": [examples[i] for i in order[:n_val]], "train": [examples[i] for i in order[n_val:]]}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, exs in splits.items():
        x = np.concatenate([e[1][0] for e in exs]).astype(np.uint16)
        y = np.concatenate([e[1][1] for e in exs]).astype(np.int32)
        offsets = np.cumsum([0] + [len(e[1][0]) for e in exs]).astype(np.int64)
        source = np.asarray([0 if e[0] == "oasst1" else 1 for e in exs], dtype=np.int8)
        np.savez(out / f"{split}.npz", x=x, y=y, offsets=offsets, source=source)
        stats[f"{split}_examples"] = len(exs)
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
