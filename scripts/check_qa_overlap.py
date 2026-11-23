"""Check whether the SQuAD evaluation questions and paragraphs appear in the training data.

    uv run python scripts/check_qa_overlap.py > evals/qa_overlap.txt

For each paragraph, the share of its runs of --n consecutive tokens that also appear in the
training tokens (13 is close to the 13-word overlap used to check contamination in the GPT-3
paper). A few shared runs are common phrases; a share near 100% means the paragraph, or a copy of
it, was in the data. Pretraining (data/full/train.bin and val.bin) and fine-tuning
(data/sft/train.npz, val.npz) are checked separately, and the shares are written to --shares so
the evaluation can be split by them. Questions are also compared with the fine-tuning user prompts
after lowercasing and dropping punctuation, as in check_eval_overlap.py.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from check_eval_overlap import norm, user_prompts  # noqa: E402
from eval_qa import SQUAD_REVISION, select  # noqa: E402

from pocketchat.tokenizer import BPETokenizer  # noqa: E402

_MULT = np.uint64(1_000_003)


def ngram_hashes(tokens: np.ndarray, n: int) -> np.ndarray:
    """One 64-bit hash per run of n consecutive tokens (empty when there are fewer than n)."""
    t = np.asarray(tokens, dtype=np.uint64)
    if len(t) < n:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(len(t) - n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * _MULT + t[k : len(t) - n + 1 + k] + np.uint64(1)
    return h


def found(tokens: np.ndarray, queries: np.ndarray, n: int, chunk: int = 20_000_000) -> np.ndarray:
    """The query hashes that occur among the n-grams of `tokens`, read in overlapping chunks."""
    queries = np.unique(queries)
    hits = []
    for start in range(0, max(len(tokens) - n + 1, 1), chunk):
        part = np.asarray(tokens[start : start + chunk + n - 1])
        hits.append(queries[np.isin(queries, ngram_hashes(part, n))])
    return np.unique(np.concatenate(hits)) if hits else queries[:0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="data/tokenizer_full.json")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--questions", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=464, help="seq_len - max_new used by eval_qa.py")
    ap.add_argument("--shares", default="evals/qa_overlap.json", help="share of each paragraph's runs found, per source")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    rows = list(load_dataset("rajpurkar/squad", split="validation", revision=SQUAD_REVISION))
    picked = select(rows, tok, args.questions, args.seed, args.budget)
    per_paragraph = [ngram_hashes(np.array(tok.encode(r["context"]), dtype=np.uint16), args.n) for r in picked]
    queries = np.concatenate(per_paragraph)
    print(f"{len(picked)} questions, one per paragraph; a paragraph is seen when a run of {args.n} of its tokens appears")

    sources = {
        "pretraining": [np.memmap(p, dtype=np.uint16, mode="r") for p in ("data/full/train.bin", "data/full/val.bin")],
        "fine-tuning": [np.load(p)["x"] for p in ("data/sft/train.npz", "data/sft/val.npz")],
    }
    shares = {r["id"]: {} for r in picked}
    for name, arrays in sources.items():
        seen = np.unique(np.concatenate([found(a, queries, args.n) for a in arrays]))
        share = [float(np.isin(h, seen).mean()) if len(h) else 0.0 for h in per_paragraph]
        for r, s in zip(picked, share):
            shares[r["id"]][name] = round(s, 4)
        counts = ", ".join(f"{sum(s >= t for s in share) if t else sum(s > 0 for s in share)} with "
                           + (f"at least {t:.0%}" if t else "any run") for t in (0, 0.1, 0.5))
        print(f"{name}: paragraphs sharing {args.n}-token runs with the data: {counts}")
        for i in sorted(range(len(picked)), key=lambda i: -share[i]):
            if share[i] < 0.5:
                break
            print(f"  {picked[i]['id']} ({picked[i]['title']}): {share[i]:.0%} of its runs")
    Path(args.shares).write_text(json.dumps(shares, indent=0), encoding="utf-8")

    for split in ("train", "val"):
        prompts = [norm(p) for p in user_prompts(tok, Path(f"data/sft/{split}.npz"))]
        exact = set(prompts)
        same = [r["question"] for r in picked if norm(r["question"]) in exact]
        print(f"fine-tuning {split}: {len(same)} questions are also a user prompt" + (f": {same[:5]}" if same else ""))


if __name__ == "__main__":
    main()
