"""Encode the corpus into train/val token files (uint16), using all CPU cores.

Each document ends with <|endoftext|> so the model learns where texts stop.

    uv run python scripts/tokenize_corpus.py --corpus data/corpus_small.jsonl
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pocketchat.data import write_tokens
from pocketchat.tokenizer import BPETokenizer

_tok: BPETokenizer | None = None


def _init(path: str) -> None:
    global _tok
    _tok = BPETokenizer.load(path)


def _encode(line: str) -> np.ndarray:
    text = json.loads(line)["text"]
    ids = _tok.encode_ordinary(text) + [_tok.special_tokens["<|endoftext|>"]]
    return np.asarray(ids, dtype=np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-fraction", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    lines = Path(args.corpus).read_text(encoding="utf-8").splitlines()
    n_val = max(1, int(len(lines) * args.val_fraction))
    splits = {"val": lines[:n_val], "train": lines[n_val:]}

    with Pool(args.workers, initializer=_init, initargs=(args.tokenizer,)) as pool:
        for name, docs in splits.items():
            arrays = list(tqdm(pool.imap(_encode, docs, chunksize=64), total=len(docs), desc=name))
            ids = np.concatenate(arrays)
            write_tokens(Path(args.out_dir) / f"{name}.bin", ids)
            print(f"{name}: {len(docs)} docs, {len(ids) / 1e6:.2f}M tokens")


if __name__ == "__main__":
    main()
