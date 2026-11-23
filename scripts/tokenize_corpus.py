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

from pocketchat.checkpoint import write_data_record
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

    with Path(args.corpus).open(encoding="utf-8") as f:
        n_docs = sum(1 for _ in f)
    n_val = max(1, int(n_docs * args.val_fraction))

    def lines():
        with Path(args.corpus).open(encoding="utf-8") as f:
            yield from f

    out = {"val": [], "train": []}
    with Pool(args.workers, initializer=_init, initargs=(args.tokenizer,)) as pool:
        for i, arr in enumerate(tqdm(pool.imap(_encode, lines(), chunksize=64), total=n_docs, unit="doc")):
            out["val" if i < n_val else "train"].append(arr)
    for name, arrays in out.items():
        ids = np.concatenate(arrays)
        write_tokens(Path(args.out_dir) / f"{name}.bin", ids)
        print(f"{name}: {len(arrays)} docs, {len(ids) / 1e6:.2f}M tokens")
    write_data_record(args.out_dir, args.tokenizer)


if __name__ == "__main__":
    main()
