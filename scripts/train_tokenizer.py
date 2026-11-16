"""Train the BPE tokenizer on part of the corpus.

    uv run python scripts/train_tokenizer.py --corpus data/corpus_small.jsonl --docs 5000
"""

import argparse
import json
import time
from itertools import islice
from pathlib import Path

from pocketchat.tokenizer import BPETokenizer

SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--docs", type=int, default=20000, help="documents used to learn merges")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--out", default="data/tokenizer.json")
    args = ap.parse_args()

    with Path(args.corpus).open(encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in islice(f, args.docs)]

    t0 = time.time()
    tok = BPETokenizer.train(texts, args.vocab_size, SPECIAL_TOKENS, verbose=True)
    tok.save(args.out)
    sample = texts[0][:300]
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample
    chars = sum(len(t) for t in texts[:200])
    tokens = sum(len(tok.encode(t)) for t in texts[:200])
    print(f"trained {tok.vocab_size} tokens in {time.time() - t0:.0f}s -> {args.out}")
    print(f"compression: {chars / tokens:.2f} chars per token")
    print("sample tokens:", [tok.decode([i]) for i in ids[:25]])


if __name__ == "__main__":
    main()
