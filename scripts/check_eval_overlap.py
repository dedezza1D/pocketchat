"""Check whether the fixed evaluation questions appear among the fine-tuning prompts.

    uv run python scripts/check_eval_overlap.py

Prints, for the train and validation conversations, the evaluation questions that appear as a user
prompt (after lowercasing and dropping punctuation) and how many prompts contain each question.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_chat import QUESTIONS  # noqa: E402

from pocketchat.tokenizer import BPETokenizer  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def user_prompts(tok: BPETokenizer, path: Path) -> list[str]:
    d = np.load(path)
    x, off = d["x"], d["offsets"]
    user, end = tok.special_tokens["<|user|>"], tok.special_tokens["<|end|>"]
    prompts = []
    for i in range(len(off) - 1):
        ids = x[off[i] : off[i + 1]].tolist()
        j = 0
        while j < len(ids):
            if ids[j] == user:
                k = j + 1
                while k < len(ids) and ids[k] != end:
                    k += 1
                prompts.append(tok.decode(ids[j + 1 : k]))
                j = k
            j += 1
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="data/tokenizer_full.json")
    ap.add_argument("--data-dir", default="data/sft")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    for split in ("train", "val"):
        prompts = [norm(p) for p in user_prompts(tok, Path(args.data_dir) / f"{split}.npz")]
        exact = set(prompts)
        print(f"{split}: {len(prompts)} user prompts")
        print("  same prompt as an eval question:", [q for q in QUESTIONS if norm(q) in exact] or "none")
        contained = {q: n for q in QUESTIONS if (n := sum(norm(q) in p for p in prompts))}
        print("  prompts containing an eval question:", contained or "none")


if __name__ == "__main__":
    main()
