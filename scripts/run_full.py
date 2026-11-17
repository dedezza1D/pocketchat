"""Run the full pretraining pipeline end to end: download, tokenizer, tokenization, pretraining.

On Windows it asks the OS not to idle-sleep while it runs (released when the process exits).
Steps whose output already exists are skipped, and pretraining resumes from its last checkpoint,
so the script can be restarted after an interruption.

    uv run python scripts/run_full.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


def keep_awake() -> None:
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def run(*args: str) -> None:
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=400_000)
    ap.add_argument("--tokenizer-docs", type=int, default=40_000)
    ap.add_argument("--max-steps", type=int, default=15_000)
    args = ap.parse_args()
    keep_awake()

    corpus = Path("data/corpus.jsonl")
    if not corpus.exists():
        run("scripts/download_corpus.py", "--docs", str(args.docs), "--out", str(corpus))
    if not Path("data/tokenizer_full.json").exists():
        run("scripts/train_tokenizer.py", "--corpus", str(corpus), "--docs", str(args.tokenizer_docs),
            "--out", "data/tokenizer_full.json")
    if not Path("data/full/train.bin").exists():
        Path("data/full").mkdir(parents=True, exist_ok=True)
        run("scripts/tokenize_corpus.py", "--corpus", str(corpus), "--tokenizer", "data/tokenizer_full.json",
            "--out-dir", "data/full")
    run("scripts/pretrain.py", "--data-dir", "data/full", "--tokenizer", "data/tokenizer_full.json",
        "--max-steps", str(args.max_steps), "--warmup", "300", "--eval-every", "500",
        "--out", "checkpoints/pretrain.pt", "--resume")


if __name__ == "__main__":
    main()
