"""Stream English documents from FineWeb-Edu (ODC-By license) into a local JSONL file.

FineWeb-Edu is web text filtered for educational value, which gives small models
cleaner language than raw web crawl. Streaming avoids downloading the full dataset.

    uv run python scripts/download_corpus.py --docs 5000 --out data/corpus_small.jsonl
"""

import argparse
import json
import logging
from contextlib import closing
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# the dataset revision the published model was trained on; datasets on the Hub can change
FINEWEB_EDU_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=5000)
    ap.add_argument("--out", default="data/corpus.jsonl")
    ap.add_argument("--min-chars", type=int, default=200)
    args = ap.parse_args()

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True,
                      revision=FINEWEB_EDU_REVISION)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = chars = 0
    with closing(iter(ds)) as rows, out.open("w", encoding="utf-8") as f, tqdm(total=args.docs, unit="doc") as bar:
        for row in rows:
            text = row["text"].strip()
            if len(text) < args.min_chars:
                continue
            f.write(json.dumps({"text": text}) + "\n")
            written += 1
            chars += len(text)
            bar.update(1)
            if written >= args.docs:
                break
        # Stopping mid-file leaves a pyarrow read of the next block in flight. When the stream closes, or the
        # interpreter exits, its connection is closed and huggingface_hub logs a failed request and a retry,
        # although every document is already written. Those reads are no longer needed, so their warnings are hidden.
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    print(f"wrote {written} docs, {chars / 1e6:.1f}M chars to {out}")


if __name__ == "__main__":
    main()
