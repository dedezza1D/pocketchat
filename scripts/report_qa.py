"""Summarize eval_qa.py results as Markdown, split by how much of each paragraph pretraining saw.

    uv run python scripts/report_qa.py evals/qa_pretrain.json evals/qa_sft_keep.json evals/qa_sft_reinit.json

The split uses evals/qa_overlap.json (scripts/check_qa_overlap.py): "unseen" paragraphs share
less than 10% of their 13-token runs with the pretraining data, "near copies" share 50% or more.
"""

import argparse
import json
from pathlib import Path

GROUPS = {"all": (0.0, 1.01), "unseen (< 10%)": (0.0, 0.1), "near copies (>= 50%)": (0.5, 1.01)}


def group_scores(answers: list[dict], shares: dict, mode: str, low: float, high: float) -> dict:
    rows = [a for a in answers if a["mode"] == mode and low <= shares[a["id"]]["pretraining"] < high]
    if not rows:
        return {"n": 0}
    return {"n": len(rows), **{k: sum(r[k] for r in rows) / len(rows) for k in ("exact", "f1", "contains")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--overlap", default="evals/qa_overlap.json")
    args = ap.parse_args()

    shares = json.loads(Path(args.overlap).read_text(encoding="utf-8"))
    print("| Model | Book | Questions | Exact | F1 | Contains |")
    print("|---|---|---|---|---|---|")
    for path in args.results:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        name = Path(data["ckpt"]).stem
        for mode in ("closed", "open"):
            for group, (low, high) in GROUPS.items():
                s = group_scores(data["answers"], shares, mode, low, high)
                if s["n"]:
                    print(f"| {name} | {mode}, {group} | {s['n']} | {s['exact']:.1%} | {s['f1']:.1%} | {s['contains']:.1%} |")


if __name__ == "__main__":
    main()
