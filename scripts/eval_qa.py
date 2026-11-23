"""Measure whether answers are right: short-answer questions from SQuAD v1.1 (CC BY-SA 4.0).

    uv run python scripts/eval_qa.py --ckpt checkpoints/sft_reinit.pt --name qa_sft_reinit
    uv run python scripts/eval_qa.py --ckpt checkpoints/pretrain_final.pt --name qa_pretrain --format plain

Each question is asked twice, with the same greedy decoding:
- closed book: the question alone, so the answer has to come from what the model learned;
- open book: the Wikipedia paragraph the answer comes from, then the question.

Questions come from the validation split, one per paragraph, drawn with a fixed seed and kept only
when the longest prompt still leaves room for the answer, so every checkpoint gets the same set.
Scores are exact match, F1 and "contains" (pocketchat/qa.py). --format plain asks a pretrained model
with "Question: ... Answer:" text and stops at the end of the line.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset

from pocketchat.chat_format import END, USER
from pocketchat.checkpoint import load_model
from pocketchat.qa import contains, encode, exact, f1

# the dataset revision the published results were measured on; datasets on the Hub can change
SQUAD_REVISION = "7b6d24c440a36b6815f21b70d25016731768db1f"
MODES = {"closed": False, "open": True}


def select(rows: list[dict], tok, n: int, seed: int, budget: int) -> list[dict]:
    """n questions from different paragraphs whose longest prompt fits in `budget` tokens."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    seen, picked = set(), []
    for i in order:
        r = rows[i]
        if r["context"] in seen:
            continue
        longest = max(len(encode(tok, r["question"], r["context"], fmt)) for fmt in ("chat", "plain"))
        if longest > budget:
            continue
        seen.add(r["context"])
        picked.append(r)
        if len(picked) == n:
            break
    return picked


def line_break_ids(tok) -> set[int]:
    """Tokens whose text contains a line break: a plain-text answer ends at the first one."""
    specials = set(tok.special_tokens.values())
    return {i for i in range(tok.vocab_size) if i not in specials and "\n" in tok.decode([i])}


@torch.no_grad()
def answer(model, tok, ids: list[int], fmt: str, max_new: int, stop: set[int], device: str) -> str:
    x = torch.tensor([ids], device=device)
    with torch.autocast("cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model.generate(x, max_new, temperature=1.0, top_k=1, stop_ids=stop)
    text = tok.decode([t for t in out[0, x.shape[1]:].tolist() if t not in tok.special_tokens.values()])
    if fmt == "plain":
        text = text.split("\n")[0]
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--format", choices=["chat", "plain"], default="chat")
    ap.add_argument("--tokenizer", default=None, help="default: the tokenizer recorded in the checkpoint")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--out-dir", default="evals")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load_model(args.ckpt, device, args.tokenizer)
    rows = list(load_dataset("rajpurkar/squad", split="validation", revision=SQUAD_REVISION))
    questions = select(rows, tok, args.n, args.seed, model.cfg.seq_len - args.max_new)
    if args.format == "chat":
        stop = {tok.special_tokens[END], tok.special_tokens[USER]}
    else:
        stop = line_break_ids(tok) | {tok.special_tokens["<|endoftext|>"]}

    records, summary = [], {}
    for mode, with_context in MODES.items():
        scores = {"exact": 0.0, "f1": 0.0, "contains": 0.0, "words": 0.0}
        for r in questions:
            refs = r["answers"]["text"]
            ids = encode(tok, r["question"], r["context"] if with_context else None, args.format)
            a = answer(model, tok, ids, args.format, args.max_new, stop, device)
            row = {"mode": mode, "id": r["id"], "question": r["question"], "references": sorted(set(refs)),
                   "answer": a, "exact": exact(a, refs), "f1": f1(a, refs), "contains": contains(a, refs)}
            records.append(row)
            for k in ("exact", "f1", "contains"):
                scores[k] += row[k]
            scores["words"] += len(a.split())
        summary[mode] = {k: v / len(questions) for k, v in scores.items()}

    settings = {"n": len(questions), "seed": args.seed, "max_new": args.max_new, "format": args.format,
                "decoding": "greedy", "squad_revision": SQUAD_REVISION}
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(
        {"ckpt": args.ckpt, "settings": settings, "summary": summary, "answers": records}, indent=1), encoding="utf-8")
    md = [f"# {args.name}", "", f"`{args.ckpt}`, {len(questions)} SQuAD v1.1 validation questions, greedy, "
          f"format {args.format}", "", "| Mode | Exact | F1 | Contains | Words per answer |", "|---|---|---|---|---|"]
    for mode, s in summary.items():
        md.append(f"| {mode} book | {s['exact']:.1%} | {s['f1']:.1%} | {s['contains']:.1%} | {s['words']:.1f} |")
    md += ["", "The first 20 questions in each mode:", ""]
    for r in records[:20] + [x for x in records if x["mode"] == "open"][:20]:
        md += [f"**{r['mode']} · Q:** {r['question']}", "", f"**A:** {r['answer']}", "",
               f"*references: {' / '.join(r['references'])} · exact {r['exact']:.0f} · f1 {r['f1']:.2f} · "
               f"contains {r['contains']:.0f}*", "", "---", ""]
    (out / f"{args.name}.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"ckpt": args.ckpt, **summary}, indent=1))


if __name__ == "__main__":
    main()
