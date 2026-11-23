"""Answer a fixed set of questions and measure the behaviour that SFT is meant to change.

    uv run python scripts/eval_chat.py --ckpt checkpoints/sft_reinit.pt --name sft_reinit

Metrics per checkpoint (same questions, same seed, same sampling settings):
- ended: share of answers that close with <|end|> before the token limit
- length: mean answer length in tokens
- repetition: share of repeated 3-grams inside an answer (0 = no repeated 3-gram)
"""

import argparse
import json
from pathlib import Path

import torch

from pocketchat.chat_format import END, USER, prompt_ids
from pocketchat.checkpoint import load_model

QUESTIONS = [
    "What is the water cycle?",
    "Explain photosynthesis in simple terms.",
    "What is the capital of France?",
    "Give me three tips to sleep better.",
    "What is a black hole?",
    "How do vaccines work?",
    "Write a short poem about the ocean.",
    "What is the difference between weather and climate?",
    "Why is the sky blue?",
    "How can I improve my writing skills?",
    "What causes earthquakes?",
    "Summarize the benefits of regular exercise.",
    "What is machine learning?",
    "How does a computer store information?",
    "What should I pack for a trip to the mountains?",
    "Explain what inflation is.",
    "What are the planets in the solar system?",
    "How do I make a cup of tea?",
    "What is democracy?",
    "Give me a few ideas for a birthday party.",
    "Why do leaves change color in autumn?",
    "What is the role of the heart in the human body?",
    "How do bees make honey?",
    "What is a good way to learn a new language?",
    "Explain the difference between a virus and bacteria.",
    "What is renewable energy?",
    "How does the internet work?",
    "What were the main causes of World War I?",
    "Recommend a healthy breakfast.",
    "What is the Pythagorean theorem?",
]


def repetition(ids: list[int]) -> float:
    grams = [tuple(ids[i : i + 3]) for i in range(len(ids) - 2)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokenizer", default=None, help="default: the tokenizer recorded in the checkpoint")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-dir", default="evals")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, state = load_model(args.ckpt, device, args.tokenizer)
    end_id, user_id = tok.special_tokens[END], tok.special_tokens[USER]

    rows = []
    for q in QUESTIONS:
        g = torch.Generator(device=device).manual_seed(args.seed)
        prompt = torch.tensor([prompt_ids(tok, [("user", q)])], device=device)
        with torch.autocast("cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model.generate(prompt, args.max_new, args.temperature, top_k=None, stop_ids={end_id, user_id},
                                 top_p=args.top_p, repetition_penalty=args.repetition_penalty, generator=g)
        new = out[0, prompt.shape[1]:].tolist()
        ended = bool(new) and new[-1] == end_id
        body = [t for t in new if t not in (end_id, user_id)]
        rows.append({"question": q, "answer": tok.decode(body), "ended": ended,
                     "length": len(body), "repetition": repetition(body)})

    summary = {
        "name": args.name,
        "ckpt": args.ckpt,
        "ended": sum(r["ended"] for r in rows) / len(rows),
        "length": sum(r["length"] for r in rows) / len(rows),
        "repetition": sum(r["repetition"] for r in rows) / len(rows),
        "settings": {k: getattr(args, k) for k in ("temperature", "top_p", "repetition_penalty", "max_new", "seed")},
    }
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps({"summary": summary, "answers": rows}, indent=1))
    md = [f"# {args.name}", "", f"ended {summary['ended']:.0%} · length {summary['length']:.0f} tokens · "
          f"repetition {summary['repetition']:.1%}", ""]
    for r in rows:
        md += [f"**Q:** {r['question']}", "", f"**A:** {r['answer']}", "", "---", ""]
    (out / f"{args.name}.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
