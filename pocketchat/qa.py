"""Short-answer questions: prompts and scoring, as in the SQuAD evaluation.

Answers are compared after lowercasing and removing punctuation, the articles a/an/the and extra
spaces (the normalization of the official SQuAD script). Against a list of reference answers:
- exact: the answer is one of the references;
- f1: word overlap with the closest reference (harmonic mean of precision and recall);
- contains: one of the references appears, as whole words, inside the answer. It is lenient on
  purpose, because a small model answers in a full sentence ("It was completed in 1889."),
  which scores 0 on exact; it also rewards long answers that happen to include the reference.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from pocketchat.chat_format import prompt_ids
from pocketchat.tokenizer import BPETokenizer

_PUNCT = set(string.punctuation)


def normalize(text: str) -> str:
    text = "".join(ch for ch in text.lower() if ch not in _PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact(answer: str, refs: list[str]) -> float:
    a = normalize(answer)
    return float(any(a == normalize(r) for r in refs))


def f1(answer: str, refs: list[str]) -> float:
    words = normalize(answer).split()
    best = 0.0
    for r in refs:
        ref = normalize(r).split()
        common = sum((Counter(words) & Counter(ref)).values())
        if common:
            p, rec = common / len(words), common / len(ref)
            best = max(best, 2 * p * rec / (p + rec))
    return best


def contains(answer: str, refs: list[str]) -> float:
    a = f" {normalize(answer)} "
    return float(any(normalize(r) and f" {normalize(r)} " in a for r in refs))


def user_message(question: str, context: str | None) -> str:
    """The question alone (closed book) or after the passage it is answered from (open book)."""
    if context is None:
        return question
    return f"Read this text and answer the question.\n\n{context}\n\nQuestion: {question}"


def plain_text(question: str, context: str | None) -> str:
    """The same task as continuation text, for a model that has never seen the chat format."""
    head = "" if context is None else f"{context}\n\n"
    return f"{head}Question: {question}\nAnswer:"


def encode(tok: BPETokenizer, question: str, context: str | None, fmt: str) -> list[int]:
    if fmt == "chat":
        return prompt_ids(tok, [("user", user_message(question, context))])
    if fmt == "plain":
        return tok.encode(plain_text(question, context), allow_special=False)
    raise ValueError(f"unknown format {fmt!r}")
