"""Chat formatting shared by fine-tuning, evaluation and the chat CLI.

A conversation is a list of (role, text) turns, role in {"user", "assistant"}, encoded as

    <|user|>text<|end|><|assistant|>text<|end|><|user|>...

Only assistant text and the <|end|> that closes an assistant turn are trained on:
the model learns to answer and to stop, not to write the user's side.
Turn text is encoded with allow_special=False, so a string such as "<|assistant|>"
inside a message can never become a control token.
"""

from __future__ import annotations

import numpy as np

from pocketchat.tokenizer import BPETokenizer

USER, ASSISTANT, END = "<|user|>", "<|assistant|>", "<|end|>"
IGNORE = -100


def encode_conversation(tok: BPETokenizer, turns: list[tuple[str, str]]) -> tuple[list[int], list[bool]]:
    """Token ids and, per token, whether the model should learn to predict it."""
    sp = tok.special_tokens
    ids: list[int] = []
    train: list[bool] = []
    for role, text in turns:
        body = tok.encode(text, allow_special=False)
        if role == "user":
            seg = [sp[USER], *body, sp[END]]
            ids += seg
            train += [False] * len(seg)
        elif role == "assistant":
            ids += [sp[ASSISTANT], *body, sp[END]]
            train += [False] + [True] * len(body) + [True]
        else:
            raise ValueError(f"unknown role {role!r}")
    return ids, train


def prompt_ids(tok: BPETokenizer, turns: list[tuple[str, str]]) -> list[int]:
    """Encode the history and open an assistant turn for the model to complete."""
    ids, _ = encode_conversation(tok, turns)
    return ids + [tok.special_tokens[ASSISTANT]]


def to_example(ids: list[int], train: list[bool], max_len: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Truncate to max_len + 1 tokens and build (input, target) with non-trained targets ignored.

    Returns None when nothing trainable survives truncation.
    """
    ids, train = ids[: max_len + 1], train[: max_len + 1]
    if len(ids) < 2 or not any(train[1:]):
        return None
    x = np.asarray(ids[:-1], dtype=np.int64)
    y = np.asarray(ids[1:], dtype=np.int64)
    y[~np.asarray(train[1:], dtype=bool)] = IGNORE
    return x, y
