"""Byte-level BPE tokenizer, trained from scratch.

Text is first split into chunks with a GPT-4-style regex (words, numbers,
punctuation and whitespace stay apart), each chunk is turned into UTF-8 bytes,
and the most frequent adjacent pair is merged into a new token until the
vocabulary reaches the requested size. Because the base alphabet is the 256
byte values, any string can be encoded: there is no unknown token.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import regex as re

SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
_SPLIT = re.compile(SPLIT_PATTERN)


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: dict[tuple[int, int], int], special_tokens: dict[str, int]):
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in sorted(merges.items(), key=lambda kv: kv[1]):
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.inverse_special = {v: k for k, v in special_tokens.items()}
        self._special_split = (
            re.compile("(" + "|".join(re.escape(s) for s in special_tokens) + ")")
            if special_tokens
            else None
        )
        self._encode_chunk = lru_cache(maxsize=1 << 20)(self._encode_chunk_uncached)

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    # ------------------------------------------------------------------ training
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: Iterable[str] = (),
        verbose: bool = False,
    ) -> "BPETokenizer":
        special_tokens = list(special_tokens)
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError("vocab_size is smaller than 256 bytes plus special tokens")

        chunk_counts: Counter[str] = Counter()
        for text in texts:
            chunk_counts.update(_SPLIT.findall(text))

        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for i, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[i]
                where[pair].add(i)

        # Max-heap with lazy deletion: stale entries are skipped when popped.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: dict[tuple[int, int], int] = {}
        next_id = 256
        while len(merges) < n_merges and heap:
            neg, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0)
            if count != -neg:
                continue
            if count < 2:
                break
            merges[pair] = next_id
            changed: set[tuple[int, int]] = set()
            for i in where.pop(pair, ()):
                old = words[i]
                new = _merge(old, pair, next_id)
                if len(new) == len(old):
                    continue  # stale index: this word no longer contains the pair
                f = freqs[i]
                for p in zip(old, old[1:]):
                    pair_counts[p] -= f
                    changed.add(p)
                for p in zip(new, new[1:]):
                    pair_counts[p] += f
                    where[p].add(i)
                    changed.add(p)
                words[i] = new
            pair_counts.pop(pair, None)
            changed.discard(pair)
            for p in changed:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)
            if verbose and len(merges) % 500 == 0:
                print(f"  {len(merges)}/{n_merges} merges")
            next_id += 1

        specials = {tok: next_id + k for k, tok in enumerate(special_tokens)}
        return cls(merges, specials)

    # ------------------------------------------------------------------ encoding
    def _encode_chunk_uncached(self, chunk: str) -> tuple[int, ...]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # pick the adjacent pair that was learned earliest
            best = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = _merge(ids, best, self.merges[best])
        return tuple(ids)

    def encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in _SPLIT.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        if not allow_special or self._special_split is None:
            return self.encode_ordinary(text)
        out: list[int] = []
        for part in self._special_split.split(text):
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        parts: list[bytes] = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self.inverse_special:
                parts.append(self.inverse_special[i].encode("utf-8"))
            else:
                raise ValueError(f"unknown token id {i}")
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        data = {
            "merges": [[a, b, new] for (a, b), new in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = {(a, b): new for a, b, new in data["merges"]}
        return cls(merges, data["special_tokens"])
