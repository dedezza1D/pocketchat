"""Checkpoints that remember which tokenizer they were trained with.

Two tokenizers with the same vocabulary size are interchangeable as far as tensor shapes go,
so loading a model with the wrong one fails silently: the text just comes out scrambled.
Every checkpoint written by the training scripts records the tokenizer's path and SHA-256,
and `load_model` refuses a tokenizer whose hash does not match.
"""

import hashlib
import json
from pathlib import Path

import torch

from pocketchat.model import GPT, GPTConfig
from pocketchat.tokenizer import BPETokenizer


def tokenizer_record(path: str | Path) -> dict:
    """Path and SHA-256 of a tokenizer file, stored inside each checkpoint."""
    data = Path(path).read_bytes()
    return {"path": Path(path).as_posix(), "sha256": hashlib.sha256(data).hexdigest()}


def resolve_tokenizer(state: dict, override: str | Path | None = None) -> BPETokenizer:
    """The tokenizer a checkpoint was trained with, or `override` if its hash matches."""
    record = state.get("tokenizer")
    if override is None:
        if record is None:
            raise ValueError("this checkpoint does not record its tokenizer; pass --tokenizer")
        override = record["path"]
    if record is not None and tokenizer_record(override)["sha256"] != record["sha256"]:
        raise ValueError(
            f"tokenizer {override} does not match the one this checkpoint was trained with ({record['path']})"
        )
    return BPETokenizer.load(override)


DATA_RECORD = "tokenizer_record.json"


def write_data_record(data_dir: str | Path, tokenizer_path: str | Path) -> None:
    """Record, next to train.bin and val.bin, the tokenizer the token files were written with."""
    (Path(data_dir) / DATA_RECORD).write_text(json.dumps(tokenizer_record(tokenizer_path), indent=1))


def check_data_tokenizer(state: dict, data_dir: str | Path) -> bool:
    """Refuse token files written with a different tokenizer than the checkpoint.

    Returns False when there is nothing to compare (no record next to the data, or none in the
    checkpoint), True when both match, and raises ValueError when they differ.
    """
    path = Path(data_dir) / DATA_RECORD
    if "tokenizer" not in state or not path.exists():
        return False
    data = json.loads(path.read_text())
    if data["sha256"] != state["tokenizer"]["sha256"]:
        raise ValueError(
            f"{data_dir} was tokenized with {data['path']}, but the checkpoint was trained with {state['tokenizer']['path']}"
        )
    return True


def load_model(path: str | Path, device: str, tokenizer: str | Path | None = None) -> tuple[GPT, BPETokenizer, dict]:
    """Load a checkpoint with `config` and `model`, plus the tokenizer it was trained with."""
    state = torch.load(path, map_location=device)
    tok = resolve_tokenizer(state, tokenizer)
    model = GPT(GPTConfig(**state["config"])).to(device).eval()
    model.load_state_dict(state["model"])
    return model, tok, state
