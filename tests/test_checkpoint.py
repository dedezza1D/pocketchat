import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pocketchat.checkpoint import load_model, tokenizer_record
from pocketchat.data import write_tokens
from pocketchat.model import GPT, GPTConfig
from pocketchat.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]
TEXT = "Water is made of hydrogen and oxygen. Stars are made of hydrogen and helium. " * 30


def _tokenizers(tmp_path):
    a = BPETokenizer.train([TEXT], vocab_size=300, special_tokens=["<|endoftext|>"])
    b = BPETokenizer.train([TEXT.upper()], vocab_size=300, special_tokens=["<|endoftext|>"])
    a.save(tmp_path / "a.json")
    b.save(tmp_path / "b.json")
    return a, tmp_path / "a.json", tmp_path / "b.json"


def _save_tiny(path, tok, record):
    cfg = GPTConfig(vocab_size=tok.vocab_size, seq_len=16, n_layer=1, n_head=2, n_embd=32)
    state = {"config": cfg.to_dict(), "model": GPT(cfg).state_dict()}
    if record is not None:
        state["tokenizer"] = record
    torch.save(state, path)


def test_checkpoint_loads_its_own_tokenizer_and_rejects_another(tmp_path):
    tok, a, b = _tokenizers(tmp_path)
    _save_tiny(tmp_path / "m.pt", tok, tokenizer_record(a))

    _, loaded, _ = load_model(tmp_path / "m.pt", "cpu")
    assert loaded.encode(TEXT) == tok.encode(TEXT)
    load_model(tmp_path / "m.pt", "cpu", a)  # same file passed explicitly is fine
    with pytest.raises(ValueError, match="does not match"):
        load_model(tmp_path / "m.pt", "cpu", b)


def test_checkpoint_without_tokenizer_record_needs_one(tmp_path):
    tok, a, _ = _tokenizers(tmp_path)
    _save_tiny(tmp_path / "old.pt", tok, None)
    with pytest.raises(ValueError, match="pass --tokenizer"):
        load_model(tmp_path / "old.pt", "cpu")
    load_model(tmp_path / "old.pt", "cpu", a)


def test_pretrain_writes_the_final_checkpoint_fine_tuning_starts_from(tmp_path):
    tok, a, _ = _tokenizers(tmp_path)
    ids = np.array(tok.encode(TEXT * 4), dtype=np.int64)
    write_tokens(tmp_path / "train.bin", ids)
    write_tokens(tmp_path / "val.bin", ids[:400])
    out = tmp_path / "run.pt"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pretrain.py"), "--data-dir", str(tmp_path), "--tokenizer", str(a),
         "--out", str(out), "--n-layer", "1", "--n-head", "2", "--n-embd", "32", "--seq-len", "16",
         "--batch-size", "2", "--grad-accum", "1", "--max-steps", "4", "--warmup", "1", "--eval-every", "2"],
        check=True, cwd=ROOT, capture_output=True,
    )

    final = torch.load(tmp_path / "run_final.pt", map_location="cpu")
    assert final["step"] == 3
    assert final["tokenizer"] == tokenizer_record(a)
    model, _, _ = load_model(tmp_path / "run_final.pt", "cpu")
    assert model.cfg.n_embd == 32
