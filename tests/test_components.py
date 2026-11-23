"""Tests for the parts that are easy to get wrong without noticing: BPE training, sampling,
resuming, schedules, the fine-tuning batches and loss, RoPE, data helpers, the chat history and
the scripts readers use to get the weights and reproduce the numbers in the blog series."""

import os
import random
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pocketchat.chat_format import IGNORE, encode_conversation, prompt_ids, to_example
from pocketchat.checkpoint import check_data_tokenizer, tokenizer_record, write_data_record
from pocketchat.data import write_tokens
from pocketchat.model import GPT, GPTConfig, apply_rope, rope_tables
from pocketchat import qa
from pocketchat.tokenizer import _SPLIT, BPETokenizer, _merge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if "datasets" not in sys.modules:
    try:
        import datasets  # noqa: F401
    except ImportError:  # the CI installs only what the tests need; prepare_sft imports it at the top
        sys.modules["datasets"] = types.SimpleNamespace(load_dataset=None)

import chat  # noqa: E402
import check_eval_overlap  # noqa: E402
import download_weights  # noqa: E402
import eval_qa  # noqa: E402
import next_word  # noqa: E402
import prepare_sft  # noqa: E402
import sft  # noqa: E402
from pretrain import lr_at  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>"]
TEXT = "Water is made of hydrogen and oxygen. Stars are made of hydrogen and helium. " * 30


def chat_tokenizer():
    return BPETokenizer.train([TEXT, "What is water made of? Why is the sky blue? " * 20], vocab_size=320,
                              special_tokens=SPECIALS)


def tiny_gpt(vocab_size, seq_len=32):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=vocab_size, seq_len=seq_len, n_layer=1, n_head=2, n_embd=32)).eval()


# T1 ------------------------------------------------------------------------------------------------
def reference_merges(texts, n_merges):
    """Textbook BPE: recount every pair after every merge; ties go to the smallest pair."""
    counts = Counter()
    for t in texts:
        counts.update(_SPLIT.findall(t))
    words, freqs = [list(c.encode("utf-8")) for c in counts], list(counts.values())
    merges, next_id = {}, 256
    while len(merges) < n_merges:
        pairs = Counter()
        for w, f in zip(words, freqs):
            for p in zip(w, w[1:]):
                pairs[p] += f
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        if pairs[best] < 2:
            break
        merges[best] = next_id
        words = [_merge(w, best, next_id) for w in words]
        next_id += 1
    return merges


@pytest.mark.parametrize("seed", range(12))
def test_fast_bpe_training_matches_textbook_bpe(seed):
    rng = random.Random(seed)
    texts = ["".join(rng.choice("aab c") for _ in range(rng.randint(5, 40))) for _ in range(30)]
    texts += ["aaaa aaaa aaa", "abab abab", TEXT[: rng.randint(20, 200)]]
    n = 60
    assert BPETokenizer.train(texts, vocab_size=256 + n).merges == reference_merges(texts, n)


# T2 ------------------------------------------------------------------------------------------------
class FixedLogits(GPT):
    """A model whose next-token scores never change, to test the sampling code alone."""

    def __init__(self, logits):
        super().__init__(GPTConfig(vocab_size=len(logits), seq_len=16, n_layer=1, n_head=1, n_embd=8))
        self.fixed = torch.tensor(logits, dtype=torch.float32)

    def forward(self, idx, targets=None, ignore_index=-100):
        return self.fixed.expand(idx.shape[0], idx.shape[1], -1).clone(), None


def generated(model, n, **kw):
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), n, **kw)
    return out[0, 1:].tolist()


def test_generate_top_k_1_is_greedy():
    assert generated(FixedLogits([0.0, 3.0, 1.0, 2.0]), 5, temperature=1.0, top_k=1) == [1] * 5


def test_generate_top_p_keeps_only_the_head_of_the_distribution():
    model = FixedLogits(np.log([0.5, 0.3, 0.15, 0.05]).tolist())
    draws = {generated(model, 1, temperature=1.0, top_k=None, top_p=0.7,
                       generator=torch.Generator().manual_seed(s))[0] for s in range(200)}
    assert draws == {0, 1}


@pytest.mark.parametrize("logits", [[2.0, 1.9, -5.0, -5.0], [-1.0, -1.5, -9.0, -9.0]])
def test_generate_repetition_penalty_lowers_tokens_already_generated(logits):
    # positive scores are divided by the penalty, negative ones multiplied: both move down
    assert generated(FixedLogits(logits), 2, temperature=1.0, top_k=1, repetition_penalty=2.0) == [0, 1]


def test_generate_stops_at_a_stop_token_and_repeats_with_the_same_seed():
    assert generated(FixedLogits([0.0, 0.0, 5.0, 0.0]), 5, temperature=1.0, top_k=1, stop_ids={2}) == [2]
    model = FixedLogits([0.1, 0.2, 0.3, 0.4])
    first = generated(model, 20, temperature=1.0, top_k=None, generator=torch.Generator().manual_seed(7))
    again = generated(model, 20, temperature=1.0, top_k=None, generator=torch.Generator().manual_seed(7))
    assert first == again


# T3 ------------------------------------------------------------------------------------------------
def run_pretrain(tmp_path, data_dir, tok_path, out, *extra):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pretrain.py"), "--data-dir", str(data_dir), "--tokenizer", str(tok_path),
         "--out", str(out), "--n-layer", "1", "--n-head", "2", "--n-embd", "32", "--seq-len", "16",
         "--batch-size", "2", "--grad-accum", "2", "--warmup", "1", "--eval-every", "3", *extra],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    assert run.returncode == 0, run.stderr[-2000:]


def test_resumed_training_ends_with_the_same_weights(tmp_path):
    tok = BPETokenizer.train([TEXT], vocab_size=300, special_tokens=["<|endoftext|>"])
    tok.save(tmp_path / "tok.json")
    ids = np.array(tok.encode(TEXT * 6), dtype=np.int64)
    write_tokens(tmp_path / "train.bin", ids)
    write_tokens(tmp_path / "val.bin", ids[:300])

    run_pretrain(tmp_path, tmp_path, tmp_path / "tok.json", tmp_path / "a.pt", "--max-steps", "6")
    run_pretrain(tmp_path, tmp_path, tmp_path / "tok.json", tmp_path / "b.pt", "--max-steps", "4", "--schedule-steps", "6")
    run_pretrain(tmp_path, tmp_path, tmp_path / "tok.json", tmp_path / "b.pt", "--max-steps", "6", "--schedule-steps", "6",
                 "--resume")

    a = torch.load(tmp_path / "a_final.pt", map_location="cpu")
    b = torch.load(tmp_path / "b_final.pt", map_location="cpu")
    assert a["step"] == b["step"] == 5
    for name, tensor in a["model"].items():
        torch.testing.assert_close(b["model"][name], tensor, rtol=1e-5, atol=1e-6, msg=name)


# T4 ------------------------------------------------------------------------------------------------
def test_learning_rate_warmup_cosine_and_floor():
    peak, floor = 1e-3, 1e-4
    assert lr_at(0, 100, 10, peak, floor) == pytest.approx(peak / 10)
    assert lr_at(9, 100, 10, peak, floor) == pytest.approx(peak)
    assert lr_at(100, 100, 10, peak, floor) == pytest.approx(floor)
    assert lr_at(150, 100, 10, peak, floor) == pytest.approx(floor)  # extended run: stays at the floor
    after_warmup = [lr_at(s, 100, 10, peak, floor) for s in range(10, 101)]
    assert all(x >= y for x, y in zip(after_warmup, after_warmup[1:]))


# T5 ------------------------------------------------------------------------------------------------
def test_chat_batches_pad_on_the_right_and_loss_is_weighted_by_trained_tokens(tmp_path):
    rng = np.random.default_rng(0)
    xs = [rng.integers(0, 40, n) for n in (7, 3, 5)]
    ys = []
    for x in xs:
        y = rng.integers(0, 40, len(x))
        y[: len(x) // 2] = IGNORE
        ys.append(y)
    offsets = np.cumsum([0] + [len(x) for x in xs])
    np.savez(tmp_path / "set.npz", x=np.concatenate(xs), y=np.concatenate(ys), offsets=offsets)
    data = sft.ChatSet(tmp_path / "set.npz")

    x, y = data.batch([1, 0], pad_id=39, device="cpu")
    assert x.shape == (2, 7)
    assert x[0, 3:].tolist() == [39] * 4 and y[0, 3:].tolist() == [IGNORE] * 4
    assert x[0, :3].tolist() == xs[1].tolist() and y[1].tolist() == ys[0].tolist()

    model = tiny_gpt(40)
    total, count = 0.0, 0
    with torch.no_grad():
        for xi, yi in zip(xs, ys):
            logits, _ = model(torch.tensor(xi)[None])
            t = torch.tensor(yi)
            total += F.cross_entropy(logits[0], t, ignore_index=IGNORE, reduction="sum").item()
            count += int((t != IGNORE).sum())
    assert sft.val_loss(model, data, batch_size=2, pad_id=39, device="cpu") == pytest.approx(total / count, rel=1e-5)


# T6 ------------------------------------------------------------------------------------------------
def test_rope_scores_depend_only_on_the_distance():
    torch.manual_seed(0)
    cos, sin = rope_tables(8, 32, 10000.0)
    q, k = torch.randn(8), torch.randn(8)
    rq = apply_rope(q.expand(1, 1, 32, 8), cos, sin)[0, 0]
    rk = apply_rope(k.expand(1, 1, 32, 8), cos, sin)[0, 0]
    assert torch.dot(rq[3], rk[7]).item() == pytest.approx(torch.dot(rq[13], rk[17]).item(), abs=1e-5)
    assert torch.dot(rq[3], rk[7]).item() != pytest.approx(torch.dot(rq[3], rk[9]).item(), abs=1e-3)


# T7 ------------------------------------------------------------------------------------------------
def test_overlap_check_reads_the_user_prompts_back(tmp_path):
    tok = chat_tokenizer()
    convos = [[("user", "What is water made of?"), ("assistant", "Hydrogen and oxygen.")],
              [("user", "Hi"), ("assistant", "Hello."), ("user", "Why is the sky blue?"), ("assistant", "Light.")]]
    xs = []
    for turns in convos:
        ids, train = encode_conversation(tok, turns)
        xs.append(to_example(ids, train, 512)[0])
    np.savez(tmp_path / "train.npz", x=np.concatenate(xs), offsets=np.cumsum([0] + [len(x) for x in xs]))
    prompts = check_eval_overlap.user_prompts(tok, tmp_path / "train.npz")
    assert prompts == ["What is water made of?", "Hi", "Why is the sky blue?"]
    assert check_eval_overlap.norm("Why is the sky blue?") == "why is the sky blue"


# T8 ------------------------------------------------------------------------------------------------
def msg(mid, parent, role, text, rank=None, lang="en", deleted=False, review=True):
    return {"message_id": mid, "parent_id": parent, "role": role, "text": text, "rank": rank, "lang": lang,
            "deleted": deleted, "review_result": review}


def test_oasst_follows_the_best_ranked_reply_and_skips_bad_messages(monkeypatch):
    rows = [
        msg("p1", None, "prompter", "Q1"),
        msg("a1", "p1", "assistant", "worse", rank=2),
        msg("a2", "p1", "assistant", "best", rank=1),
        msg("a3", "p1", "assistant", "failed review", rank=0, review=False),
        msg("p2", "a2", "prompter", "deleted question", deleted=True),
        msg("p3", "a2", "prompter", "Q2"),
        msg("a4", "p3", "assistant", "A2", rank=0),
        msg("p4", "a4", "prompter", "Q3"),
        msg("a5", "p4", "assistant", "A3", rank=0),
        msg("x1", None, "prompter", "Olá", lang="pt"),
        msg("x2", "x1", "assistant", "Oi", lang="pt", rank=0),
        msg("y1", None, "prompter", "no answer"),
    ]
    monkeypatch.setattr(prepare_sft, "load_dataset", lambda name, split, revision: rows if split == "train" else [])
    assert prepare_sft.oasst_conversations(max_turns=2) == [
        [("user", "Q1"), ("assistant", "best"), ("user", "Q2"), ("assistant", "A2")],
    ]


# T9 ------------------------------------------------------------------------------------------------
def test_reinit_changes_only_the_chat_rows_and_gives_them_an_ordinary_size():
    tok = chat_tokenizer()
    model = tiny_gpt(tok.vocab_size)
    before = model.tok_emb.weight.detach().clone()
    sft.reinit_chat_rows(model, tok, seed=0)
    after = model.tok_emb.weight.detach()
    changed = {i for i in range(len(before)) if not torch.equal(before[i], after[i])}
    chat_rows = {tok.special_tokens[n] for n in ("<|user|>", "<|assistant|>", "<|end|>")}
    assert changed == chat_rows
    ordinary = [i for i in range(len(before)) if i not in tok.special_tokens.values()]
    mean_norm = before[ordinary].norm(dim=1).mean().item()
    for i in chat_rows:
        assert 0.5 * mean_norm < after[i].norm().item() < 2 * mean_norm


# T10 -----------------------------------------------------------------------------------------------
def test_chat_history_drops_whole_pairs_and_keeps_the_last_message():
    tok = chat_tokenizer()
    turns = [("user", "water " * 30), ("assistant", "oxygen " * 30), ("user", "stars " * 30),
             ("assistant", "helium " * 30), ("user", "last question")]
    kept = chat.fit_history(tok, turns, seq_len=60, reserve=20)
    assert kept[-1] == ("user", "last question")
    assert kept[0][0] == "user" and len(kept) % 2 == 1
    assert len(kept) == 1 or len(prompt_ids(tok, kept)) <= 40
    assert kept == turns[len(turns) - len(kept):]


# data record for evaluate.py -----------------------------------------------------------------------
def test_evaluation_data_must_use_the_checkpoint_tokenizer(tmp_path):
    a, b = chat_tokenizer(), BPETokenizer.train([TEXT.upper()], vocab_size=300, special_tokens=SPECIALS)
    a.save(tmp_path / "a.json")
    b.save(tmp_path / "b.json")
    state = {"tokenizer": tokenizer_record(tmp_path / "a.json")}
    assert check_data_tokenizer(state, tmp_path) is False  # no record next to the data: cannot check
    write_data_record(tmp_path, tmp_path / "a.json")
    assert check_data_tokenizer(state, tmp_path) is True
    write_data_record(tmp_path, tmp_path / "b.json")
    with pytest.raises(ValueError, match="tokenized with"):
        check_data_tokenizer(state, tmp_path)


# next_word.py and chat.py --seed -------------------------------------------------------------------
def test_next_word_chances_follow_softmax_temperature_and_the_chain_rule():
    tok = chat_tokenizer()
    model = tiny_gpt(tok.vocab_size)
    ids = tok.encode("Water is made of hydrogen and")
    scores = next_word.next_token_scores(model, ids)
    assert scores.shape == (tok.vocab_size,)

    oxygen = next_word.word_ids(tok, "oxygen")
    assert oxygen == tok.encode(" oxygen") and next_word.word_ids(tok, " oxygen") == oxygen
    assert next_word.word_ids(tok, "<|end|>") == [tok.special_tokens["<|end|>"]]

    first = oxygen[0]
    for t in (1.0, 0.5):
        expected = F.softmax(scores / t, dim=-1)[first].item()
        assert next_word.word_chance(model, ids, [first], temperature=t) == pytest.approx(expected)

    two = tok.encode(" oxygen.")
    assert len(two) > 1
    by_hand = 1.0
    for i, t in enumerate(two):
        by_hand *= F.softmax(next_word.next_token_scores(model, ids + two[:i]), dim=-1)[t].item()
    assert next_word.word_chance(model, ids, two) == pytest.approx(by_hand)


def test_next_word_reads_special_tokens_in_the_prompt_and_keeps_the_last_seq_len_tokens():
    tok = chat_tokenizer()
    model = tiny_gpt(tok.vocab_size, seq_len=8)
    ids = tok.encode("<|user|>What is water made of?<|end|><|assistant|>")
    assert tok.special_tokens["<|assistant|>"] in ids
    long = tok.encode(TEXT)[:40]
    assert torch.equal(next_word.next_token_scores(model, long), next_word.next_token_scores(model, long[-8:]))


def test_percent_keeps_two_significant_digits_for_small_chances():
    assert next_word.percent(0.529) == "52.9%"
    assert next_word.percent(0.0084) == "0.84%"
    assert next_word.percent(1.1e-6) == "0.00011%"
    assert next_word.percent(3.2e-10) == "0.000000032%"
    assert next_word.percent(0.0) == "0%"


def test_seeded_chat_replies_repeat_and_unseeded_ones_vary():
    tok = chat_tokenizer()
    model = tiny_gpt(tok.vocab_size, seq_len=64)
    turns = [("user", "What is water made of?")]
    kw = dict(device="cpu", max_new=12, temperature=1.0, top_p=0.95, repetition_penalty=1.1)
    assert chat.reply(model, tok, turns, seed=1337, **kw) == chat.reply(model, tok, turns, seed=1337, **kw)
    torch.manual_seed(0)
    unseeded = {chat.reply(model, tok, turns, **kw) for _ in range(5)}
    assert len(unseeded) > 1


def test_weights_download_checks_the_hash_and_never_overwrites_another_file(tmp_path):
    import hashlib
    src = tmp_path / "release" / "model.pt"
    src.parent.mkdir()
    src.write_bytes(b"weights")
    url, digest = src.as_uri(), hashlib.sha256(b"weights").hexdigest()
    dest = tmp_path / "checkpoints" / "model.pt"

    assert download_weights.fetch(url, dest, digest) == "downloaded"
    assert dest.read_bytes() == b"weights"
    assert download_weights.fetch(url, dest, digest) == "already there"

    dest.write_bytes(b"my own model")
    with pytest.raises(SystemExit, match="not the published file"):
        download_weights.fetch(url, dest, digest)
    assert dest.read_bytes() == b"my own model"

    other = tmp_path / "checkpoints" / "other.pt"
    with pytest.raises(SystemExit, match="did not match"):
        download_weights.fetch(url, other, "0" * 64)
    assert not other.exists() and not other.with_name("other.pt.part").exists()


# eval_qa.py and pocketchat/qa.py ---------------------------------------------------------------------
def test_answers_are_scored_as_in_squad():
    refs = ["Denver Broncos", "the Denver Broncos"]
    assert qa.normalize("The  Denver Broncos!") == "denver broncos"
    assert qa.exact("denver broncos.", refs) == 1.0 and qa.exact("Broncos", refs) == 0.0
    assert qa.f1("Broncos", refs) == pytest.approx(2 * 1 * 0.5 / 1.5)
    assert qa.f1("Carolina Panthers", refs) == 0.0
    assert qa.contains("It was the Denver Broncos, in 2016.", refs) == 1.0
    assert qa.contains("It was completed in 1889.", ["1"]) == 0.0  # whole words only
    assert qa.contains("", refs) == 0.0 and qa.contains("anything", [""]) == 0.0


def test_qa_prompts_put_the_passage_before_the_question():
    tok = chat_tokenizer()
    chat_ids = qa.encode(tok, "What is water made of?", "Water is made of hydrogen and oxygen.", "chat")
    assert chat_ids[0] == tok.special_tokens["<|user|>"] and chat_ids[-1] == tok.special_tokens["<|assistant|>"]
    text = tok.decode(chat_ids)
    assert text.index("hydrogen and oxygen") < text.index("What is water made of?")
    assert qa.plain_text("Q?", None) == "Question: Q?\nAnswer:"
    assert qa.plain_text("Q?", "C.") == "C.\n\nQuestion: Q?\nAnswer:"
    with pytest.raises(ValueError):
        qa.encode(tok, "Q?", None, "other")


def test_qa_questions_are_fixed_one_per_paragraph_and_fit_the_context():
    tok = chat_tokenizer()
    rows = [{"id": str(i), "question": f"What is water made of? {i}", "context": TEXT[: 40 + (i % 5) * 600],
             "answers": {"text": ["oxygen"]}} for i in range(40)]
    budget = 120
    first = eval_qa.select(rows, tok, n=10, seed=3, budget=budget)
    assert first == eval_qa.select(rows, tok, n=10, seed=3, budget=budget)
    assert first != eval_qa.select(rows, tok, n=10, seed=4, budget=budget)
    assert len({r["context"] for r in first}) == len(first)
    for r in first:
        for fmt in ("chat", "plain"):
            assert len(qa.encode(tok, r["question"], r["context"], fmt)) <= budget


def test_plain_answers_stop_at_the_end_of_the_line():
    tok = chat_tokenizer()
    breaks = eval_qa.line_break_ids(tok)
    assert breaks and all("\n" in tok.decode([i]) for i in breaks)
    assert not breaks & set(tok.special_tokens.values())


def test_overlap_check_finds_a_shared_run_even_across_chunk_borders():
    import check_qa_overlap as ov
    rng = np.random.default_rng(0)
    corpus = rng.integers(0, 8000, 1000).astype(np.uint16)
    passage = corpus[495:510].copy()  # 15 tokens straddling the border of 500-token chunks
    other = rng.integers(0, 8000, 15).astype(np.uint16)
    q_seen, q_new = ov.ngram_hashes(passage, 13), ov.ngram_hashes(other, 13)
    assert len(q_seen) == 3 and len(ov.ngram_hashes(passage[:12], 13)) == 0
    hits = ov.found(corpus, np.concatenate([q_seen, q_new]), 13, chunk=500)
    assert set(hits) == set(q_seen)


def test_qa_report_splits_scores_by_pretraining_overlap():
    import report_qa
    answers = [
        {"id": "a", "mode": "open", "exact": 1.0, "f1": 1.0, "contains": 1.0},
        {"id": "b", "mode": "open", "exact": 0.0, "f1": 0.5, "contains": 1.0},
        {"id": "c", "mode": "open", "exact": 0.0, "f1": 0.0, "contains": 0.0},
        {"id": "a", "mode": "closed", "exact": 0.0, "f1": 0.0, "contains": 0.0},
    ]
    shares = {"a": {"pretraining": 0.0}, "b": {"pretraining": 0.6}, "c": {"pretraining": 1.0}}
    unseen = report_qa.group_scores(answers, shares, "open", 0.0, 0.1)
    assert unseen == {"n": 1, "exact": 1.0, "f1": 1.0, "contains": 1.0}
    copies = report_qa.group_scores(answers, shares, "open", 0.5, 1.01)
    assert copies["n"] == 2 and copies["f1"] == pytest.approx(0.25) and copies["contains"] == pytest.approx(0.5)
    assert report_qa.group_scores(answers, shares, "closed", 0.5, 1.01) == {"n": 0}
