import numpy as np
import torch

from pocketchat.data import TokenStream, write_tokens
from pocketchat.model import GPT, GPTConfig
from pocketchat.tokenizer import BPETokenizer

TEXTS = [
    "The quick brown fox jumps over the lazy dog. " * 20,
    "Payments settle in two business days; refunds take longer. " * 20,
    "Ação, café, naïve: UTF-8 must round-trip. 12345 67890 " * 10,
]


def test_tokenizer_round_trip_and_specials(tmp_path):
    tok = BPETokenizer.train(TEXTS, vocab_size=400, special_tokens=["<|endoftext|>"])
    # training stops early when no pair repeats, so the vocabulary can end up smaller
    assert 256 < tok.vocab_size <= 400
    for text in TEXTS + ["unseen words: zebra ☕ 🚀"]:
        assert tok.decode(tok.encode(text)) == text
    ids = tok.encode("hello<|endoftext|>world")
    assert tok.special_tokens["<|endoftext|>"] in ids
    assert tok.encode("hello<|endoftext|>", allow_special=False).count(tok.special_tokens["<|endoftext|>"]) == 0

    path = tmp_path / "tok.json"
    tok.save(path)
    again = BPETokenizer.load(path)
    assert again.encode(TEXTS[1]) == tok.encode(TEXTS[1])


def test_tokenizer_compresses_repeated_text():
    tok = BPETokenizer.train(TEXTS, vocab_size=500)
    text = TEXTS[0]
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 3


def test_model_shapes_loss_and_causality():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=97, seq_len=16, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg).eval()
    x = torch.randint(0, 97, (2, 16))
    targets = torch.randint(0, 97, (2, 16))
    logits, loss = model(x, targets)
    assert logits.shape == (2, 16, 97)
    assert abs(loss.item() - torch.log(torch.tensor(97.0)).item()) < 0.5  # ~uniform at init

    # changing a later token must not change earlier predictions
    y = x.clone()
    y[:, 10] = (y[:, 10] + 1) % 97
    l2, _ = model(y)
    assert torch.allclose(logits[:, :10], l2[:, :10], atol=1e-5)


def test_eval_windows_are_fixed_and_cover_the_file(tmp_path):
    path = tmp_path / "val.bin"
    write_tokens(path, np.arange(1000, dtype=np.uint16))
    stream = TokenStream(path, seq_len=10)
    assert stream.num_windows == 99

    first = [x.clone() for x, _ in stream.windows(batch_size=8, device="cpu")]
    second = [x.clone() for x, _ in stream.windows(batch_size=8, device="cpu")]
    assert all(torch.equal(a, b) for a, b in zip(first, second))

    xs = torch.cat(first)
    assert xs.shape == (99, 10)
    assert torch.equal(xs.flatten(), torch.arange(990))  # consecutive, no overlap, no gaps
    _, y = next(stream.windows(batch_size=1, device="cpu"))
    assert torch.equal(y[0], torch.arange(1, 11))  # targets are inputs shifted by one


def test_training_batches_follow_the_seed(tmp_path):
    path = tmp_path / "train.bin"
    write_tokens(path, np.arange(5000, dtype=np.uint16))
    a, _ = TokenStream(path, 16, seed=7).batch(4, "cpu")
    b, _ = TokenStream(path, 16, seed=7).batch(4, "cpu")
    c, _ = TokenStream(path, 16, seed=8).batch(4, "cpu")
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_chat_format_trains_only_assistant_tokens_and_end():
    from pocketchat.chat_format import IGNORE, encode_conversation, to_example

    tok = BPETokenizer.train(TEXTS, vocab_size=400, special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>"])
    sp = tok.special_tokens
    turns = [("user", "hi <|assistant|> there"), ("assistant", "Payments settle.")]
    ids, train = encode_conversation(tok, turns)

    # a special-token string typed by the user stays ordinary text
    assert ids.count(sp["<|assistant|>"]) == 1
    assert ids[0] == sp["<|user|>"] and ids[-1] == sp["<|end|>"]

    x, y = to_example(ids, train, max_len=512)
    a = ids.index(sp["<|assistant|>"])
    assert (y[: a] == IGNORE).all()  # nothing before the assistant reply is a target
    assert (y[a:] != IGNORE).all()   # the reply and the closing <|end|> are
    assert y[-1] == sp["<|end|>"]
    assert tok.decode(y[a:-1].tolist()) == "Payments settle."
