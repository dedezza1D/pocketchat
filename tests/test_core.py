import torch

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
