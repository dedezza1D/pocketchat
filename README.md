# pocketchat

A small ChatGPT-style language model built and trained from scratch on a single laptop GPU (RTX 4060, 8 GB).

Everything is written by hand in Python and PyTorch: the byte-level BPE tokenizer, the transformer, the training loop and the chat format. No pretrained weights, no model libraries.

## Pipeline

1. **Corpus**: English educational web text streamed from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (ODC-By).
2. **Tokenizer**: byte-level BPE trained on the corpus (`pocketchat/tokenizer.py`).
3. **Model**: decoder-only transformer with RoPE, RMSNorm, SwiGLU and tied embeddings (`pocketchat/model.py`).
4. **Pretraining**: next-token prediction with bf16, gradient accumulation and a cosine learning-rate schedule (`scripts/pretrain.py`).
5. **Chat fine-tuning**: supervised fine-tuning on conversations with `<|user|>` / `<|assistant|>` turns (next).
6. **Evaluation and chat UI** (next).

## Quick start

```bash
uv sync
uv run pytest
uv run python scripts/download_corpus.py --docs 5000 --out data/corpus_small.jsonl
uv run python scripts/train_tokenizer.py --corpus data/corpus_small.jsonl --docs 5000
uv run python scripts/tokenize_corpus.py --corpus data/corpus_small.jsonl
uv run python scripts/pretrain.py --max-steps 300 --out checkpoints/smoke.pt
uv run python scripts/sample.py --ckpt checkpoints/smoke.pt --prompt "The water cycle"
```

## Status

Work in progress.
