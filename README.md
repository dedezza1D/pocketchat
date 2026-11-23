# pocketchat

[![tests](https://github.com/dedezza1D/pocketchat/actions/workflows/tests.yml/badge.svg)](https://github.com/dedezza1D/pocketchat/actions/workflows/tests.yml)

<p align="center">
  <img src="assets/pocketchat.png" alt="pocketchat — a mini ChatGPT built and trained from scratch on a laptop GPU: 29.9M parameters, 491M tokens, about 4 hours on an RTX 4060, answers that end from 0% to 80%" width="100%">
</p>

A small ChatGPT-style language model built and trained from scratch on a single laptop GPU (RTX 4060, 8 GB).

Everything is written by hand in Python and PyTorch: the byte-level BPE tokenizer, the transformer, the training loop and the chat format. No pretrained weights, no model libraries.

I explain how it works, step by step and with every number measured on this model, in the series **Inside a ChatGPT**: [on the web](https://dedezza1d.github.io/inside-a-chatgpt/) and as a [LinkedIn newsletter](https://www.linkedin.com/newsletters/inside-a-chatgpt-7528472821451268096/).

## Pipeline

1. **Corpus**: English educational web text streamed from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (ODC-By).
2. **Tokenizer**: byte-level BPE trained on the corpus (`pocketchat/tokenizer.py`).
3. **Model**: decoder-only transformer with RoPE, RMSNorm, SwiGLU and tied embeddings (`pocketchat/model.py`).
4. **Pretraining**: next-token prediction with bf16, gradient accumulation and a cosine learning-rate schedule (`scripts/pretrain.py`), evaluated on fixed validation windows (`scripts/evaluate.py`).
5. **Chat fine-tuning**: supervised fine-tuning on `<|user|>` / `<|assistant|>` conversations, with the loss on assistant tokens only (`scripts/prepare_sft.py`, `scripts/sft.py`).
6. **Evaluation and chat**: a fixed question set measuring whether answers end, their length and their repetition (`scripts/eval_chat.py`); a terminal chat (`scripts/chat.py`).

Every checkpoint records the tokenizer it was trained with (path and SHA-256), and loading refuses a tokenizer that does not match (`pocketchat/checkpoint.py`).

## Model and training

| | |
|---|---|
| Architecture | 8 layers, 8 heads, width 512, SwiGLU hidden 1,408, context 512, vocabulary 8,192, tied embeddings |
| Parameters | 29.9M |
| Pretraining | 15,000 steps × 32,768 tokens (491M token positions), AdamW (β 0.9, 0.95), weight decay 0.1, learning rate 1e-3 → 1e-4 (warmup + cosine), bf16, peak 3.2 GiB, about 4 h |
| Fine-tuning | 17,411 conversations, 3 passes (1,632 steps), learning rate 1e-4 → 1e-5, about 20 min |

<p align="center">
  <img src="assets/training_curves.svg" alt="Validation loss during pretraining, from 4.58 at step 500 to 3.24 at step 15,000, and assistant-token validation loss during fine-tuning for the two runs, from 2.96 and 3.10 to about 2.58" width="100%">
</p>

(`scripts/plot_curves.py`. During pretraining each point used a sampled subset of the validation windows; the final value uses all of them.)

## Results

| Stage | Result |
|---|---|
| Pretraining | validation loss 3.243 (perplexity 25.6) on all 4,997 fixed windows |
| Chat fine-tuning on [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) (Apache-2.0) and [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (CC BY-SA 3.0) | assistant-token validation loss 2.574 (lowest of 17 evaluations; 2.580 at the last step) |

A fixed set of 30 questions, one sample per question with the same seed and sampling (temperature 0.7, top-p 0.9, repetition penalty 1.1, at most 200 tokens):

| Model | Answers that end | Mean length | Repeated 3-grams |
|---|---|---|---|
| Pretrained only | 0% | 200 tokens (limit) | 66% |
| Fine-tuned, chat-token rows kept | 77% | 106 tokens | 9.3% |
| Fine-tuned, chat-token rows reinitialized | 80% | 99 tokens | 6.7% |

The pretrained model has never seen the chat tokens, so it cannot end an answer by construction; the 0% shows the format, not a weakness of the model. Full answers are in `evals/`.

Two answers from the fine-tuned model:

> **What is the capital of France?** Capital of France is the capital of France.
>
> **Why is the sky blue?** The sky blue is blue because of the color green.

Fine-tuning teaches the format (answer, then stop); it does not add knowledge. At this size the answers are fluent and on topic but often wrong.

## Findings

- **The "best" pretraining checkpoint was noise.** Validation during training used a different random sample of windows each time, and near the end consecutive values moved by a few hundredths. Measured on all fixed windows, the last step (3.2429) was better than the step marked as best (3.2514). Since then, validation uses fixed windows and fine-tuning starts from the last step (`pretrain_final.pt`).
- **The chat tokens collapsed during pretraining.** With tied embeddings, the rows of `<|user|>`, `<|assistant|>` and `<|end|>` were the wrong answer at every step and ended up almost identical (cosine 0.996) and with a larger norm than ordinary rows (1.83 vs 1.10). Reinitializing them lowers the starting fine-tuning loss (3.10 → 2.96); after fine-tuning the two runs are within noise (2.579 vs 2.574, one seed each, and one answer apart on the 30 questions).
- **Fine-tuning made the model forget.** Perplexity on the pretraining validation set rose from 25.6 to 30.1 after fine-tuning (`evals/forgetting.txt`), and the chance of " oxygen" after "Water is made of hydrogen and" fell from 52.9% to 28.8%.

## Limitations

- One seed per experiment and one sample per question: differences of a few points between the two fine-tuning runs are not meaningful.
- Four of the 30 evaluation questions also appear as prompts in the fine-tuning data ("Why is the sky blue?" 11 times; "What is the capital of France?", "What causes earthquakes?", "What is the Pythagorean theorem?" once or twice; `scripts/check_eval_overlap.py`, `evals/eval_overlap.txt`). None appears in the validation conversations. The model still answers them wrong.
- The tokenizer was trained on documents that also include the validation split.
- Training samples windows with replacement, so about a third of the corpus was never seen.
- Generation recomputes the whole context for every token (no KV cache).
- The same seed reproduces answers only on the same GPU and PyTorch version (bf16 sampling on CUDA).

## Pretrained weights

The pretrained model (step 15,000, before chat fine-tuning) and its tokenizer are in the [v0.1.0 release](https://github.com/dedezza1D/pocketchat/releases/tag/v0.1.0). To sample from it without training:

```bash
gh release download v0.1.0 --repo dedezza1D/pocketchat --pattern pretrain_final.pt --dir checkpoints
gh release download v0.1.0 --repo dedezza1D/pocketchat --pattern tokenizer_full.json --dir data
uv run python scripts/sample.py --prompt "Water is made of hydrogen and" --seed 0
```

The fine-tuned chat weights are not published: part of their training data, Dolly 15k, is CC BY-SA 3.0.

## Quick start

```bash
uv sync
uv run pytest
uv run python scripts/download_corpus.py --docs 5000 --out data/corpus_small.jsonl
uv run python scripts/train_tokenizer.py --corpus data/corpus_small.jsonl --docs 5000
uv run python scripts/tokenize_corpus.py --corpus data/corpus_small.jsonl
uv run python scripts/pretrain.py --max-steps 300 --out checkpoints/smoke.pt
uv run python scripts/sample.py --ckpt checkpoints/smoke_final.pt --prompt "The water cycle" --seed 0
```

Full run (download, tokenizer, tokenization and pretraining, resumable; writes `checkpoints/pretrain_final.pt`):

```bash
uv run python scripts/run_full.py
```

Chat fine-tuning and evaluation:

```bash
uv run python scripts/prepare_sft.py
uv run python scripts/sft.py --out checkpoints/sft_reinit.pt --reinit-chat-tokens
uv run python scripts/eval_chat.py --ckpt checkpoints/sft_reinit.pt --name sft_reinit
uv run python scripts/evaluate.py --ckpt checkpoints/pretrain_final.pt checkpoints/sft_reinit.pt
uv run python scripts/chat.py --ckpt checkpoints/sft_reinit.pt
```

`uv sync` installs the CUDA build of PyTorch; the CI runs the tests on the CPU build.

## License

The code is released under the [MIT License](LICENSE). The datasets keep their own licenses, listed below.

## Data licenses

FineWeb-Edu is released under ODC-By; the published pretrained weights were trained only on it (attribution: Hugging Face, FineWeb-Edu) and are released under the MIT License, like the code. oasst1 is Apache-2.0. databricks-dolly-15k is CC BY-SA 3.0 (attribution: Databricks). No outputs of commercial models were used for training.
