# tokenizer — Byte Pair Encoding from scratch

A minimal, dependency-free implementation of **byte-level Byte Pair Encoding (BPE)** — the same tokenization scheme that powers GPT-2, GPT-3, LLaMA and most modern LLMs — written in ~70 lines of readable Python, with a companion notebook that derives every piece of the algorithm from first principles.

This is a learning project: the goal is not a production tokenizer, but to be able to answer *"what exactly happens when a string becomes tokens?"* without hand-waving.

---

## Quick start

Requires Python 3.13+. No third-party packages are needed for the core algorithm.

```bash
# with uv (project uses a uv lockfile)
uv sync

# or just run it — the core needs only the standard library
python tokenizer.py
```

Training on your own corpus and round-tripping text:

```python
from tokenizer import BasicTokenizer

tok = BasicTokenizer()

# learn merges until the vocab reaches 500 tokens
# (256 raw byte values + 244 learned merges)
tok.train(open("train.txt", encoding="utf-8").read(), vocab_size=500)

ids = tok.encode("Hello, world!")   # → [some, token, ids]
txt = tok.decode(ids)               # → "Hello, world!" (always lossless)
```

`python tokenizer.py` runs a demo: it trains on `train.txt` (vocab size 500, printing every merge), then decodes a hard-coded token sequence into a passage of the Taylor Swift Wikipedia article.

---

## How the algorithm works

Byte-level BPE rests on one observation: **any text is just a sequence of bytes**, and byte sequences that co-occur a lot are probably meaningful units.

Training is a greedy compression loop:

1. Start with the corpus as a list of integers in `0..255` (the raw UTF-8 bytes).
2. Count every *adjacent pair* of tokens.
3. Replace every occurrence of the most frequent pair with a brand-new token id, starting at `256`.
4. Repeat until the vocabulary reaches the target size.

Encoding new text re-applies the learned merges in the order they were learned (earliest merge = highest priority). Decoding expands each id back through the merge table into bytes, then decodes the bytes as UTF-8. Since every text is representable as bytes and merges only ever group bytes together, **encode/decode is lossless by construction**.

### A worked example

On the string `"the cat in the hat"`, the first four training merges are:

```
merge (t, h) → 256      count = 2      # "th"
merge (256, e) → 257    count = 2      # "the"
merge (257, ' ') → 258  count = 2      # "the "
merge (a, t) → 259      count = 2      # "at"
```

After training, the encoded sequence is

```
[258, 99, 259, 32, 105, 110, 32, 258, 104, 259]
  │       │               │        │
"the "   "at"  ' ' i n ' '  "the"  h  "at"
```

— 19 bytes compressed into 10 tokens, and decoding reproduces the input exactly.

---

## Repository layout

| Path | What it is |
|---|---|
| `tokenizer.py` | `BasicTokenizer` — the complete algorithm in one class (`train` / `encode` / `decode`). |
| `impl.ipynb` | The derivation notebook: byte encoding → pair statistics → merges → decode, plus comparisons with `tiktoken`. |
| `train.txt` | Training corpus — a Wikipedia article dump (~75 KB). |
| `taylor_swift.txt` | held-out evaluation text — the Taylor Swift Wikipedia article as of Feb 16, 2024 (~186 KB), used to check that merges learned on `train.txt` transfer. |
| `pyproject.toml`, `uv.lock` | Project metadata; managed with `uv`. Python ≥ 3.13. |
| `requirements.txt` | Notebook-only extras (`ipykernel`, `torch`). |

## API

```python
tok = BasicTokenizer()
```

| Method | Signature | Notes |
|---|---|---|
| `train` | `train(text, vocab_size, verbose=False)` | Learns `vocab_size − 256` merges. `verbose=True` prints each merge. Returns the tokenized training corpus. |
| `encode` | `encode(text) -> list[int]` | Greedily applies learned merges, earliest first. |
| `decode` | `decode(ids) -> str` | Expands ids → bytes → UTF-8 text. Invalid UTF-8 (e.g. a truncated merge) is replaced, not raised. |

---

## What the notebook covers

`impl.ipynb` is the "show your work" companion:

- Why tokenizers operate on **bytes**, not characters — with tricky Unicode (fullwidth characters, flag emoji built from regional indicators + zero-width joiners, `‽`) that breaks naive character-level schemes.
- Building `get_stats` / `merge` incrementally and inspecting real pair frequencies.
- Measuring the **compression ratio** (≈1.27× on the sample at vocab size 276) — the practical reason tokenizers exist.
- Verifying round-trip fidelity on in-domain and out-of-domain text.
- A look at what real tokenizers add: the **GPT-2 regex pre-tokenization** pattern (which forbids merges across word/category boundaries) and side-by-side encodings from `tiktoken` (`gpt2` vs `cl100k_base`).

The natural next step beyond this repo is combining the two: BPE merges *within* the GPT-2 regex chunks. That combination is exactly GPT-2's tokenizer.

---

## Design notes & known limitations

Honest list, since this is a learning artifact:

- **`torch` is imported but unused** by `tokenizer.py` — the algorithm is pure Python. Safe to remove.
- **No pre-tokenization.** Unlike GPT-2, merges are free to cross whitespace and category boundaries, so learned tokens are corpus-specific.
- **`decode` rebuilds the vocabulary on every call.** Fine for a demo; hoist it into `train`/`__init__` for real use.
- **Training is O(vocab_size × corpus_length)** — stats are recounted from scratch each iteration. Real tokenizers use incremental pair counts and linked lists / priority queues.
- **No serialization** — merges live in memory; add `save()`/`load()`.
- **No special tokens** (BOS/EOS/pad, chat templates).
- `pyproject.toml` declares a `tokenizer = "tokenizer:main"` console script, but no `main()` exists yet.

## Ideas to extend

1. `RegexTokenizer`: split text with the GPT-2 pattern before BPE → tokens stop spanning word boundaries.
2. Save/load the merge table and compare compression on held-out text (that's what `taylor_swift.txt` is for).
3. Special tokens + a `GPT4Tokenizer`-style byte-level mapping.
4. Incremental training for speed, then benchmark against `tiktoken`'s BPE on the same corpus.

