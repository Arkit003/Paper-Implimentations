# Llama 2 inference implementation

A PyTorch implementation of the Llama 2 decoder-only transformer. It loads Hugging Face-style checkpoint weights and generates text autoregressively: the model predicts one next token at a time, appends it to the prompt, and repeats until generation ends.

## What’s included

### `model.py`: transformer architecture

The model is made of repeated decoder layers. Each layer combines self-attention and a feed-forward network, with normalization and residual connections around both sub-blocks. The concepts below explain the main techniques used in those layers and in generation.

### Key concepts

#### RMSNorm (Root Mean Square Layer Normalization)

Neural network activations can grow or shrink as they pass through many layers. RMSNorm stabilizes their scale by dividing each token’s feature vector by its root-mean-square value, then multiplying by learned per-feature weights:

```text
RMSNorm(x) = x / sqrt(mean(x²) + epsilon) * learned_weight
```

Unlike standard LayerNorm, RMSNorm does not subtract the feature mean. This makes normalization simpler while still controlling activation magnitude. `epsilon` prevents division by zero; the learned weights let the network adjust the scale it needs.

#### RoPE (Rotary Positional Embeddings)

Self-attention compares queries and keys to decide which earlier tokens are relevant. RoPE injects position into that comparison by rotating pairs of query and key features by angles that depend on token position and feature dimension. Because the rotation differs by position, attention scores reflect relative token positions as well as content. RoPE is applied to queries and keys, not values, and does not add a separate position vector to the token embedding.

#### GQA (Grouped-Query Attention)

Ordinary multi-head attention gives each query head its own key and value heads. GQA has many query heads but fewer key/value heads: groups of query heads share the same key/value head. For example, 32 query heads can share 8 key/value heads in groups of four. The model still computes attention for every query head, but stores fewer keys and values in its inference cache. This reduces cache memory and memory traffic, which is especially useful when generating long sequences or serving multiple requests.

#### SwiGLU

SwiGLU is the gated feed-forward function used inside each decoder layer. The input is projected along two paths: one path is passed through SiLU (also called Swish), and its result gates the other path through element-wise multiplication. A final projection maps the result back to the model width:

```text
SwiGLU(x) = (SiLU(x W_gate) * (x W_value)) W_output
```

The gate lets the network control which features flow through the feed-forward block, giving it a more flexible transformation than a plain two-layer feed-forward network.

#### Temperature

Temperature controls how sharply next-token probabilities favor high-scoring tokens. The model first produces logits (unnormalized scores); generation divides them by temperature before applying softmax. A lower positive temperature makes the distribution more concentrated and outputs more predictable. A higher temperature flattens it and makes less likely tokens more plausible. In this implementation, `temperature=0` is handled as greedy decoding: choose the token with the highest logit, without sampling.

#### Top-p (nucleus) sampling

Top-p sampling limits random choice to a probability “nucleus.” After converting scores to probabilities, tokens are ordered from most to least probable. The sampler keeps the smallest leading group whose cumulative probability reaches the `top_p` threshold, removes the rest, renormalizes the retained probabilities, and samples one token. For example, `top_p=0.9` usually keeps a variable number of candidates containing 90% of the probability mass. Unlike top-k, which always keeps a fixed number of tokens, top-p adapts the candidate set to how confident or uncertain the model is. Temperature shapes the probabilities first; top-p then truncates that distribution.

### `inference.py`: loading and text generation

`Llama.build(...)` reads the checkpoint configuration and tokenizer, constructs the model, and loads checkpoint tensors. A conversion helper renames Hugging Face parameter keys to match this implementation’s module names. SentencePiece converts input text into token IDs (adding a beginning-of-sequence token) and decodes generated IDs back into text.

During generation, the model caches keys and values from previous tokens so it does not need to recompute them at every step. It predicts next-token logits, selects a token using greedy decoding or temperature plus top-p sampling, and continues until the generation limit or an end-of-sequence token. The method returns both token IDs and decoded text.

### `models/llama-2-7b-hf/`: model assets

This directory contains the local Llama 2 7B checkpoint shards, model configuration, SentencePiece tokenizer, and the accompanying license and responsible-use documents. The inference loader expects `config.json`, `tokenizer.model`, and one or more `model-*.safetensors` files here.

The inference loader expects `config.json`, `tokenizer.model`, and one or more `model-*.safetensors` files in the checkpoint directory.

## Requirements

- Python 3.13 or later
- PyTorch and the packages declared in `pyproject.toml` (`safetensors`, `sentencepiece`, `numpy`, and `tqdm`)
- Enough RAM/VRAM for the checkpoint and inference cache. The 7B checkpoint files in this directory total roughly 13.5 GB on disk; runtime memory requirements vary with device, dtype, batch size, and sequence length.

## Setup

From this directory, install the project dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Alternatively, create an environment with Python 3.13+ and install the dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Run generation

The example at the bottom of `inference.py` loads the local 7B checkpoint and runs a few sample prompts. With the environment activated, run:

```bash
python inference.py
```

The example uses the relative checkpoint path `models/llama-2-7b-hf/` and defaults to CPU (`allow_cuda = False`). To attempt CUDA inference, edit `allow_cuda` to `True` in the example; CUDA is selected only when available. The `Llama.build(...)` method accepts a checkpoint directory, tokenizer path, sequence and batch limits, device, and a `load_model` flag. `text_completion(...)` accepts a list of prompts plus `temperature`, `top_p`, and `max_gen_len` options; setting temperature to `0` selects greedy decoding.

Example API usage:

```python
from inference import Llama
import torch

prompts = ["Once upon a time"]
device = "cuda" if torch.cuda.is_available() else "cpu"

llama = Llama.build(
    checkpoints_dir="models/llama-2-7b-hf",
    tokenizer_path="models/llama-2-7b-hf/tokenizer.model",
    load_model=True,
    max_seq_len=1024,
    max_batch_size=len(prompts),
    device=device,
)

_tokens, completions = llama.text_completion(
    prompts,
    temperature=0.6,
    top_p=0.9,
    max_gen_len=64,
)
print(completions[0])
```

## Notes and limitations

- The implementation currently processes one token per forward pass during generation and maintains a key/value cache sized from `max_batch_size` and `max_seq_len`.
- This is an inference-oriented implementation; it does not include training, evaluation, batching infrastructure, or a command-line argument parser.
- Checkpoint files are large. Do not download another copy if the local files are already available. If you obtain weights separately, use a compatible Hugging Face-format Llama checkpoint and ensure you have permission to use it.
- Respect the Llama 2 license and responsible-use requirements included with the checkpoint. Model outputs can be inaccurate or unsafe; review them before use.

## References

- Touvron et al., [*Llama 2: Open Foundation and Fine-Tuned Chat Models*](https://arxiv.org/abs/2307.09288), 2023.
- Meta, [Llama 2](https://ai.meta.com/llama/).
