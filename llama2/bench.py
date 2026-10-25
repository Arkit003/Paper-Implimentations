"""
Benchmarks for the Llama-2 inference implementation.

Measures the numbers that actually describe inference:
  * TTFT   - time to first token (the prompt / prefill cost)
  * TPOT   - time per output token (the decode cost)
  * tok/s  - decode throughput
  * KV cache size

Every bench_* function takes:

    verbose       : bool  -> print a formatted table to the terminal
    save_to_file  : bool  -> write a JSON record into benchmark_results/
    output_dir    : str   -> where those JSON files go
    tag           : str   -> optional label folded into the filename

and returns a list of plain dicts, so results stay usable programmatically:

    rows = bench_ttft(model)
    worst = max(rows, key=lambda r: r["ttft_ms"])

Run everything:

    python bench.py --tiny            # fast smoke test, no checkpoint needed
    python bench.py                   # the real 7B checkpoint
    python bench.py --benches ttft decode_throughput

Each function is also runnable on its own; see the __main__ block at the bottom.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch

# Allow `python bench.py` and `python bench.py` from outside this directory:
# the sibling modules are the implementation under test.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import Llama
from model import ModelArgs, Transformer

# Anchor outputs next to this file, not the current working directory, so
# `python /path/to/bench.py` always writes into <llama2>/benchmark_results/.
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(_SELF_DIR, "benchmark_results")
DEFAULT_CHECKPOINT = os.path.join(_SELF_DIR, "models", "llama-2-7b-hf")



# infrastructure

def resolve_device(allow_cuda: bool = True, requested: str = "auto") -> str:
    """Pick the device. `requested` is 'auto', 'cpu' or 'cuda'.

    An explicit request is honoured even when it looks unavailable, so a
    benchmark run never silently switches device between iterations -- which
    would make the numbers meaningless. It warns instead.
    """
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: --device cuda requested but CUDA is unavailable; "
                  "falling back to cpu")
            return "cpu"
        return "cuda"
    return "cuda" if (allow_cuda and torch.cuda.is_available()) else "cpu"


def _sync(device: str) -> None:
    """Make sure queued work has finished before stopping the clock.

    Without this a CUDA benchmark measures how fast the CPU can enqueue
    kernels, not how fast the GPU executes them.
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cache(model: Llama) -> None:
    """Zero the KV cache so each timed run starts from a clean slate.

    The cache is allocated once in SelfAttention.__init__ and lives for the
    lifetime of the model, so without this every measurement after the first
    inherits stale keys/values from the previous run.
    """
    for layer in model.model.layers:
        att = layer.attention
        att.cache_k.zero_()
        att.cache_v.zero_()


def cache_bytes(model: Llama) -> int:
    """Total bytes held by all KV caches, K and V together."""
    total = 0
    for layer in model.model.layers:
        att = layer.attention
        total += att.cache_k.numel() * att.cache_k.element_size()
        total += att.cache_v.numel() * att.cache_v.element_size()
    return total


def _percentile(values: list[float], pct: float) -> float:
    """Simple percentile (nearest-rank); fine for the sample sizes here."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * len(ordered))) - 1))
    return ordered[idx]


def _summarise(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p90": _percentile(values, 90),
        "n": len(values),
    }


def _time(call: Callable[[], Any], device: str, repeat: int = 1) -> list[float]:
    """Run `call` `repeat` times, returning each wall-clock duration in ms."""
    out = []
    for _ in range(repeat):
        _sync(device)
        t0 = time.perf_counter()
        call()
        _sync(device)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out



# model construction


def build_model(
    checkpoints_dir: str = DEFAULT_CHECKPOINT,
    device: Optional[str] = None,
    max_seq_len: int = 1024,
    max_batch_size: int = 4,
    dtype: str = "bfloat16",
    tiny: bool = False,
    device_request: str = "auto",
) -> Llama:
    """Load the checkpoint
    """
    device = device or resolve_device(requested=device_request)
    ckpt = Path(checkpoints_dir)
    tok_path = str(ckpt / "tokenizer.model")

    if tiny:
        from sentencepiece import SentencePieceProcessor

        sp = SentencePieceProcessor()
        sp.load(tok_path)
        args = ModelArgs(
            dim=128,
            n_layers=4,
            n_heads=8,
            n_kv_heads=8,
            vocab_size=sp.vocab_size(),
            intermediate_size=256,
            norm_eps=1e-5,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
        )
        torch.manual_seed(0)
        # .to(device) matters: text_completion allocates its token/pad tensors on
        # args.device, so a model left on CPU raises a device mismatch on CUDA.
        model = Transformer(args).eval().to(device)
        # keep a real tokenizer so prompt lengths are realistic
        return Llama(model, sp, args)

    return Llama.build(
        checkpoints_dir=checkpoints_dir,
        tokenizer_path=tok_path,
        load_model=True,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
    )


def make_prompts(tokenizer, target_len: int, batch_size: int) -> list[str]:
    """`batch_size` prompts whose token length is close to `target_len`.

    Built by repetition from a varied seed so the tokenizer produces roughly
    target_len tokens; exact length does not matter, only that it is realistic.
    """
    seed = (
        "The history of science is a history of careful measurement, of asking "
        "questions that can be answered by observation, and of discarding the "
        "answers that turn out to be wrong. "
    )
    base_len = len(tokenizer.tokenize(seed, out_type=int, add_bos=True))
    repeats = max(1, target_len // max(1, base_len))
    prompt = seed * repeats
    return [prompt + f" Case {i}." for i in range(batch_size)]


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.tokenize(text, out_type=int, add_eos=False, add_bos=True))


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def _print_table(rows: list[dict], columns: Optional[list[str]] = None) -> None:
    if not rows:
        print("  (no results)")
        return
    columns = columns or list(rows[0].keys())
    widths = {
        c: max(len(str(c)), max(len(_fmt(r.get(c))) for r in rows)) for c in columns
    }
    header = "  ".join(str(c).rjust(widths[c]) for c in columns)
    print("  " + header)
    print("  " + "-" * len(header))
    for r in rows:
        print("  " + "  ".join(_fmt(r.get(c)).rjust(widths[c]) for c in columns))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.2f}"
    return str(value)


def _emit(
    name: str,
    rows: list[dict],
    meta: dict,
    verbose: bool,
    save_to_file: bool,
    output_dir: str,
    tag: str,
    columns: Optional[list[str]] = None,
) -> list[dict]:
    """Print and/or persist one benchmark's results. Returns rows unchanged."""
    if verbose:
        print(f"\n=== {name} ===")
        if meta:
            print("  " + ", ".join(f"{k}={v}" for k, v in meta.items()))
        _print_table(rows, columns)

    if save_to_file:
        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = f"_{tag}" if tag else ""
        path = Path(output_dir) / f"{name}{suffix}_{stamp}.json"
        record = {
            "benchmark": name,
            "tag": tag,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "host": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cpu_count": os.cpu_count(),
                "torch_threads": torch.get_num_threads(),
                "cuda_available": torch.cuda.is_available(),
                "device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
                ),
            },
            "meta": meta,
            "results": rows,
        }
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        if verbose:
            print(f"  -> saved {path}")

    return rows


def _model_meta(model: Llama) -> dict:
    a = model.args
    return {
        "dim": a.dim,
        "n_layers": a.n_layers,
        "n_heads": a.n_heads,
        "n_kv_heads": a.n_kv_heads,
        "max_seq_len": a.max_seq_len,
        "max_batch_size": a.max_batch_size,
        "device": a.device,
        "dtype": str(next(model.model.parameters()).dtype),
        "kv_cache_mb": round(cache_bytes(model) / 1024 ** 2, 2),
    }



# 1. TTFT


def bench_ttft(
    model: Llama,
    prompt_lengths: Iterable[int] = (32, 128, 512),
    batch_size: int = 1,
    repeats: int = 3,
    *,
    verbose: bool = False,
    save_to_file: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    tag: str = "",
) -> list[dict]:
    """Time to first token for short/medium/long prompts, prefill on and off.

    This is where prefill shows up. Without prefill the prompt is processed one
    token per forward pass; with it, one batched pass.
    """
    device = model.args.device
    rows: list[dict] = []

    for target in prompt_lengths:
        prompts = make_prompts(model.tokenizer, target, batch_size)
        actual = max(token_len(model.tokenizer, p) for p in prompts)
        for use_prefill in (True, False):
            # how many forward passes produce the first token
            passes = 1 if use_prefill else actual

            def run() -> None:
                reset_cache(model)
                with torch.no_grad():
                    model.text_completion(
                        prompts, temperature=0, max_gen_len=1, use_prefill=use_prefill
                    )

            samples = _time(run, device, repeat=repeats)
            rows.append(
                {
                    "target_prompt_len": target,
                    "prompt_tokens": actual,
                    "batch_size": batch_size,
                    "use_prefill": use_prefill,
                    "prompt_forward_passes": passes,
                    "ttft_ms_mean": statistics.fmean(samples),
                    "ttft_ms_min": min(samples),
                    "ttft_ms_p90": _percentile(samples, 90),
                    "samples_ms": [round(s, 2) for s in samples],
                }
            )

    meta = _model_meta(model)
    meta.update({"batch_size": batch_size, "repeats": repeats})
    return _emit("ttft", rows, meta, verbose, save_to_file, output_dir, tag)


# --------------------------------------------------------------------------- #
# 2. decode throughput: prefill on/off, batching on/off
# --------------------------------------------------------------------------- #

def bench_decode_throughput(
    model: Llama,
    prompt_lengths: Iterable[int] = (32, 256),
    batch_sizes: Iterable[int] = (1, 4),
    max_gen_len: int = 16,
    *,
    verbose: bool = False,
    save_to_file: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    tag: str = "",
) -> list[dict]:
    """Decode cost as a function of prefill, batching, and prompt length.

    total_s covers the whole call; decode_s/token covers only the generation
    loop, which is the part that grows with max_gen_len and is what you would
    quote as tok/s.
    """
    device = model.args.device
    rows: list[dict] = []

    def time_gen(prompts: list[str], gen_len: int, use_prefill: bool, repeats: int = 3) -> float:
        def run() -> None:
            reset_cache(model)
            with torch.no_grad():
                model.text_completion(
                    prompts, temperature=0, max_gen_len=gen_len, use_prefill=use_prefill
                )

        return statistics.fmean(_time(run, device, repeat=repeats))

    for target in prompt_lengths:
        for batch_size in batch_sizes:
            if batch_size > model.args.max_batch_size:
                continue
            prompts = make_prompts(model.tokenizer, target, batch_size)
            actual = max(token_len(model.tokenizer, p) for p in prompts)

            for use_prefill in (True, False):
                total_ms = time_gen(prompts, max_gen_len, use_prefill)
                # Marginal cost of decoding: run with fewer generated tokens and
                # difference the totals. Tokenization, prompt passes and other
                # fixed overhead cancel out, leaving pure per-token decode cost.
                short = max(1, max_gen_len // 2)
                base_ms = time_gen(prompts, short, use_prefill)

                # The difference isolates the marginal decode cost, but with
                # use_prefill=False the fixed term is itself many per-token
                # passes, so the two timings can invert on noise. Report NaN
                # ("n/a") rather than a fabricated number when that happens.
                delta = total_ms - base_ms
                if max_gen_len > short and delta > 0:
                    per_token = delta / (max_gen_len - short)
                else:
                    per_token = float("nan")

                rows.append(
                    {
                        "prompt_tokens": actual,
                        "batch_size": batch_size,
                        "use_prefill": use_prefill,
                        "gen_len": max_gen_len,
                        "total_ms": total_ms,
                        "decode_ms_per_token": per_token,
                        # end-to-end throughput: generated tokens / whole call
                        "tokens_per_s": (max_gen_len * batch_size) / (total_ms / 1000.0)
                        if total_ms > 0
                        else float("nan"),
                    }
                )

    meta = _model_meta(model)
    meta.update({"max_gen_len": max_gen_len})
    return _emit(
        "decode_throughput", rows, meta, verbose, save_to_file, output_dir, tag
    )



# 3. batching curve


def bench_batch_scaling(
    model: Llama,
    batch_sizes: Iterable[int] = (1, 2, 4),
    prompt_len: int = 128,
    decode_steps: int = 8,
    *,
    verbose: bool = False,
    save_to_file: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    tag: str = "",
) -> list[dict]:
    """Does throughput scale with batch size, or just latency?

    On CPU or a small GPU, decode is memory-bandwidth bound: the weight read is
    the same regardless of batch, so tokens/s should climb with batch size while
    per-step latency barely moves. That is the whole argument for batching.
    """
    device = model.args.device
    rows: list[dict] = []

    for batch_size in batch_sizes:
        if batch_size > model.args.max_batch_size:
            continue
        prompts = make_prompts(model.tokenizer, prompt_len, batch_size)
        actual = max(token_len(model.tokenizer, p) for p in prompts)

        reset_cache(model)
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.text_completion(prompts, temperature=0, max_gen_len=decode_steps,
                                  use_prefill=True)
        _sync(device)
        total_s = time.perf_counter() - t0

        # forward passes issued: 1 prefill + (decode_steps - 1) decode steps
        steps = max(1, decode_steps)
        rows.append(
            {
                "batch_size": batch_size,
                "prompt_tokens": actual,
                "total_s": total_s,
                "ms_per_step": (total_s * 1000.0) / steps,
                "tokens_per_s": (steps * batch_size) / total_s,
            }
        )

    meta = _model_meta(model)
    meta.update({"prompt_len": prompt_len, "decode_steps": decode_steps})
    return _emit("batch_scaling", rows, meta, verbose, save_to_file, output_dir, tag)



# 4. prefill vs per-token, phase breakdown


def bench_prefill_vs_decode(
    model: Llama,
    prompt_len: int = 256,
    *,
    verbose: bool = False,
    save_to_file: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    tag: str = "",
) -> list[dict]:
    """Cost of the prompt phase as one batched pass vs N single-token passes.

    Splits the prompt into two equal halves and times:
      (a) one pass over [0, n)
      (b) n passes over [0, i] for i in range(n)
    """
    device = model.args.device
    prompts = make_prompts(model.tokenizer, prompt_len, 1)
    n = token_len(model.tokenizer, prompts[0])

    ids = torch.tensor(
        [model.tokenizer.tokenize(prompts[0], out_type=int, add_eos=False, add_bos=True)],
        dtype=torch.long,
        device=device,
    )

    def batched() -> None:
        reset_cache(model)
        with torch.no_grad():
            model.model.forward(ids, 0, valid_len=n)

    def per_token() -> None:
        reset_cache(model)
        with torch.no_grad():
            for i in range(n):
                model.model.forward(ids[:, i:i + 1], i)

    prefill_ms = statistics.fmean(_time(batched, device, repeat=3))
    per_token_ms = statistics.fmean(_time(per_token, device, repeat=3))

    rows = [
        {
            "prompt_tokens": n,
            "mode": "batched_prefill",
            "forward_passes": 1,
            "total_ms": prefill_ms,
            "ms_per_prompt_token": prefill_ms / n,
        },
        {
            "prompt_tokens": n,
            "mode": "per_token",
            "forward_passes": n,
            "total_ms": per_token_ms,
            "ms_per_prompt_token": per_token_ms / n,
        },
        {
            "prompt_tokens": n,
            "mode": "speedup",
            "forward_passes": f"{n}x fewer",
            "total_ms": per_token_ms / prefill_ms if prefill_ms else float("nan"),
            "ms_per_prompt_token": float("nan"),
        },
    ]
    return _emit("prefill_vs_decode", rows, _model_meta(model),
                 verbose, save_to_file, output_dir, tag)



# 5. dtype


def bench_dtype(
    checkpoints_dir: str = DEFAULT_CHECKPOINT,
    dtypes: Iterable[str] = ("bfloat16", "float32"),
    tiny: bool = True,
    device_request: str = "auto",
    *,
    verbose: bool = False,
    save_to_file: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    tag: str = "",
) -> list[dict]:
    """Per-token decode cost per dtype.

    Kept on the tiny model by default: three copies of the 7B checkpoint will
    not fit in RAM on a 16 GB machine, and float32 definitely will not.
    """
    rows: list[dict] = []
    previous_dtype = torch.get_default_dtype()
    try:
        for dt in dtypes:
            torch.set_default_dtype(getattr(torch, dt))
            model = build_model(checkpoints_dir, dtype=dt, tiny=tiny, max_batch_size=1)
            device = model.args.device
            prompts = make_prompts(model.tokenizer, 64, 1)

            def run() -> None:
                reset_cache(model)
                with torch.no_grad():
                    model.text_completion(prompts, temperature=0, max_gen_len=8,
                                          use_prefill=True)

            samples = _time(run, device, repeat=3)
            mean_ms = statistics.fmean(samples)
            rows.append(
                {
                    "dtype": dt,
                    "params_mb": round(
                        sum(p.numel() * p.element_size()
                            for p in model.model.parameters()) / 1024 ** 2, 2
                    ),
                    "total_ms": mean_ms,
                    "ms_per_token": mean_ms / 8,
                }
            )
            del model
    finally:
        # set_default_dtype is process-global; leaving it changed would silently
        # alter every benchmark that builds a model after this one
        torch.set_default_dtype(previous_dtype)

    return _emit("dtype", rows, {"tiny": tiny}, verbose, save_to_file, output_dir, tag)



# 6. GQA vs MHA


def bench_kv_heads(
    checkpoints_dir: str = DEFAULT_CHECKPOINT,
    kv_head_counts: Iterable[int] = (1, 2, 4, 8),
    device_request: str = "auto",
    *,
    verbose: bool = False,
    save_to_file: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    tag: str = "",
) -> list[dict]:
    """Effect of the number of KV heads on decode speed and cache size.

    n_kv_heads == n_heads is MHA; fewer is GQA. Grouped-query attention shrinks
    the cache proportionally, which is why long-context serving uses it.
    """
    from sentencepiece import SentencePieceProcessor

    sp = SentencePieceProcessor()
    sp.load(str(Path(checkpoints_dir) / "tokenizer.model"))
    device = resolve_device(requested=device_request)
    rows: list[dict] = []
    n_heads = 8

    for n_kv in kv_head_counts:
        # GQA requires n_kv_heads <= n_heads, otherwise n_rep = n_heads // n_kv
        # is 0 and repeat_kv silently collapses the head dimension to zero.
        if n_kv > n_heads or n_heads % n_kv != 0:
            if verbose:
                print(f"  skipping n_kv_heads={n_kv}: must divide n_heads={n_heads}")
            continue
        args = ModelArgs(
            dim=128, n_layers=4, n_heads=8, n_kv_heads=n_kv,
            vocab_size=sp.vocab_size(), intermediate_size=256,
            max_seq_len=256, max_batch_size=1, device=device,
        )
        torch.manual_seed(0)
        model = Llama(Transformer(args).eval().to(device), sp, args)
        prompts = make_prompts(sp, 64, 1)

        def run() -> None:
            reset_cache(model)
            with torch.no_grad():
                model.text_completion(prompts, temperature=0, max_gen_len=8,
                                      use_prefill=True)

        samples = _time(run, device, repeat=3)
        mean_ms = statistics.fmean(samples)
        rows.append(
            {
                "n_heads": 8,
                "n_kv_heads": n_kv,
                "ratio": f"8:{n_kv}",
                "cache_kb": round(cache_bytes(model) / 1024, 1),
                "total_ms": mean_ms,
                "ms_per_token": mean_ms / 8,
            }
        )
        del model

    return _emit("kv_heads", rows, {"dim": 128, "n_layers": 4, "n_heads": 8},
                 verbose, save_to_file, output_dir, tag)



# registry + CLI


BENCHMARKS: dict[str, Callable[..., list[dict]]] = {
    "ttft": bench_ttft,
    "decode_throughput": bench_decode_throughput,
    "batch_scaling": bench_batch_scaling,
    "prefill_vs_decode": bench_prefill_vs_decode,
    "dtype": bench_dtype,
    "kv_heads": bench_kv_heads,
}


def run_all(
    checkpoints_dir: str = DEFAULT_CHECKPOINT,
    benches: Optional[Iterable[str]] = None,
    verbose: bool = True,
    save_to_file: bool = False,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    tag: str = "",
    tiny: bool = False,
    max_seq_len: int = 1024,
    max_batch_size: int = 4,
    device_request: str = "auto",
) -> dict[str, list[dict]]:
    """Run a set of benchmarks and return {name: rows}."""
    names = list(benches) if benches else list(BENCHMARKS)
    device = resolve_device(requested=device_request)
    if verbose:
        print(f"device: {device}")
    model = build_model(
        checkpoints_dir, device=device, max_seq_len=max_seq_len,
        max_batch_size=max_batch_size, tiny=tiny,
    )

    common = dict(verbose=verbose, save_to_file=save_to_file, output_dir=output_dir, tag=tag)
    out: dict[str, list[dict]] = {}
    for name in names:
        fn = BENCHMARKS[name]
        # the model-building benchmarks make their own models
        if name in ("dtype", "kv_heads"):
            out[name] = fn(checkpoints_dir=checkpoints_dir,
                           device_request=device_request, **common)
        else:
            out[name] = fn(model, **common)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Llama-2 inference benchmarks")
    p.add_argument("--model", default=DEFAULT_CHECKPOINT, help="checkpoint directory")
    p.add_argument("--benches", nargs="+", choices=list(BENCHMARKS), default=None,
                   help="which benchmarks to run (default: all)")
    p.add_argument("--tiny", action="store_true",
                   help="use a small random model instead of the 7B checkpoint")
    p.add_argument("--save", action="store_true", help="write JSON into --output-dir")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--tag", default="", help="label folded into output filenames")
    p.add_argument("--quiet", action="store_true", help="do not print tables")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--max-batch-size", type=int, default=4)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="explicit device; 'auto' prefers cuda when available")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    t_start = time.perf_counter()
    results = run_all(
        checkpoints_dir=args.model,
        benches=args.benches,
        verbose=not args.quiet,
        save_to_file=args.save,
        output_dir=args.output_dir,
        tag=args.tag,
        tiny=args.tiny,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        device_request=args.device,
    )
    elapsed = time.perf_counter() - t_start
    print(f"\ndone in {elapsed:.1f}s -- {sum(len(v) for v in results.values())} rows "
          f"across {len(results)} benchmarks")
