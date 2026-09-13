#!/usr/bin/env python3
"""
Per-layer forward microbenchmark for Mamba stacks loaded like ``gpu_hours.py``.

Times **one** ``backbone.layers[i]`` forward with a fixed ``(B, S, hidden_size)`` tensor,
``--iters`` times after warmup. **CUDA:** ``torch.cuda.Event`` timing (GPU kernel time on the
default stream). **CPU:** ``time.perf_counter`` wall time (no CUDA events).

This avoids full-model effects (pipeline overlap, inter-layer scheduling) relative to a full LM
forward, especially on GPU.

Variants: **mamba**, **steered**, **mamba2** (parallel gates/SSMs may overlap on GPU), **mamba2_straight**
(same Mamba2 math; after each gate and SSM uses ``_cuda_hard_serial_barrier``: ``torch.cuda.synchronize()``
plus a scalar ``.item()`` on that tensor). To measure time taken by each distinct stage run
sequentially using **those CUDA hard barriers** (host waits until each gate/SSM output is truly
done before enqueueing the next) captures **serial** GPU time that overlapping launches of the
parallel SSM and parallel gates do not.

Examples::

    python flops.py --device cuda --layer 0 --seq-len 512 --batch-size 1 --iters 1000
    python flops.py --device cpu --layer 0 --seq-len 256 --batch-size 1 --iters 1000


Results - on CPU
variant                  avg_ms     vs_first       tokens/s
------------------------------------------------------------
mamba                999.961128        1.000x          256.0
steered              931.002381        0.931x          275.0
mamba2              1797.127602        1.797x          142.4
mamba2_straight     2195.631208        2.196x          116.6
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from typing import List, Tuple

import torch

# Same directory as gpu_hours.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import gpu_hours as gh


def _resolve_layer(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "layers"):
        raise RuntimeError("Expected model.backbone.layers (MambaForCausalLM-style).")
    layers = model.backbone.layers
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"layer_idx {layer_idx} out of range [0, {len(layers)})")
    return layers[layer_idx]


def _bench_layer_forward(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    iters: int,
    warmup: int,
    use_cuda_events: bool,
) -> Tuple[float, float]:
    """
    Returns (elapsed_ms_total, avg_ms_per_iter) for ``iters`` calls to ``layer(x)``.
    If ``use_cuda_events``, uses CUDA events on the default stream; else wall clock (CPU).
    """
    layer.eval()
    with torch.no_grad():
        for _ in range(warmup):
            layer(x)
    if use_cuda_events:
        torch.cuda.synchronize()

    if use_cuda_events:
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(iters):
            layer(x)
        end_ev.record()
        torch.cuda.synchronize()
        total_ms = float(start_ev.elapsed_time(end_ev))
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            layer(x)
        total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms, total_ms / float(iters)


def _make_input(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)


def _load_variant(
    name: str,
    *,
    model_name: str,
    device: str,
    mamba_repo: str,
    steered_repo: str,
    mamba2_repo: str,
    steering_layers: List[int],
    steering_strength: float,
    mamba2_base_only: bool,
) -> torch.nn.Module:
    if name == "mamba":
        m, _ = gh.load_mamba_stack(model_name, device, mamba_repo)
        return m
    if name == "steered":
        m, _ = gh.load_steered_stack(
            model_name,
            device,
            steered_repo,
            steered_dir=None,
            steering_layers=list(steering_layers),
            steering_strength=steering_strength,
        )
        return m
    if name == "mamba2":
        m, _ = gh.load_mamba2_stack(
            model_name,
            device,
            mamba2_repo,
            base_mamba_only=mamba2_base_only,
            mamba2_straight=False,
        )
        return m
    if name == "mamba2_straight":
        m, _ = gh.load_mamba2_stack(
            model_name,
            device,
            mamba2_repo,
            base_mamba_only=mamba2_base_only,
            mamba2_straight=True,
        )
        return m
    raise ValueError(name)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Per-layer forward microbenchmark: mamba / steered / mamba2 / mamba2_straight."
    )
    p.add_argument("--layer", type=int, default=0, help="Index into model.backbone.layers.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--model", type=str, default=gh.MAMBA_HF_130M, help="HF id for base Mamba (all variants here).")
    p.add_argument(
        "--variants",
        nargs="+",
        choices=("mamba", "steered", "mamba2", "mamba2_straight"),
        default=["mamba", "steered", "mamba2", "mamba2_straight"],
        help="Stacks to benchmark in order (mamba2_straight = Mamba2 with CUDA barriers between gates/SSMs).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda | cpu. Default: cuda if available, otherwise cpu.",
    )
    p.add_argument("--mamba-repo", type=str, default=gh.DEFAULT_MAMBA_REPO)
    p.add_argument("--steered-repo", type=str, default=gh.DEFAULT_STEERED_REPO)
    p.add_argument("--mamba2-repo", type=str, default=gh.DEFAULT_MAMBA2_REPO)
    p.add_argument("--steering-layers", type=int, nargs="+", default=[0, 6, 12, 18])
    p.add_argument("--steering-strength", type=float, default=5.0)
    p.add_argument("--mamba2-base-only", action="store_true", help="Skip attach_mamba2_layers (for debugging).")
    p.add_argument("-q", "--quiet", action="store_true", help="Less console noise from loaders.")
    args = p.parse_args()

    if args.quiet:
        logging.basicConfig(level=logging.WARNING)

    if args.device is None:
        device_s = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_s = args.device
    if device_s.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested (--device cuda) but torch.cuda.is_available() is False. Use --device cpu.")

    use_cuda_events = device_s.startswith("cuda")
    device = torch.device(device_s)
    timer_label = "CUDA event ms" if use_cuda_events else "wall ms (perf_counter)"
    print(f"Timer: {timer_label}  device={device_s}")
    rows: List[Tuple[str, float, float, float]] = []

    for name in args.variants:
        if name in ("mamba2", "mamba2_straight") and args.mamba2_base_only:
            print(f"\n--- {name} (--mamba2-base-only): vanilla blocks only ---")
        else:
            print(f"\n--- Loading: {name} ---")
        model = _load_variant(
            name,
            model_name=args.model,
            device=device_s,
            mamba_repo=args.mamba_repo,
            steered_repo=args.steered_repo,
            mamba2_repo=args.mamba2_repo,
            steering_layers=args.steering_layers,
            steering_strength=args.steering_strength,
            mamba2_base_only=args.mamba2_base_only,
        )
        model.to(device)
        model.eval()

        hidden = int(model.config.hidden_size)
        dtype = next(model.parameters()).dtype
        x = _make_input(args.batch_size, args.seq_len, hidden, dtype, device)

        layer = _resolve_layer(model, args.layer)
        total_ms, avg_ms = _bench_layer_forward(
            layer,
            x,
            iters=args.iters,
            warmup=args.warmup,
            use_cuda_events=use_cuda_events,
        )
        tok = float(args.batch_size * args.seq_len)
        tps = tok / (avg_ms / 1000.0) if avg_ms > 0 else float("nan")

        print(
            f"{name}: layer={args.layer}  iters={args.iters}  "
            f"total_ms={total_ms:.3f}  avg_ms/iter={avg_ms:.6f}  "
            f"~tokens/s (B*S/avg_iter)={tps:.1f}"
        )
        rows.append((name, total_ms, avg_ms, tps))

        del model, layer, x
        gc.collect()
        if use_cuda_events:
            torch.cuda.empty_cache()

    baseline_avg = rows[0][2]
    baseline_name = rows[0][0]
    print(f"\n=== Summary (avg ms / iter; ratio vs first variant `{baseline_name}`) ===")
    print(f"{'variant':<18} {'avg_ms':>12} {'vs_first':>12} {'tokens/s':>14}")
    print("-" * 60)
    for name, _total, avg_ms, tps in rows:
        ratio = avg_ms / baseline_avg if baseline_avg > 0 else float("nan")
        print(f"{name:<18} {avg_ms:>12.6f} {ratio:>12.3f}x {tps:>14.1f}")


if __name__ == "__main__":
    main()
