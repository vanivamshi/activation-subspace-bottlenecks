#!/usr/bin/env python3
"""
Next-token prediction (causal LM) with two related time notions (see below), matching
next-token scoring in:
  - /home/vamshi/Videos/mamba_interpretability_1/perplexity.py

Compares stacks (each loads code + weights from its repo root on sys.path where applicable):
  - mamba: vanilla Mamba loader; **always** ``state-spaces/mamba-130m-hf`` (``--model`` does not apply).
  - mamba1.4: same loader, fixed ``state-spaces/mamba-1.4b-hf``.
  - mamba2.8: same loader, fixed ``state-spaces/mamba-2.8b-hf``.
  - steered: mamba_steered_interpretability_1 (in-memory SteeredMambaModel or --steered-dir bundle)
  - mamba2: mamba_stable_interpretability_1/mamba_2_interpretability_1 (attach_mamba2_layers)
  - mamba2_straight: same as mamba2 but Mamba2LayerStraight (CUDA sync between each gate and SSM head)
  - mamba2hf: Hugging Face Mamba-2 (SSD) via Mamba2ForCausalLM / AutoModel
    (https://huggingface.co/docs/transformers/model_doc/mamba2 ; --mamba2hf-model)
  - densemamba: DenseSSM / DenseRetNet via HuggingFace (see --densemamba-model, --densemamba-repo)
  - miniplm: MiniPLM-Mamba on HuggingFace (see --miniplm-model); optional --miniplm-repo for local clone
    (training code: https://github.com/thu-coai/MiniPLM ; weights e.g. MiniLLM/MiniPLM-Mamba-130M per steering_s6_types_miniplm.py)
  - safari: Hyena LM from HazyResearch Safari (https://github.com/HazyResearch/safari); see --safari-repo,
    optional --safari-checkpoint (steering_s6_types_hyena.load_hyena_model). GPT-2 tokenizer.

What “GPU hours” usually means for billing
------------------------------------------
**Billing GPU-hours** = (number of GPUs you reserve) × (wall-clock hours those GPUs are tied up
for the job). Example: 64 GPUs × 10 hours = 640 GPU-hours; multiply by your $/GPU-hour for cost.
Same logical work on a faster GPU finishes in fewer wall hours → fewer billing GPU-hours.

This script reports billing GPU-hours as: ``--num-gpus`` × (wall seconds in the chosen billing
window) / 3600. Use ``--billing-include-load`` to count from the start of model load; default is
workload-only (after ``model.to(device)`` through end of selected phases).

**Workloads** (``--workloads``): ``pred`` = next-token / perplexity-style eval (no grad); ``train``
= causal LM training steps (forward + loss + backward + optimizer). Default is both: prediction
first (unchanged weights), then training (GPU time for both counts toward billing wall clock).

Active / kernel time (not the same as billing)
----------------------------------------------
**Kernel GPU-seconds** use CUDA events inside each timed block (prediction: forward + CE; training:
forward + backward + step). Sum across phases is also reported. Billing GPU-hours still use wall
clock × ``--num-gpus``.


Run:
# Default: next-token prediction + training GPU accounting
python gpu_hours.py --dataset text

# Only perplexity / prediction (old behavior)
python gpu_hours.py --workloads pred

# Only training steps
python gpu_hours.py --workloads train --train-steps 100

---
Saved benchmark logs (reference only):

Results
/home/vamshi/Videos/mamba_interpretability_1/.venv/bin/python /home/vamshi/Videos/gpu_hours/gpu_hours.py --workloads pred --device cpu 2>&1

Billing GPU-hours = --num-gpus (1) × wall hours (pred)
variant           ppl   bill_gpu_h     wall_s   pred_k_h    trn_k_h    tot_k_h
------------------------------------------------------------------------------
mamba         46.9522     0.000069      0.250   0.000068   0.000000   0.000068
steered       51.8310     0.000068      0.244   0.000067   0.000000   0.000067
mamba2       166.2197     0.000073      0.261   0.000072   0.000000   0.000072


approximate for entire wikitext dataset

/home/vamshi/Videos/mamba_interpretability_1/.venv/bin/python /home/vamshi/Videos/gpu_hours/gpu_hours.py --dataset wikitext --workloads pred --wikitext-rows 10 --device cpu --variants mamba steered mamba2 2>&1
10 samples testing

Mamba 130M + 1.4B + 2.8B in one run (``mamba`` = 130M only; sizes are separate variant names):

/home/vamshi/Videos/mamba_interpretability_1/.venv/bin/python /home/vamshi/Videos/gpu_hours/gpu_hours.py --dataset wikitext --workloads pred --wikitext-rows 10 --device cpu --variants mamba mamba1.4 mamba2.8 2>&1


Billing GPU-hours = --num-gpus (1) × wall hours (pred)
variant           ppl   bill_gpu_h     wall_s   pred_k_h    trn_k_h    tot_k_h
------------------------------------------------------------------------------
mamba         39.7764     0.004191     15.088   0.004184   0.000000   0.004184
steered       46.6979     0.003424     12.325   0.003419   0.000000   0.003419
mamba2       320.8127     0.003390     12.203   0.003385   0.000000   0.003385
densemamba    31.6317     0.004218     15.183   0.004214   0.000000   0.004214
mamba2hf      41.0204     0.004500     15.799   0.004495   0.000000   0.004495
miniplm       26.7450     0.004891     17.608   0.004883   0.000000   0.004883
safari     53120.2810     0.000747      2.689   0.000743   0.000000   0.000743


Tested on RTX 4090
variant           ppl   bill_gpu_h     wall_s   pred_k_h    trn_k_h    tot_k_h
------------------------------------------------------------------------------
variant           ppl   bill_gpu_h     wall_s   pred_k_h    trn_k_h    tot_k_h
------------------------------------------------------------------------------
mamba         39.8006     0.000276      0.993   0.000260   0.000000   0.000260
mamba1.4(old)      19.5493     0.000449      1.616   0.000447   0.000000   0.000447
mamba1.4              19.5493     0.001241      4.469   0.001214   0.000000   0.001214
mamba1.4_stable 15851.6774     0.001279      4.603   0.001276   0.000000   0.001276
mamba2.8(old)      16.2310     0.000558      2.009   0.000556   0.000000   0.000556
mamba2.8              16.2310     0.001515      5.455   0.001503   0.000000   0.001503
mamba2.8_stable     6755.8377     0.001684      6.062   0.001681   0.000000   0.001681

Latency → throughput (pred)
---------------------------
If latency τ (e.g. 0.709 s) is for **one** forward on a **single** sequence of length L (batch size 1):

    tokens/s = L / τ

**Example:** L = 1024 tokens and τ = 0.709 s → tokens/s ≈ 1024 / 0.709 ≈ 1444.

If batch size B > 1 and each sequence in the batch has length L (total scored input tokens T = B × L
when lengths are uniform):

    tokens/s = (B × L) / τ = T / τ

**This script (``pred`` workload):** ``perplexity_on_corpus_texts_timed`` runs **batch size B = 1** for
every variant: one Wikitext document (one tokenizer call) per forward. There is **no** multi-row
batching; L **varies per document** (each line’s token length after truncation).

**Per-variant (table above) — pred batch size and sequence length**

==========  ===  ==========================================================
variant     B    L (tokens per forward)
==========  ===  ==========================================================
mamba       1    Per doc; cap from ``_effective_max_length`` (130M checkpoint; often **2048**).
mamba1.4    1    Same loader as **mamba**; cap from 1.4B config + tokenizer.
mamba2.8    1    Same loader as **mamba**; cap from 2.8B config + tokenizer.
steered     1    Uses ``--model`` as base; default 130M — same idea as **mamba** when default.
mamba2      1    Same HF tokenizer/cap as **mamba** (Mamba2 blocks on the same base).
densemamba  1    Per doc; cap from that checkpoint’s config + tokenizer
                 (``_effective_max_length`` after load).
mamba2hf    1    Per doc; cap from ``--mamba2hf-model`` config + tokenizer.
miniplm     1    Per doc; cap from ``--miniplm-model`` config + tokenizer.
safari      1    Per doc; cap from ``_effective_max_length`` on the Hyena stack;
                 **GPT-2** tokenizer → different L than Mamba-130M on the same text.
==========  ===  ==========================================================

**Corpus run:** total scored tokens and kernel time are printed as ``pred: tokens=…`` and
``pred_kern_s=…``; effective **aggregate** tokens/s ≈ ``tokens / pred_kernel_seconds`` for that
variant (sum of per-doc forwards, still B=1 each).


for 50 samples
------------------------------------------------------------------------------
mamba         38.9552     0.013167     47.400   0.013153   0.000000   0.013153


Summary table column meanings (printed at end of a run):
- variant: stack name (mamba / mamba1.4 / mamba2.8 / steered / mamba2 / mamba2_straight / mamba2hf / densemamba / miniplm / safari).
- ppl: perplexity from the pred phase (exp of mean NLL over scored tokens); n/a if pred was off.
- bill_gpu_h: billing GPU-hours = --num-gpus × (wall seconds in billing window) / 3600.
- wall_s: wall-clock seconds in that billing window (not multiplied by GPU count).
- pred_k_h: prediction kernel GPU-hours (CUDA events around forward + CE for pred; perf_counter on CPU).
- trn_k_h: training kernel GPU-hours (forward + backward + optimizer for train); 0 if train was off.
- tot_k_h: total kernel GPU-hours (pred_k_h + trn_k_h).
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# Default repo roots (override with --mamba-repo, --steered-repo, --mamba2-repo)
DEFAULT_MAMBA_REPO = "/home/vamshi/Videos/mamba_interpretability_1"
# HF Mamba (SSM) checkpoints for --variants mamba / mamba1.4 / mamba2.8 (fixed per variant)
MAMBA_HF_130M = "state-spaces/mamba-130m-hf"
MAMBA_HF_1_4B = "state-spaces/mamba-1.4b-hf"
MAMBA_HF_2_8B = "state-spaces/mamba-2.8b-hf"
DEFAULT_STEERED_REPO = "/home/vamshi/Videos/mamba_steered_interpretability_1"
DEFAULT_MAMBA2_REPO = "/home/vamshi/Videos/mamba_stable_interpretability_1/mamba_2_interpretability_1"
# DenseSSM clone (https://github.com/WailordHe/DenseSSM); optional for local modeling/weights
_DEFAULT_DENSESSM_NEXT_TO_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DenseSSM")
DEFAULT_DENSESSM_REPO = (
    _DEFAULT_DENSESSM_NEXT_TO_SCRIPT
    if os.path.isdir(_DEFAULT_DENSESSM_NEXT_TO_SCRIPT)
    else "/home/vamshi/DenseSSM"
)
DEFAULT_DENSEMAMBA_HF_MODEL = "jamesHD2001/DenseMamba-350M"
# HF Transformers Mamba-2 (SSD); default matches HF docs examples (large — use GPU or override).
DEFAULT_MAMBA2_HF_MODEL = "mistralai/Mamba-Codestral-7B-v0.1"
# MiniPLM: HF checkpoint (steering_s6_types_miniplm.load_miniplm_mamba_model); repo clone optional for training/scripts
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MINIPLM_SIBLING = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "MiniPLM"))
DEFAULT_MINIPLM_REPO = (
    _DEFAULT_MINIPLM_SIBLING
    if os.path.isdir(_DEFAULT_MINIPLM_SIBLING)
    else "/home/vamshi/Videos/MiniPLM"
)
DEFAULT_MINIPLM_HF_MODEL = "MiniLLM/MiniPLM-Mamba-130M"
_DEFAULT_SAFARI_SIBLING = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "safari"))
DEFAULT_SAFARI_REPO = (
    _DEFAULT_SAFARI_SIBLING
    if os.path.isdir(_DEFAULT_SAFARI_SIBLING)
    else "/home/vamshi/Videos/safari"
)

_MODULES_TO_DROP = (
    "mamba_model_loader",
    "create_steered_model",
    "steered_mamba",
    "mamba2_layer",
)


def _purge_interpretability_modules() -> None:
    for name in list(sys.modules.keys()):
        if name in _MODULES_TO_DROP:
            del sys.modules[name]


def forward_next_token_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    kwargs = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    return model(**kwargs).logits


def shift_logits_and_targets_for_causal_lm(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = targets[..., 1:].contiguous()
    return shift_logits, shift_targets


def token_nll_sum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> Tuple[float, int]:
    shift_logits, shift_targets = shift_logits_and_targets_for_causal_lm(logits, targets)
    v = shift_logits.size(-1)
    flat_logits = shift_logits.reshape(-1, v)
    flat_targets = shift_targets.reshape(-1)
    nll = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction="none")
    mask = flat_targets.ne(ignore_index)
    return float(nll[mask].sum().item()), int(mask.sum().item())


def _effective_max_length(model: torch.nn.Module, tokenizer) -> int:
    cap = getattr(model.config, "max_position_embeddings", None) or 2048
    tm = getattr(tokenizer, "model_max_length", cap)
    if tm is None or tm > 100_000:
        tm = cap
    return int(max(2, min(cap, tm)))


def perplexity_on_corpus_texts_timed(
    model: torch.nn.Module,
    tokenizer,
    texts: Iterable[str],
    device: torch.device,
    *,
    max_length: int,
    ignore_index: int = -100,
) -> Tuple[float, float, float, int, int]:
    """
    Returns (mean_nll, perplexity, prediction_gpu_seconds, total_predicted_tokens, num_docs_scored).

    `prediction_gpu_seconds` is summed **only inside the loop**, per document, across the exact
    next-token prediction work: `forward_next_token_logits` + `token_nll_sum` (same as perplexity.py).
    """
    total_nll = 0.0
    total_tokens = 0
    prediction_seconds = 0.0
    docs = 0
    model.eval()
    use_cuda_events = device.type == "cuda"

    with torch.no_grad():
        for text in texts:
            if not text or not str(text).strip():
                continue
            enc = tokenizer(
                str(text),
                max_length=max_length,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            else:
                attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
            labels = labels.masked_fill(attention_mask == 0, ignore_index)

            # Billable window: only next-token forward + NLL (runs while the program predicts tokens).
            if use_cuda_events:
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_end = torch.cuda.Event(enable_timing=True)
                ev_start.record()
                logits = forward_next_token_logits(model, input_ids, attention_mask=attention_mask)
                s, c = token_nll_sum(logits, labels, ignore_index=ignore_index)
                ev_end.record()
                torch.cuda.synchronize(device)
                prediction_seconds += ev_start.elapsed_time(ev_end) / 1000.0
            else:
                t0 = time.perf_counter()
                logits = forward_next_token_logits(model, input_ids, attention_mask=attention_mask)
                s, c = token_nll_sum(logits, labels, ignore_index=ignore_index)
                prediction_seconds += time.perf_counter() - t0

            total_nll += s
            total_tokens += c
            docs += 1

    if total_tokens == 0:
        return float("nan"), float("nan"), prediction_seconds, 0, docs
    mean_nll = total_nll / total_tokens
    return mean_nll, math.exp(mean_nll), prediction_seconds, total_tokens, docs


def training_steps_timed(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    *,
    max_length: int,
    num_steps: int,
    lr: float,
) -> Tuple[float, int, int]:
    """
    Causal LM fine-tuning style steps: ``model(..., labels=...)``, backward, AdamW step.
    Returns (kernel_seconds, label_tokens_processed, steps_executed).
    Kernel time is summed per step (CUDA events around forward+backward+optimizer).
    """
    if num_steps <= 0 or not texts:
        return 0.0, 0, 0

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    use_cuda_events = device.type == "cuda"
    kernel_seconds = 0.0
    label_tokens = 0
    steps_run = 0
    n_texts = len(texts)

    for step in range(num_steps):
        raw = texts[step % n_texts]
        if not raw or not str(raw).strip():
            continue
        enc = tokenizer(
            str(raw),
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = input_ids.clone()

        opt.zero_grad(set_to_none=True)

        if use_cuda_events:
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss
            loss.backward()
            opt.step()
            ev_end.record()
            torch.cuda.synchronize(device)
            kernel_seconds += ev_start.elapsed_time(ev_end) / 1000.0
        else:
            t0 = time.perf_counter()
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss
            loss.backward()
            opt.step()
            kernel_seconds += time.perf_counter() - t0

        # Tokens with a gradient signal (shifted LM: seq_len - 1 targets typical)
        label_tokens += int(input_ids.numel())
        steps_run += 1

    return kernel_seconds, label_tokens, steps_run


def _minipile_split_spec(
    split_name: str,
    *,
    train_docs: int,
    validation_docs: int,
    test_start: int,
    test_end: int,
) -> str:
    if split_name == "train":
        return f"train[:{train_docs}]"
    if split_name == "validation":
        return f"validation[:{validation_docs}]"
    if split_name == "test":
        return f"test[{test_start}:{test_end}]"
    raise ValueError(f"Unknown split {split_name!r}")


def iter_texts_for_dataset(args: argparse.Namespace) -> Tuple[Iterable[str], str]:
    if args.dataset == "text":
        return ([args.text], "single --text")

    if args.dataset == "wikitext":
        from datasets import load_dataset

        spec = f"{args.wikitext_split}[:{args.wikitext_rows}]"
        ds = load_dataset("wikitext", args.wikitext_config, split=spec)
        texts = (row[args.text_field] for row in ds)
        return (texts, f"wikitext/{args.wikitext_config} {spec}")

    if args.dataset == "minipile":
        from datasets import load_dataset

        spec = _minipile_split_spec(
            args.minipile_split,
            train_docs=args.train_docs,
            validation_docs=args.validation_docs,
            test_start=args.test_start,
            test_end=args.test_end,
        )
        ds = load_dataset(args.hf_dataset, split=spec)
        texts = (row[args.text_field] for row in ds)
        return (texts, f"{args.hf_dataset} {spec}")

    raise ValueError(args.dataset)


def load_mamba_stack(model_name: str, device: str, repo_root: str):
    _purge_interpretability_modules()
    sys.path.insert(0, repo_root)
    try:
        import mamba_model_loader

        return mamba_model_loader.load_mamba_model_and_tokenizer(model_name, device=device)
    finally:
        if sys.path and sys.path[0] == repo_root:
            sys.path.pop(0)


def load_steered_stack(
    model_name: str,
    device: str,
    repo_root: str,
    *,
    steered_dir: Optional[str],
    steering_layers: List[int],
    steering_strength: float,
):
    _purge_interpretability_modules()
    sys.path.insert(0, repo_root)
    try:
        import mamba_model_loader
        from create_steered_model import SteeredMambaModel, load_steered_model
        from steered_mamba import SimpleSteering

        if steered_dir:
            model, tokenizer, _cfg = load_steered_model(steered_dir, device=device)
            return model, tokenizer
        model, tokenizer = mamba_model_loader.load_mamba_model_and_tokenizer(model_name, device=device)
        steering = SimpleSteering(model)
        model = SteeredMambaModel(
            model,
            steering,
            layer_indices=list(steering_layers),
            strength=steering_strength,
        )
        return model, tokenizer
    finally:
        if sys.path and sys.path[0] == repo_root:
            sys.path.pop(0)


def load_mamba2_stack(
    model_name: str,
    device: str,
    repo_root: str,
    *,
    base_mamba_only: bool,
    mamba2_straight: bool = False,
):
    _purge_interpretability_modules()
    sys.path.insert(0, repo_root)
    try:
        import mamba_model_loader

        model, tokenizer = mamba_model_loader.load_mamba_model_and_tokenizer(model_name, device=device)
        if not base_mamba_only:
            from mamba2_layer import attach_mamba2_layers

            attach_mamba2_layers(model, straight=mamba2_straight)
        return model, tokenizer
    finally:
        if sys.path and sys.path[0] == repo_root:
            sys.path.pop(0)


def _causal_lm_from_pretrained(model_cls, model_name: str, device: str):
    """Call ``from_pretrained`` with ``dtype`` or ``torch_dtype`` depending on transformers version."""
    dtype = torch.float16 if device == "cuda" else torch.float32
    kw = {"low_cpu_mem_usage": True}
    try:
        return model_cls.from_pretrained(model_name, dtype=dtype, **kw)
    except TypeError:
        return model_cls.from_pretrained(model_name, torch_dtype=dtype, **kw)


def load_mamba2_hf_stack(model_name: str, device: str) -> Tuple[torch.nn.Module, object]:
    """
    Load official Hugging Face Mamba-2 (SSD) causal LM.

    Tries ``Mamba2ForCausalLM`` first (see transformers model doc), then ``AutoModelForCausalLM``,
    matching the fallback style in ``steering_s6_types_mamba2.py`` ``load_model_variant``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    try:
        from transformers import Mamba2ForCausalLM

        model = _causal_lm_from_pretrained(Mamba2ForCausalLM, model_name, device)
    except ImportError:
        pass
    except Exception:
        model = None

    if model is None:
        model = _causal_lm_from_pretrained(AutoModelForCausalLM, model_name, device)

    return model, tokenizer


def load_miniplm_stack(model_name: str, device: str, miniplm_repo: str) -> Tuple[torch.nn.Module, object]:
    """
    Load MiniPLM-Mamba from Hugging Face (``steering_s6_types_miniplm.load_miniplm_mamba_model``).

    Default weights: https://huggingface.co/MiniLLM/MiniPLM-Mamba-130M
    Paper: https://arxiv.org/abs/2410.17215

    Preprends ``miniplm_repo`` on ``sys.path`` when that directory exists (e.g. clone of
    https://github.com/thu-coai/MiniPLM) so local utilities can be imported alongside HF loads.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, MambaForCausalLM

    repo_norm = os.path.normpath(miniplm_repo) if miniplm_repo else ""
    path_inserted = False
    if repo_norm and os.path.isdir(repo_norm):
        norms = {os.path.normpath(p) for p in sys.path}
        if repo_norm not in norms:
            sys.path.insert(0, repo_norm)
            path_inserted = True
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = None
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                try:
                    model = _causal_lm_from_pretrained(MambaForCausalLM, model_name, device)
                except Exception:
                    model = _causal_lm_from_pretrained(AutoModelForCausalLM, model_name, device)
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(5 * (2**attempt))
        if model is None:
            raise RuntimeError("MiniPLM load failed after retries") from last_err

        return model, tokenizer
    finally:
        if path_inserted and sys.path and os.path.normpath(sys.path[0]) == repo_norm:
            sys.path.pop(0)


class _SafariHyenaHFAdapter(torch.nn.Module):
    """Wrap Safari ``SimpleLMHeadModel`` so ``gpu_hours`` can use ``.logits`` / ``.loss`` like HuggingFace."""

    def __init__(self, inner: torch.nn.Module, *, max_position_embeddings: int):
        super().__init__()
        self.inner = inner
        self.config = types.SimpleNamespace(max_position_embeddings=max_position_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> types.SimpleNamespace:
        out = self.inner(input_ids)
        first = out[0] if isinstance(out, tuple) else out
        logits = first.logits
        loss = None
        if labels is not None:
            lab = labels
            if attention_mask is not None:
                lab = lab.clone()
                lab = lab.masked_fill(attention_mask == 0, -100)
            shift_logits, shift_labels = shift_logits_and_targets_for_causal_lm(logits, lab)
            v = shift_logits.size(-1)
            loss = F.cross_entropy(
                shift_logits.reshape(-1, v),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        return types.SimpleNamespace(logits=logits, loss=loss)


def load_safari_hyena_stack(
    device: str,
    safari_repo: str,
    *,
    checkpoint_path: Optional[str],
    model_size: str,
    hidden_dim: int,
    num_layers: int,
    l_max: int,
) -> Tuple[torch.nn.Module, object]:
    """
    Hyena language model from the Safari repo (``steering_s6_types_hyena.load_hyena_model``).

    Expects a clone of https://github.com/HazyResearch/safari on ``sys.path``. Optional
    ``.pt`` checkpoint; otherwise builds the architecture (random init — metrics are not meaningful).
    Tokenizer: ``gpt2``.
    """
    from transformers import AutoTokenizer

    repo_norm = os.path.normpath(safari_repo)
    if not os.path.isdir(repo_norm):
        raise FileNotFoundError(
            f"Safari repo not found: {repo_norm!r}. Clone: git clone https://github.com/HazyResearch/safari.git"
        )

    inserted = False
    norms = {os.path.normpath(p) for p in sys.path}
    if repo_norm not in norms:
        sys.path.insert(0, repo_norm)
        inserted = True
    try:
        import src  # noqa: F401

        from src.models.sequence.simple_lm import SimpleLMHeadModel

        vocab_size = 50257
        d_inner = hidden_dim * 4
        # d_model is supplied by Block as mixer_cls(dim); do not pass it here (duplicate arg).
        hyena_layer_config = {
            "_name_": "hyena",
            "l_max": l_max,
            "order": 2,
            "filter_order": 64,
        }

        ck: Optional[Path] = None
        if checkpoint_path:
            p = Path(checkpoint_path)
            if p.is_file():
                ck = p
        if ck is None:
            for cand in (
                Path(repo_norm) / "checkpoints" / f"hyena-{model_size.lower()}",
                Path(repo_norm) / "checkpoints" / "hyena",
                Path(repo_norm) / "hyena-150m.pt",
                Path(repo_norm) / "checkpoints" / "hyena-150m.pt",
            ):
                if cand.is_file():
                    ck = cand
                    break

        inner = SimpleLMHeadModel(
            d_model=hidden_dim,
            n_layer=num_layers,
            d_inner=d_inner,
            vocab_size=vocab_size,
            layer=hyena_layer_config,
            max_position_embeddings=0,
        )
        if ck is not None:
            try:
                blob = torch.load(ck, map_location="cpu", weights_only=False)
            except TypeError:
                blob = torch.load(ck, map_location="cpu")
            if isinstance(blob, dict):
                if "model" in blob:
                    inner.load_state_dict(blob["model"], strict=False)
                elif "state_dict" in blob:
                    inner.load_state_dict(blob["state_dict"], strict=False)
                else:
                    inner.load_state_dict(blob, strict=False)
            else:
                inner.load_state_dict(blob, strict=False)

        dtype = torch.float16 if device == "cuda" else torch.float32
        inner = inner.to(device=device, dtype=dtype)
        inner.eval()

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        wrapped = _SafariHyenaHFAdapter(inner, max_position_embeddings=l_max)
        return wrapped, tokenizer
    finally:
        if inserted and sys.path and os.path.normpath(sys.path[0]) == repo_norm:
            sys.path.pop(0)


def _patch_transformers_top_k_top_p_filtering() -> None:
    """DenseSSM checkpoints may import removed transformers helpers; mirror steering_s6_types_densemamba."""
    try:
        from transformers import top_k_top_p_filtering  # noqa: F401
    except ImportError:
        import transformers

        def top_k_top_p_filtering_compat(
            logits: torch.Tensor,
            top_k: int = 0,
            top_p: float = 1.0,
            filter_value: float = float("-inf"),
        ) -> torch.Tensor:
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, filter_value)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, filter_value)
            return logits

        transformers.top_k_top_p_filtering = top_k_top_p_filtering_compat  # type: ignore[attr-defined]


_DENSEMAMBA_LOCAL_DIRS = {
    "jamesHD2001/DenseMamba-350M": ("modeling", "dense_gau_retnet_350m"),
    "jamesHD2001/DenseMamba-1.3B": ("modeling", "dense_gau_retnet_1p3b"),
}


def load_densemamba_stack(
    hf_model_id: str,
    device: str,
    densessm_repo: str,
    *,
    hf_local_files_only: bool = False,
) -> Tuple[torch.nn.Module, object]:
    """
    Load DenseMamba (DenseRetNet) with ``trust_remote_code=True``.
    Tries local weights under ``densessm_repo``/modeling/... first when present, else HF id.
    """
    _patch_transformers_top_k_top_p_filtering()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device == "cuda" else torch.float32
    common = {"torch_dtype": dtype, "low_cpu_mem_usage": True, "trust_remote_code": True}
    hf_kw = dict(common)
    if hf_local_files_only:
        hf_kw["local_files_only"] = True

    repo_norm = os.path.normpath(densessm_repo) if densessm_repo else ""
    path_inserted = False
    if repo_norm and os.path.isdir(repo_norm) and repo_norm not in map(os.path.normpath, sys.path):
        sys.path.insert(0, repo_norm)
        path_inserted = True
    try:
        model = None
        tokenizer = None

        if repo_norm and os.path.isdir(repo_norm):
            rel = _DENSEMAMBA_LOCAL_DIRS.get(hf_model_id)
            if rel:
                local_path = os.path.join(repo_norm, *rel)
                if os.path.isdir(local_path):
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            local_path,
                            **common,
                            local_files_only=True,
                        )
                        tokenizer = AutoTokenizer.from_pretrained(
                            local_path,
                            trust_remote_code=True,
                            local_files_only=True,
                        )
                    except Exception:
                        model, tokenizer = None, None

        if model is None or tokenizer is None:
            model = AutoModelForCausalLM.from_pretrained(hf_model_id, **hf_kw)
            tokenizer = AutoTokenizer.from_pretrained(
                hf_model_id,
                trust_remote_code=True,
                local_files_only=hf_local_files_only,
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    finally:
        if path_inserted and sys.path and os.path.normpath(sys.path[0]) == repo_norm:
            sys.path.pop(0)


def _free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_workloads(
    name: str,
    model,
    tokenizer,
    device: torch.device,
    texts: Sequence[str],
    max_length_pred: int,
    workloads: Sequence[str],
    *,
    train_max_length: int,
    train_steps: int,
    train_lr: float,
) -> dict:
    """
    Run ``pred`` and/or ``train`` phases (prediction first, then training).
    Returns metrics including separate and total kernel GPU-seconds.
    """
    workloads_set = frozenset(workloads)
    pred_kernel_s = 0.0
    train_kernel_s = 0.0
    mean_nll = float("nan")
    ppl = float("nan")
    n_tok = 0
    n_docs = 0
    train_label_tokens = 0
    train_steps_run = 0

    if "pred" in workloads_set:
        mean_nll, ppl, pred_kernel_s, n_tok, n_docs = perplexity_on_corpus_texts_timed(
            model, tokenizer, texts, device, max_length=max_length_pred
        )

    if "train" in workloads_set:
        train_kernel_s, train_label_tokens, train_steps_run = training_steps_timed(
            model,
            tokenizer,
            texts,
            device,
            max_length=train_max_length,
            num_steps=train_steps,
            lr=train_lr,
        )

    total_kernel_s = pred_kernel_s + train_kernel_s
    return {
        "variant": name,
        "workloads": list(workloads),
        "mean_nll": mean_nll,
        "perplexity": ppl,
        "pred_kernel_gpu_seconds": pred_kernel_s,
        "train_kernel_gpu_seconds": train_kernel_s,
        "kernel_gpu_seconds": total_kernel_s,
        "pred_kernel_gpu_hours": pred_kernel_s / 3600.0,
        "train_kernel_gpu_hours": train_kernel_s / 3600.0,
        "kernel_gpu_hours": total_kernel_s / 3600.0,
        "predicted_tokens": n_tok,
        "documents": n_docs,
        "train_label_tokens": train_label_tokens,
        "train_steps_run": train_steps_run,
        "kernel_seconds_per_million_tokens": (pred_kernel_s / n_tok * 1e6) if n_tok else float("nan"),
        "train_kernel_s_per_million_label_toks": (train_kernel_s / train_label_tokens * 1e6)
        if train_label_tokens
        else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "GPU-hour accounting: next-token prediction (eval) and/or causal LM training steps. "
            "Billing = num_gpus × wall time; kernel = CUDA event sums per phase."
        )
    )
    p.add_argument(
        "--workloads",
        nargs="+",
        choices=("pred", "train"),
        default=["pred", "train"],
        help="pred: perplexity-style next-token eval; train: forward+backward+AdamW (default: both).",
    )
    p.add_argument("--train-steps", type=int, default=10, help="Training optimizer steps (per variant).")
    p.add_argument("--train-lr", type=float, default=3e-5, help="AdamW learning rate for training phase.")
    p.add_argument(
        "--train-max-length",
        type=int,
        default=512,
        help="Max sequence length for training steps (truncate; smaller avoids OOM).",
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        metavar="N",
        help="GPUs counted for billing GPU-hours: N × wall_hours (default: 1).",
    )
    p.add_argument(
        "--billing-include-load",
        action="store_true",
        help=(
            "Start billing wall clock at model load start. "
            "Default: billing starts after model.to(device), covering only selected --workloads."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default="state-spaces/mamba-130m-hf",
        help=(
            "HF base weights for **steered** and **mamba2** only. "
            "Variant **mamba** is always mamba-130m-hf; use **mamba1.4** / **mamba2.8** for larger Mamba checkpoints."
        ),
    )
    p.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "mamba",
            "mamba1.4",
            "mamba2.8",
            "steered",
            "mamba2",
            "mamba2_straight",
            "mamba2hf",
            "densemamba",
            "miniplm",
            "safari",
        ),
        default=["mamba", "steered", "mamba2"],
    )
    p.add_argument("--mamba-repo", type=str, default=DEFAULT_MAMBA_REPO)
    p.add_argument("--steered-repo", type=str, default=DEFAULT_STEERED_REPO)
    p.add_argument("--mamba2-repo", type=str, default=DEFAULT_MAMBA2_REPO)
    p.add_argument(
        "--densemamba-repo",
        type=str,
        default=DEFAULT_DENSESSM_REPO,
        help="DenseSSM git root (optional local modeling/weights under modeling/).",
    )
    p.add_argument(
        "--densemamba-model",
        type=str,
        default=DEFAULT_DENSEMAMBA_HF_MODEL,
        help="HuggingFace model id for densemamba variant (e.g. jamesHD2001/DenseMamba-350M).",
    )
    p.add_argument(
        "--densemamba-local-only",
        action="store_true",
        help="DenseMamba: HF load uses local_files_only (cache/repo only, no download).",
    )
    p.add_argument(
        "--mamba2hf-model",
        type=str,
        default=DEFAULT_MAMBA2_HF_MODEL,
        help=(
            "Hugging Face model id for mamba2hf (Transformers Mamba2/SSD). "
            "Default is mistralai/Mamba-Codestral-7B-v0.1 per HF docs; needs GPU RAM or a smaller checkpoint."
        ),
    )
    p.add_argument(
        "--miniplm-repo",
        type=str,
        default=DEFAULT_MINIPLM_REPO,
        help="MiniPLM git root (e.g. clone of https://github.com/thu-coai/MiniPLM); optional for HF-only eval.",
    )
    p.add_argument(
        "--miniplm-model",
        type=str,
        default=DEFAULT_MINIPLM_HF_MODEL,
        help="Hugging Face model id for miniplm (default MiniLLM/MiniPLM-Mamba-130M).",
    )
    p.add_argument(
        "--safari-repo",
        type=str,
        default=DEFAULT_SAFARI_REPO,
        help="HazyResearch Safari git root (https://github.com/HazyResearch/safari).",
    )
    p.add_argument(
        "--safari-checkpoint",
        type=str,
        default=None,
        help="Optional path to a Hyena .pt checkpoint; else search under safari repo or random init.",
    )
    p.add_argument(
        "--safari-model-size",
        type=str,
        default="150M",
        help="Label for default checkpoint search (e.g. hyena-150m.pt).",
    )
    p.add_argument("--safari-hidden", type=int, default=768, help="Hyena d_model.")
    p.add_argument("--safari-layers", type=int, default=12, help="Hyena layer count.")
    p.add_argument("--safari-l-max", type=int, default=2048, help="Hyena operator l_max / max seq hint.")
    p.add_argument("--steered-dir", type=str, default=None, help="Saved bundle; else in-memory steering.")
    p.add_argument("--steering-layers", type=int, nargs="+", default=[0, 6, 12, 18])
    p.add_argument("--steering-strength", type=float, default=5.0)
    p.add_argument(
        "--mamba2-base-only",
        action="store_true",
        help="Mamba2 repo: skip attach_mamba2_layers (vanilla HF Mamba only).",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dataset", choices=("text", "wikitext", "minipile"), default="text")
    p.add_argument(
        "--text",
        type=str,
        default="The game began development in 2010, carried out by a small team of people.",
    )
    p.add_argument("--wikitext-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--wikitext-split", type=str, default="train", choices=["train", "validation", "test"])
    p.add_argument("--wikitext-rows", type=int, default=10)
    p.add_argument("--hf-dataset", type=str, default="JeanKaddour/minipile")
    p.add_argument("--minipile-split", type=str, default="test", choices=["train", "validation", "test"])
    p.add_argument("--train-docs", type=int, default=100)
    p.add_argument("--validation-docs", type=int, default=100)
    p.add_argument("--test-start", type=int, default=100)
    p.add_argument("--test-end", type=int, default=120)
    p.add_argument("--text-field", type=str, default="text")
    args = p.parse_args()

    device_s = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    # Materialize texts once so each variant sees the same corpus (minipile/wikitext iterators are one-shot)
    texts_iter, desc = iter_texts_for_dataset(args)
    texts_list = list(texts_iter)
    print(f"Corpus: {desc}  (documents after empty filter: {len(texts_list)})")
    print(f"Workloads: {list(args.workloads)}")
    if args.dataset == "minipile" and args.test_end <= args.test_start:
        raise SystemExit("--test-end must be > --test-start")
    if not texts_list and (set(args.workloads) & {"pred", "train"}):
        raise SystemExit("No texts to run; pick another --dataset or increase rows.")

    rows = []
    for variant in args.variants:
        print(f"\n--- Loading: {variant} ---")
        t_load0 = time.perf_counter()
        t_bill_start = t_load0 if args.billing_include_load else None
        if variant == "mamba":
            model, tokenizer = load_mamba_stack(MAMBA_HF_130M, device_s, args.mamba_repo)
        elif variant == "mamba1.4":
            model, tokenizer = load_mamba_stack(MAMBA_HF_1_4B, device_s, args.mamba_repo)
        elif variant == "mamba2.8":
            model, tokenizer = load_mamba_stack(MAMBA_HF_2_8B, device_s, args.mamba_repo)
        elif variant == "steered":
            if args.steered_dir and args.model:
                print("Note: --steered-dir set; base --model is ignored for loading saved bundle.")
            model, tokenizer = load_steered_stack(
                args.model,
                device_s,
                args.steered_repo,
                steered_dir=args.steered_dir,
                steering_layers=list(args.steering_layers),
                steering_strength=args.steering_strength,
            )
        elif variant == "mamba2":
            model, tokenizer = load_mamba2_stack(
                args.model,
                device_s,
                args.mamba2_repo,
                base_mamba_only=args.mamba2_base_only,
                mamba2_straight=False,
            )
        elif variant == "mamba2_straight":
            model, tokenizer = load_mamba2_stack(
                args.model,
                device_s,
                args.mamba2_repo,
                base_mamba_only=args.mamba2_base_only,
                mamba2_straight=True,
            )
        elif variant == "mamba2hf":
            print(
                f"Note: mamba2hf uses --mamba2hf-model ({args.mamba2hf_model!r}); "
                "see https://huggingface.co/docs/transformers/model_doc/mamba2"
            )
            model, tokenizer = load_mamba2_hf_stack(args.mamba2hf_model, device_s)
        elif variant == "densemamba":
            print(f"Note: densemamba uses --densemamba-model ({args.densemamba_model!r}); --model is for other variants.")
            model, tokenizer = load_densemamba_stack(
                args.densemamba_model,
                device_s,
                args.densemamba_repo,
                hf_local_files_only=args.densemamba_local_only,
            )
        elif variant == "miniplm":
            print(
                f"Note: miniplm uses --miniplm-model ({args.miniplm_model!r}); "
                f"repo on path if present: {args.miniplm_repo!r}."
            )
            model, tokenizer = load_miniplm_stack(args.miniplm_model, device_s, args.miniplm_repo)
        elif variant == "safari":
            print(
                f"Note: safari (Hyena) repo={args.safari_repo!r} checkpoint={args.safari_checkpoint!r} "
                "(see steering_s6_types_hyena.load_hyena_model)."
            )
            model, tokenizer = load_safari_hyena_stack(
                device_s,
                args.safari_repo,
                checkpoint_path=args.safari_checkpoint,
                model_size=args.safari_model_size,
                hidden_dim=args.safari_hidden,
                num_layers=args.safari_layers,
                l_max=args.safari_l_max,
            )
        else:
            raise SystemExit(f"Unknown variant {variant}")
        t_load1 = time.perf_counter()
        model.to(device)
        model.eval()
        max_length = _effective_max_length(model, tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if t_bill_start is None:
            t_bill_start = time.perf_counter()
        print(
            f"Model load wall time: {t_load1 - t_load0:.3f}s "
            f"({'included in' if args.billing_include_load else 'excluded from'} billing window)"
        )
        row = run_workloads(
            variant,
            model,
            tokenizer,
            device,
            texts_list,
            max_length,
            args.workloads,
            train_max_length=args.train_max_length,
            train_steps=args.train_steps,
            train_lr=args.train_lr,
        )
        t_bill_end = time.perf_counter()
        wall_bill_s = t_bill_end - t_bill_start
        row["load_wall_seconds"] = t_load1 - t_load0
        row["wall_billing_seconds"] = wall_bill_s
        row["billing_gpu_hours"] = args.num_gpus * wall_bill_s / 3600.0
        row["num_gpus"] = args.num_gpus
        rows.append(row)
        ppl_s = f"{row['perplexity']:.4f}" if math.isfinite(row["perplexity"]) else "n/a"
        print(
            f"{variant}: ppl={ppl_s}  "
            f"bill_gpu_h={row['billing_gpu_hours']:.6f} ({args.num_gpus}×{wall_bill_s:.3f}s wall)  "
            f"pred_kern_s={row['pred_kernel_gpu_seconds']:.4f}  "
            f"train_kern_s={row['train_kernel_gpu_seconds']:.4f}  "
            f"total_kern_s={row['kernel_gpu_seconds']:.4f}"
        )
        if "pred" in args.workloads:
            print(
                f"         pred: tokens={row['predicted_tokens']}  "
                f"kernel_s/Mtok={row['kernel_seconds_per_million_tokens']:.4f}"
            )
        if "train" in args.workloads:
            print(
                f"         train: steps={row['train_steps_run']}  label_toks={row['train_label_tokens']}  "
                f"kernel_s/Mlabel_tok={row['train_kernel_s_per_million_label_toks']:.4f}"
            )
        _free_model(model)

    print("\n=== Summary ===")
    wl = "+".join(args.workloads)
    print(
        f"Billing GPU-hours = --num-gpus ({args.num_gpus}) × wall hours "
        f"({'load+' if args.billing_include_load else ''}{wl})"
    )
    hdr = (
        f"{'variant':<10} {'ppl':>10} {'bill_gpu_h':>12} {'wall_s':>10} "
        f"{'pred_k_h':>10} {'trn_k_h':>10} {'tot_k_h':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ppl_s = f"{r['perplexity']:.4f}" if math.isfinite(r["perplexity"]) else "n/a"
        print(
            f"{r['variant']:<10} {ppl_s:>10} {r['billing_gpu_hours']:>12.6f} {r['wall_billing_seconds']:>10.3f} "
            f"{r['pred_kernel_gpu_hours']:>10.6f} {r['train_kernel_gpu_hours']:>10.6f} "
            f"{r['kernel_gpu_hours']:>10.6f}"
        )


if __name__ == "__main__":
    main()
