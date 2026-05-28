#!/usr/bin/env python3
"""ESMC-300M D2Deep pathogenicity benchmark.

Evaluates ESMC-300M on the D2Deep variant dataset using a zero-shot
mutation log-probability delta:

    pathogenic_score = -(log p(mutant AA | context) - log p(WT AA | context))

Pathogenic variants are expected to be less probable than wild type in the
same sequence context, so high pathogenic_score → likely pathogenic.

Sequences longer than 510 residues are windowed around the mutation site to
fit the model's 512-token context (BOS + 510 residues + EOS).

Primary quality metric: ROC-AUC (deterministic across hardware for the same
floating-point precision).

Examples
--------
# Trainium2 — single NeuronCore, batch=1, regional compile
NEURON_RT_VISIBLE_CORES=0 NEURON_RT_NUM_CORES=1 \\
python experiments/esmc_d2deep_benchmark.py \\
    --device neuron \\
    --d2deep-csv data/d2deep_data/compiled_data.csv \\
    --batch-size 1 --compile-strategy regional \\
    --summary-json results/run.json

# H100 / CUDA — batch=16, no compile
python experiments/esmc_d2deep_benchmark.py \\
    --device cuda \\
    --d2deep-csv data/d2deep_data/compiled_data.csv \\
    --batch-size 16 --no-compile \\
    --hourly-cost 10.18 \\
    --summary-json results/run.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

# Add repo root to path so esm_neuron is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esm_neuron import NeuronESMC  # noqa: E402

_BOS = 0
_PAD = 1
_EOS = 2

_REQUIRED_COLUMNS = [
    "uniprot", "WT_sequence", "mut_sequence",
    "AA_orig", "position", "AA_targ", "label",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _crop_window(sequence: str, position: int, max_residues: int) -> tuple[str, int]:
    """Return a (window, window_position) pair centred on the mutation."""
    if len(sequence) <= max_residues:
        return sequence, position
    half = max_residues // 2
    start = max(1, min(position - half, len(sequence) - max_residues + 1))
    end = start + max_residues - 1
    return sequence[start - 1 : end], position - start + 1


def load_d2deep(csv_path: Path, seqlen: int, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[_REQUIRED_COLUMNS].dropna().copy()
    for col in ("WT_sequence", "mut_sequence", "AA_orig", "AA_targ"):
        df[col] = df[col].astype(str).str.upper()
    df["position"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["position", "label"])

    wt_len = df["WT_sequence"].str.len()
    valid = (
        df["label"].isin([0, 1])
        & (wt_len == df["mut_sequence"].str.len())
        & (df["position"] >= 1) & (df["position"] <= wt_len)
        & (df["AA_orig"].str.len() == 1) & (df["AA_targ"].str.len() == 1)
        & df.apply(lambda r: r["WT_sequence"][r["position"] - 1] == r["AA_orig"]
                   and r["mut_sequence"][r["position"] - 1] == r["AA_targ"], axis=1)
    )
    df = df[valid].copy()

    max_res = seqlen - 2
    windows = [_crop_window(str(r["WT_sequence"]), int(r["position"]), max_res)
               for _, r in df.iterrows()]
    df["window"] = [w[0] for w in windows]
    df["window_pos"] = [w[1] for w in windows]

    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _aa_token_ids(tokenizer: Any) -> dict[str, int]:
    return {aa: int(tokenizer.encode(aa, add_special_tokens=False)[0])
            for aa in "ACDEFGHIKLMNPQRSTVWY"}


def _tokenize_batch(tokenizer: Any, seqs: list[str], seqlen: int,
                    device: torch.device, pad_to: int) -> torch.Tensor:
    padded = list(seqs)
    while len(padded) < pad_to:
        padded.append(seqs[0])
    rows = []
    for s in padded:
        ids = tokenizer.encode(s, add_special_tokens=True)
        ids += [_PAD] * (seqlen - len(ids))
        rows.append(torch.tensor(ids, dtype=torch.long))
    return torch.stack(rows).to(device)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _summarise(vals: list[float]) -> tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0
    s = sorted(vals)
    p95 = s[min(len(s) - 1, round(0.95 * (len(s) - 1)))]
    return mean(vals), median(vals), p95


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    from esm.tokenization import EsmSequenceTokenizer

    print(f"Loading D2Deep from {args.d2deep_csv} …", flush=True)
    df = load_d2deep(args.d2deep_csv, args.seqlen, args.limit)
    print(f"  {len(df)} usable variants  "
          f"(benign={int((df['label']==0).sum())}  "
          f"pathogenic={int((df['label']==1).sum())})")

    tokenizer = EsmSequenceTokenizer()
    aa_ids = _aa_token_ids(tokenizer)
    device = torch.device(args.device)

    print(f"Loading ESMC-300M on {args.device} …", flush=True)
    t0 = time.perf_counter()
    compile_flag = (not args.no_compile) and args.device == "neuron"

    if compile_flag and args.compile_strategy == "regional":
        # Regional compile: compile transformer blocks individually.
        from esm.pretrained import ESMC_300M_202412
        from esm_neuron.utils.neuron import compile_esm_model

        model = ESMC_300M_202412("cpu", use_flash_attn=False)
        model = model.to(torch.bfloat16).to(args.device).eval()
        blocks = getattr(getattr(model, "transformer", None), "blocks", None)
        if blocks is None:
            raise RuntimeError("Regional compile requires model.transformer.blocks")
        for block in blocks:
            block.forward = torch.compile(block.forward, backend=args.device,
                                          dynamic=False, fullgraph=False)
    else:
        model = NeuronESMC("esmc_300m", device=args.device, compile=compile_flag)

    model_load_s = time.perf_counter() - t0

    # Warmup (triggers compilation on Neuron).
    print("Warming up …", flush=True)
    t_warm = time.perf_counter()
    warmup_seqs = df["window"].head(max(args.warmup, 1)).tolist()
    wt = _tokenize_batch(tokenizer, warmup_seqs, args.seqlen, device, args.batch_size)
    wi = {"sequence_tokens": wt, "sequence_id": wt != _PAD}
    with torch.no_grad():
        for _ in range(args.warmup):
            model(**wi)
    _sync(args.device)
    warmup_s = time.perf_counter() - t_warm
    print(f"  warmup done in {warmup_s:.1f}s", flush=True)

    # Evaluation loop.
    rows: list[dict] = []
    latencies: list[float] = []
    t_eval = time.perf_counter()
    n_batches = (len(df) + args.batch_size - 1) // args.batch_size

    for bi, start in enumerate(range(0, len(df), args.batch_size), 1):
        batch = df.iloc[start : start + args.batch_size]
        real = len(batch)
        tokens = _tokenize_batch(tokenizer, batch["window"].tolist(),
                                 args.seqlen, device, args.batch_size)
        inputs = {"sequence_tokens": tokens, "sequence_id": tokens != _PAD}

        _sync(args.device)
        t_b = time.perf_counter()
        with torch.no_grad():
            out = model(**inputs)
            lp = F.log_softmax(out.sequence_logits[:real].float(), dim=-1).cpu()
        _sync(args.device)
        latencies.append(time.perf_counter() - t_b)

        for i, (_, row) in enumerate(batch.iterrows()):
            pos = int(row["window_pos"])
            wt_lp = float(lp[i, pos, aa_ids[row["AA_orig"]]])
            mt_lp = float(lp[i, pos, aa_ids[row["AA_targ"]]])
            score = mt_lp - wt_lp
            rows.append({
                "uniprot": row["uniprot"],
                "position": int(row["position"]),
                "aa_orig": row["AA_orig"],
                "aa_targ": row["AA_targ"],
                "label": int(row["label"]),
                "mutation_score": score,
                "pathogenic_score": -score,
                "prediction": int(score < args.threshold),
            })

        if bi == 1 or bi % args.log_every == 0:
            elapsed = time.perf_counter() - t_eval
            rate = min(start + real, len(df)) / max(elapsed, 1e-9)
            print(f"  progress={min(start+real,len(df))}/{len(df)} "
                  f"batches={bi}/{n_batches} "
                  f"rate={rate:.1f} samples/s", flush=True)

    eval_wall_s = time.perf_counter() - t_eval

    labels = [r["label"] for r in rows]
    scores = [r["pathogenic_score"] for r in rows]
    lat_mean, lat_p50, lat_p95 = _summarise(latencies)
    two_class = len(set(labels)) == 2

    summary: dict[str, Any] = {
        "model": "esmc_300m",
        "task": "d2deep",
        "device": args.device,
        "compile": compile_flag,
        "compile_strategy": args.compile_strategy if compile_flag else None,
        "batch_size": args.batch_size,
        "seqlen": args.seqlen,
        "samples": len(rows),
        "model_load_s": model_load_s,
        "warmup_s": warmup_s,
        "eval_wall_s": eval_wall_s,
        "batch_latency_s_mean": lat_mean,
        "batch_latency_s_p50": lat_p50,
        "batch_latency_s_p95": lat_p95,
        "throughput_samples_per_s": len(rows) / max(eval_wall_s, 1e-9),
        "throughput_tokens_per_s": len(rows) * args.seqlen / max(eval_wall_s, 1e-9),
        "roc_auc": roc_auc_score(labels, scores) if two_class else None,
        "average_precision": average_precision_score(labels, scores) if two_class else None,
        "label_counts": {str(k): int(v)
                         for k, v in pd.Series(labels).value_counts().sort_index().items()},
        "threshold": args.threshold,
        "estimated_cost_usd": (args.hourly_cost * eval_wall_s / 3600.0
                               if args.hourly_cost else None),
    }

    # Print summary.
    print(f"\n{'='*50}")
    print(f"throughput_samples_per_s  = {summary['throughput_samples_per_s']:.2f}")
    print(f"batch_latency_s_p50       = {summary['batch_latency_s_p50']*1000:.1f} ms")
    print(f"roc_auc                   = {summary['roc_auc']}")
    print(f"eval_wall_s               = {summary['eval_wall_s']:.1f}")
    if summary["estimated_cost_usd"]:
        print(f"estimated_cost_usd        = ${summary['estimated_cost_usd']:.4f}")

    # Write outputs.
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"scores  → {args.output_csv}")

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"summary → {args.summary_json}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ESMC-300M D2Deep pathogenicity benchmark")
    p.add_argument("--device", choices=["neuron", "cuda", "cpu"], default="neuron")
    p.add_argument("--d2deep-csv", type=Path, required=True,
                   help="Path to D2Deep compiled_data.csv")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--no-compile", action="store_true",
                   help="Skip torch.compile (always skipped on CUDA/CPU)")
    p.add_argument("--compile-strategy", choices=["full", "regional"], default="regional",
                   help="'regional' compiles transformer blocks individually (recommended). "
                        "'full' compiles the entire model graph.")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Score threshold for binary prediction (default: 0.0)")
    p.add_argument("--hourly-cost", type=float, default=0.0,
                   help="Instance hourly cost in USD for cost estimation")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit evaluation to first N variants (for smoke testing)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--summary-json", type=Path, default=None)
    return p


if __name__ == "__main__":
    args = _parser().parse_args()
    run(args)
