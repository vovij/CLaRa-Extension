#!/usr/bin/env python3
"""
Run this directly inside checkpoints/stage2_kg_sweep/ to summarize all variant results.

Usage:
    cd /vol/biomedic3/vd425/clara-main/checkpoints/stage2_kg_sweep
    python3 check_results.py
    python3 check_results.py --metric recall@1
    python3 check_results.py --vanilla /vol/biomedic3/vd425/clara-main/checkpoints/stage2/cr4
    python3 check_results.py --kg-baseline /vol/biomedic3/vd425/clara-main/checkpoints/stage2_kg/cr4
"""

import json
import os
import re
import argparse

DATASETS = ["nq", "hotpotqa", "musique", "2wiki"]
DEFAULT_METRICS = ["f1", "em", "recall@1", "recall@3", "recall@5"]


def find_results_file(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = [f for f in os.listdir(ckpt_dir) if f.startswith("results_metrics_") and f.endswith(".json")]
    if not candidates:
        return None
    preferred = [c for c in candidates if "stage2_False_5" in c]
    chosen = preferred[0] if preferred else candidates[0]
    return os.path.join(ckpt_dir, chosen)


def load_metrics(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def format_cell(data, ds, metrics):
    if data is None or ds not in data:
        return "N/A"
    parts = []
    for m in metrics:
        val = data[ds].get(m)
        parts.append(f"{m}={val:.3f}" if val is not None else f"{m}=?")
    return " ".join(parts)


def sort_key(name):
    """Sort sweep variant names numerically by hops then damping, not alphabetically."""
    m = re.match(r"hops(\d+)_damp([\d.]+)", name)
    if m:
        return (int(m.group(1)), float(m.group(2)))
    return (999, 999.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default=None, help="Show only this metric (e.g. recall@1). Default: f1, em, recall@1.")
    parser.add_argument("--vanilla", default="/vol/biomedic3/vd425/clara-main/checkpoints/stage2/cr4")
    parser.add_argument("--kg-baseline", default="/vol/biomedic3/vd425/clara-main/checkpoints/stage2_kg/cr4")
    args = parser.parse_args()

    metrics = [args.metric] if args.metric else DEFAULT_METRICS

    runs = {}
    runs["vanilla (no KG)"] = args.vanilla
    runs["KG baseline (hops=2 damp=0.5, full eval)"] = args.kg_baseline

    cwd = os.getcwd()
    variant_names = sorted(
        [d for d in os.listdir(cwd) if os.path.isdir(d) and re.match(r"hops\d+_damp[\d.]+", d)],
        key=sort_key
    )
    for v in variant_names:
        runs[f"sweep {v}"] = os.path.join(cwd, v)

    col_width = 13 + 8 * len(metrics)
    header = f"{'Run':<40}" + "".join(f"{ds:<{col_width}}" for ds in DATASETS)
    print(header)
    print("-" * len(header))

    for run_name, ckpt_dir in runs.items():
        f = find_results_file(ckpt_dir)
        data = load_metrics(f)
        row = f"{run_name:<40}"
        for ds in DATASETS:
            row += f"{format_cell(data, ds, metrics):<{col_width}}"
        print(row)


if __name__ == "__main__":
    main()