#!/usr/bin/env python3
"""
summarize_kg_sweep.py

Walks a stage2 KG sweep directory (e.g. checkpoints/stage2_kg_sweep), finds each
hopsX_dampY subfolder's results_metrics_*.json file, and prints a comparison
table of f1 / em (and optionally recall@k / precision@k) per dataset per config.

Usage:
    python summarize_kg_sweep.py [sweep_dir] [--metrics-file NAME] [--full] [--sort-by KEY] [--csv OUT.csv]

Examples:
    # Run from inside stage2_kg_sweep/, defaults to current dir
    python summarize_kg_sweep.py

    # Explicit path
    python summarize_kg_sweep.py /vol/biomedic3/vd425/clara-main/checkpoints/stage2_kg_sweep

    # Include recall@k / precision@k columns too
    python summarize_kg_sweep.py --full

    # Sort rows by hotpotqa f1 descending (default sorts by config name)
    python summarize_kg_sweep.py --sort-by hotpotqa_f1

    # Also dump a CSV
    python summarize_kg_sweep.py --csv sweep_results.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

DEFAULT_METRICS_FILENAME_PATTERN = "results_metrics_stage2_False_5.json"

CORE_METRICS = ["f1", "em"]
FULL_METRICS = ["f1", "em", "acc", "recall@1", "recall@3", "recall@5",
                "precision@1", "precision@3", "precision@5"]

CONFIG_DIR_RE = re.compile(r"^hops(\d+)_damp([\d.]+)$")


def find_config_dirs(sweep_dir: Path):
    """Return sorted list of hopsX_dampY subdirectories."""
    dirs = [d for d in sweep_dir.iterdir() if d.is_dir() and CONFIG_DIR_RE.match(d.name)]

    def sort_key(d):
        m = CONFIG_DIR_RE.match(d.name)
        hops, damp = int(m.group(1)), float(m.group(2))
        return (hops, damp)

    return sorted(dirs, key=sort_key)


def load_metrics(config_dir: Path, metrics_filename: str):
    """Load the metrics JSON for a given config dir. Returns dict or None."""
    metrics_path = config_dir / metrics_filename
    if not metrics_path.exists():
        # Fall back: look for any results_metrics_*.json in the folder
        candidates = sorted(config_dir.glob("results_metrics_*.json"))
        if not candidates:
            return None, None
        metrics_path = candidates[0]
    try:
        with open(metrics_path, "r") as f:
            return json.load(f), metrics_path.name
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [warn] failed to read {metrics_path}: {e}", file=sys.stderr)
        return None, None


def collect_rows(sweep_dir: Path, metrics_filename: str, full: bool):
    """
    Returns:
        rows: list of dicts, one per config, with keys like 'config', 'hops',
              'damp', and '<dataset>_<metric>' for every dataset/metric found.
        datasets: sorted list of dataset names encountered.
        metrics: list of metric names used (core or full).
    """
    config_dirs = find_config_dirs(sweep_dir)
    if not config_dirs:
        print(f"No hopsX_dampY folders found in {sweep_dir}", file=sys.stderr)
        return [], [], []

    metrics_wanted = FULL_METRICS if full else CORE_METRICS

    rows = []
    datasets_seen = set()

    for cdir in config_dirs:
        m = CONFIG_DIR_RE.match(cdir.name)
        hops, damp = m.group(1), m.group(2)

        data, used_filename = load_metrics(cdir, metrics_filename)
        row = {"config": cdir.name, "hops": hops, "damp": damp}

        if data is None:
            print(f"  [warn] no metrics file found in {cdir}", file=sys.stderr)
            rows.append(row)
            continue

        for dataset, dmetrics in data.items():
            datasets_seen.add(dataset)
            for metric in metrics_wanted:
                key = f"{dataset}_{metric}"
                val = dmetrics.get(metric)
                row[key] = val

        rows.append(row)

    return rows, sorted(datasets_seen), metrics_wanted


def format_val(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:12f}"
    return str(v)


def print_table(rows, datasets, metrics):
    if not rows:
        return

    # Build header
    headers = ["config"]
    for ds in datasets:
        for m in metrics:
            headers.append(f"{ds}_{m}")

    # Compute column widths
    col_widths = {h: len(h) for h in headers}
    str_rows = []
    for row in rows:
        str_row = {"config": row["config"]}
        for ds in datasets:
            for m in metrics:
                key = f"{ds}_{m}"
                str_row[key] = format_val(row.get(key))
        str_rows.append(str_row)
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str_row[h]))

    # Print header
    def fmt_row(vals):
        return "  ".join(str(v).ljust(col_widths[h]) for h, v in zip(headers, vals))

    print(fmt_row(headers))
    print("  ".join("-" * col_widths[h] for h in headers))
    for str_row in str_rows:
        print(fmt_row([str_row[h] for h in headers]))


def write_csv(rows, datasets, metrics, out_path):
    headers = ["config"]
    for ds in datasets:
        for m in metrics:
            headers.append(f"{ds}_{m}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {"config": row["config"]}
            for ds in datasets:
                for m in metrics:
                    key = f"{ds}_{m}"
                    out[key] = row.get(key)
            writer.writerow(out)
    print(f"\nCSV written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize stage2 KG sweep metrics into a table.")
    parser.add_argument("sweep_dir", nargs="?", default=".",
                         help="Path to the sweep directory containing hopsX_dampY folders (default: current dir)")
    parser.add_argument("--metrics-file", default=DEFAULT_METRICS_FILENAME_PATTERN,
                         help=f"Metrics filename to look for in each config folder (default: {DEFAULT_METRICS_FILENAME_PATTERN})")
    parser.add_argument("--full", action="store_true",
                         help="Include recall@k / precision@k / acc columns, not just f1/em")
    parser.add_argument("--sort-by", default=None,
                         help="Column to sort rows by, e.g. hotpotqa_f1 (descending). Default: sort by hops then damp.")
    parser.add_argument("--csv", default=None, help="Optional path to also write results as CSV")

    args = parser.parse_args()
    sweep_dir = Path(args.sweep_dir).resolve()

    if not sweep_dir.is_dir():
        print(f"Error: {sweep_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    rows, datasets, metrics = collect_rows(sweep_dir, args.metrics_file, args.full)
    if not rows:
        sys.exit(1)

    if args.sort_by:
        rows.sort(key=lambda r: (r.get(args.sort_by) is None, -(r.get(args.sort_by) or 0)))

    print(f"Sweep dir: {sweep_dir}")
    print(f"Datasets found: {', '.join(datasets)}")
    print()
    print_table(rows, datasets, metrics)

    if args.csv:
        write_csv(rows, datasets, metrics, args.csv)


if __name__ == "__main__":
    main()