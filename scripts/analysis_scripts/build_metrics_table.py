#!/usr/bin/env python3
"""
Recursively find results_metrics_*.json files anywhere under a directory and
build a comparison table (F1 / EM / R@1 / R@3 / R@5, etc.) across whatever
checkpoints/folders you point it at.

No assumptions about folder naming (cr4, cr16, ..., or anything else) -- it
just walks the tree, finds every matching JSON file, and by default labels
each row with its path starting right after the "checkpoints" folder (e.g.
"stage2/cr4", "stage2_kg/cr16", "stage2_kg_sweep_full/hops2_damp0.5") so two
runs that happen to share a folder name (cr4 under stage2/ vs cr4 under
stage2_kg/) never collide or get silently merged, and the label stays the
same regardless of exactly where --root points. See --anchor to change which
folder name it anchors on, and --label_from_parts / --label_regex if you
want a shorter or custom label instead.

Each results_metrics_*.json file is expected to look like:

    {
        "nq":        {"f1": ..., "em": ..., "recall@1": ..., "recall@3": ..., "recall@5": ..., ...},
        "hotpotqa":  {...},
        "musique":   {...},
        "2wiki":     {...}
    }

NOTE on "frames": evaluate.py historically routed the "frames" dataset
through calculate_basic_metrics (F1/EM/acc only), unlike every other dataset,
which goes through calculate_stage2_metrics (F1/EM/acc + recall@k/precision@k).
As a result, existing results_metrics_*.json files show NaN for frames'
recall@1/3/5, even though the sibling per-example {dataset}_{stage}_{gold}_{k}.jsonl
file already contains everything needed to compute it (topk_idx + pos_index
per example, same index space, confirmed empirically). Rather than requiring
a re-run of evaluate.py on the cluster, this script recomputes recall@k /
precision@k for "frames" directly from that sibling .jsonl whenever the
metrics file is missing them -- see recompute_frames_recall() below. This is
the ONLY change from the original script; every other code path is
unchanged.

Usage
-----
# Point it anywhere under checkpoints/ -- labels always start right after
# "checkpoints" regardless of where --root itself points, e.g. "stage2_kg/cr4"
# even when --root is .../checkpoints/stage2_kg (not just "cr4")
python3 build_metrics_table.py --root /vol/biomedic3/vd425/clara-main/checkpoints/stage2_kg

# Search the whole checkpoints tree -- labels become "stage2/cr4",
# "stage2_kg/cr16", "stage2_kg_sweep_full/hops2_damp0.5", "CLaRa-7B-E2E", etc.
python3 build_metrics_table.py --root /vol/biomedic3/vd425/clara-main/checkpoints \
    --out_csv all_checkpoints_metrics.csv

# Your folder isn't literally named "checkpoints"? Point --anchor at whatever
# it's actually called, e.g. --anchor runs
python3 build_metrics_table.py --root /some/other/path/runs --anchor runs

# Only match a specific metrics filename pattern
python3 build_metrics_table.py --root ./checkpoints --pattern "results_metrics_stage2_False_5.json"

# Per-dataset breakdown instead of averaging across datasets
python3 build_metrics_table.py --root ./checkpoints --per_dataset

# Show percentages instead of fractions
python3 build_metrics_table.py --root ./checkpoints --percent

# Want a SHORTER label instead of the full path? Use --label_from_parts to
# cap how many trailing path components are kept, e.g. 1 -> just "cr4"
# (this brings back the old collision risk between same-named folders, so
# only use it when you know the run names are unique on their own)
python3 build_metrics_table.py --root ./checkpoints --label_from_parts 1

# Disable the frames recall@k recomputation and show it as NaN like before
python3 build_metrics_table.py --root ./checkpoints --no_frames_recall_fix
"""


import argparse
import json
import re
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("This script requires pandas. Install with: pip install pandas", file=sys.stderr)
    raise

METRIC_COLS = ["f1", "em", "recall@1", "recall@3", "recall@5"]
K_VALUES = [1, 3, 5]


def natural_sort_key(label: str):
    """
    Sort labels containing numbers naturally (cr4 < cr16 < cr32 ...) instead
    of alphabetically (cr128 < cr16 < cr32 ...). Falls back to plain string
    sort for labels with no digits.
    """
    parts = [p for p in re.split(r"(\d+)", label) if p != ""]
    # Wrap each part as (0, int) or (1, str) so tuples of different lengths/types
    # remain comparable to each other (e.g. "cr4" vs "cr64/seed_0").
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in parts)


def anchored_parts(json_path: Path, root: Path, anchor: str):
    """
    Return the parent-folder path components to use for labeling, anchored
    right after the last folder named `anchor` anywhere in the full path --
    not relative to --root. This keeps labels stable regardless of exactly
    where --root points: whether --root is .../checkpoints or
    .../checkpoints/stage2_kg, a cr4 run under stage2_kg/ always labels as
    "stage2_kg/cr4", never just "cr4".

    Falls back to the path relative to --root (old behavior) if `anchor`
    doesn't appear anywhere in the path.
    """
    all_parts = json_path.parts[:-1]  # drop the filename itself
    if anchor:
        # search from the right so the LAST occurrence of the anchor name wins,
        # in case it appears more than once in the path
        for i in range(len(all_parts) - 1, -1, -1):
            if all_parts[i] == anchor:
                return all_parts[i + 1:]
    # anchor not found (or not set) -- fall back to relative-to-root
    return json_path.relative_to(root).parts[:-1]


def make_label(json_path: Path, root: Path, label_from_parts: int, label_regex: str, anchor: str):
    """
    Decide what to call this run/checkpoint in the table.

    Priority:
      1. --label_regex, if given and it matches: first capture group matched
         against the full path is used as the label.
      2. --label_from_parts N, if given (N > 0): join only the last N path
         components before the filename, e.g. N=1 -> "cr4" instead of the
         full "stage2_kg/cr4". Useful for shortening labels when you already
         know they're unique on their own.
      3. Default (label_from_parts not given, i.e. 0): use the path starting
         right after the last folder named `anchor` (default "checkpoints")
         anywhere in the full path, e.g. "stage2/cr4", "stage2_kg/cr16",
         "stage2_kg_sweep_full/hops2_damp0.5" -- so the exact location under
         checkpoints/ is always visible in the table, stays the same no
         matter where --root points, and nothing collides.
    """
    parent_parts = anchored_parts(json_path, root, anchor)

    if label_regex:
        match = re.search(label_regex, str(json_path))
        if match:
            return match.group(1) if match.groups() else match.group(0)
        # fall through to default if no match

    if label_from_parts and label_from_parts > 0:
        chosen = parent_parts[-label_from_parts:] if len(parent_parts) >= label_from_parts else parent_parts
        if chosen:
            return "/".join(chosen)

    if parent_parts:
        return "/".join(parent_parts)

    # JSON file sits directly in root (or directly in anchor/) with no
    # parent folder left to name it after
    return json_path.stem


def find_metrics_files(root: Path, pattern: str):
    return sorted(root.rglob(pattern))


def load_metrics(json_path: Path):
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Could not read {json_path}: {e}", file=sys.stderr)
        return None


def recall_missing(metrics: dict) -> bool:
    """True if any recall@k key is absent, None, or NaN (e.g. json.load turns
    a literal NaN into float('nan'), which is what a stale frames entry has)."""
    for k in K_VALUES:
        v = metrics.get(f"recall@{k}")
        if v is None or (isinstance(v, float) and v != v):  # v != v is the NaN check
            return True
    return False


def find_sibling_jsonl(metrics_json_path: Path, dataset: str):
    """
    The per-example results file sits next to results_metrics_{stage}_{gold}_{k}.json
    as {dataset}_{stage}_{gold}_{k}.jsonl (evaluate.py line ~897/937). Reuse the
    "{stage}_{gold}_{k}" suffix from the metrics filename so this works regardless
    of which stage/gold_retrieval/top_k combination produced it.
    """
    stem = metrics_json_path.stem  # "results_metrics_stage2_False_5"
    prefix = "results_metrics_"
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix):]  # "stage2_False_5"
    candidate = metrics_json_path.parent / f"{dataset}_{suffix}.jsonl"
    return candidate if candidate.exists() else None


def recompute_frames_recall(jsonl_path: Path, k_values=K_VALUES):
    """
    Recomputes recall@k / precision@k the same way evaluate.py's
    calculate_stage2_metrics does (topk_idx vs. pos_index overlap), from the
    saved per-example jsonl. Only recall@k/precision@k are recomputed here;
    f1/em/acc/avg_output_length are left as whatever the original
    results_metrics_*.json already reported for this dataset.
    """
    recall = {k: 0 for k in k_values}
    precision = {k: 0 for k in k_values}
    n = 0

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "topk_idx" not in item or "pos_index" not in item:
                return None  # can't recompute, sibling file doesn't have what's needed
            scores = item["topk_idx"]
            pos_index = set(item["pos_index"])
            for k in k_values:
                top_k = set(scores[:k])
                hit = len(top_k & pos_index)
                recall[k] += hit / len(pos_index) if len(pos_index) > 0 else 0
                precision[k] += hit / k
            n += 1

    if n == 0:
        return None

    result = {}
    for k in k_values:
        result[f"recall@{k}"] = recall[k] / n
        result[f"precision@{k}"] = precision[k] / n
    return result


def build_rows(root: Path, pattern: str, datasets, per_dataset: bool, label_from_parts: int,
               label_regex: str, anchor: str, fix_frames_recall: bool):
    files = find_metrics_files(root, pattern)
    if not files:
        return [], []

    rows = []
    used_labels = []

    for fpath in files:
        data = load_metrics(fpath)
        if data is None:
            continue

        label = make_label(fpath, root, label_from_parts, label_regex, anchor)
        used_labels.append((label, fpath))

        present_datasets = datasets if datasets else sorted(data.keys())
        available = [d for d in present_datasets if d in data and isinstance(data[d], dict)]
        missing = [d for d in present_datasets if d not in data]
        if datasets and missing:
            print(f"[WARN] {label} ({fpath}): missing datasets {missing}", file=sys.stderr)

        if not available:
            print(f"[WARN] {label} ({fpath}): no usable dataset entries found, skipping", file=sys.stderr)
            continue

        # Always calculate and populate recall/precision for "frames" by
        # recomputing from the sibling .jsonl when the metrics file has it
        # missing/NaN, since evaluate.py never wrote it for this dataset.
        if fix_frames_recall and "frames" in available and recall_missing(data["frames"]):
            sibling = find_sibling_jsonl(fpath, "frames")
            if sibling is not None:
                recomputed = recompute_frames_recall(sibling)
                if recomputed is not None:
                    data["frames"] = {**data["frames"], **recomputed}
                else:
                    print(f"[WARN] {label}: found {sibling.name} but it's missing "
                          f"topk_idx/pos_index, couldn't recompute frames recall", file=sys.stderr)
            else:
                print(f"[WARN] {label}: frames recall@k missing and no sibling "
                      f"frames_*.jsonl found next to {fpath.name} to recompute from", file=sys.stderr)

        if per_dataset:
            for ds in available:
                metrics = data[ds]
                row = {"run": label, "dataset": ds, "source_file": str(fpath)}
                for m in METRIC_COLS:
                    row[m] = metrics.get(m, float("nan"))
                rows.append(row)
        else:
            row = {"run": label, "n_datasets": len(available), "source_file": str(fpath)}
            for m in METRIC_COLS:
                vals = [data[ds].get(m) for ds in available if data[ds].get(m) is not None]
                row[m] = sum(vals) / len(vals) if vals else float("nan")
            rows.append(row)

    return rows, used_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Top-level directory to search recursively for metrics files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="results_metrics_stage2_*.json",
        help="Glob pattern (relative, searched recursively) to match metrics files "
        "(default: results_metrics_stage2_*.json). Use 'results_metrics_*.json' to match any stage.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Restrict to these dataset keys (e.g. nq hotpotqa musique 2wiki). "
        "Default: use whatever dataset keys are present in each file.",
    )
    parser.add_argument(
        "--per_dataset",
        action="store_true",
        help="Show one row per (run, dataset) instead of averaging across datasets",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="checkpoints",
        help="Folder name to anchor labels on: the label starts right after the last "
        "folder with this name anywhere in each file's path, e.g. with the default "
        "'checkpoints', a file at .../checkpoints/stage2_kg/cr4/results_metrics_....json "
        "labels as 'stage2_kg/cr4' no matter what --root points at. If this folder name "
        "never appears in the path, falls back to the path relative to --root. "
        "Set to '' to disable anchoring and always use the path relative to --root.",
    )
    parser.add_argument(
        "--label_from_parts",
        type=int,
        default=0,
        help="Cap the run label to the last N path components before the filename, "
        "e.g. 1 -> 'cr4' instead of the full 'stage2_kg/cr4'. "
        "Default: 0, which means use the full anchored path (recommended, "
        "avoids collisions between e.g. stage2/cr4 and stage2_kg/cr4).",
    )
    parser.add_argument(
        "--label_regex",
        type=str,
        default=None,
        help="Regex applied to the full file path to extract the run label from its first "
        "capture group, e.g. '(cr\\d+)' to pull just 'cr64' out of anywhere in the path. "
        "Overrides both the full-path default and --label_from_parts when it matches.",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Multiply metrics by 100 for display/export (e.g. 0.4123 -> 41.23)",
    )
    parser.add_argument(
        "--sort",
        choices=["natural", "alpha", "none"],
        default="natural",
        help="How to sort rows by run label (default: natural, so cr4 < cr16 < cr32...)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional path to save the table as CSV",
    )
    parser.add_argument(
        "--hide_source",
        action="store_true",
        help="Don't show the source_file column in the printed table (still included in CSV)",
    )
    parser.add_argument(
        "--no_frames_recall_fix",
        action="store_true",
        help="Disable recomputing frames recall@k/precision@k from the sibling .jsonl; "
        "show NaN for frames recall as originally written by evaluate.py instead.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] --root does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    rows, used_labels = build_rows(
        root,
        args.pattern,
        args.datasets,
        args.per_dataset,
        args.label_from_parts,
        args.label_regex,
        args.anchor,
        fix_frames_recall=not args.no_frames_recall_fix,
    )

    if not rows:
        print(
            f"[ERROR] No metrics files matching pattern '{args.pattern}' found under {root}.\n"
            f"        Try --pattern 'results_metrics_*.json' to widen the search, "
            f"or check the path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Warn about duplicate labels (e.g. two different runs both resolving to the same
    # label -- shouldn't normally happen with the full-path default, but can still occur
    # with --label_from_parts or --label_regex set to something too short)
    label_counts = {}
    for label, fpath in used_labels:
        label_counts.setdefault(label, []).append(fpath)
    dupes = {l: paths for l, paths in label_counts.items() if len(paths) > 1}
    if dupes:
        print("[WARN] Multiple files resolved to the same run label — consider dropping "
              "--label_from_parts / --label_regex to use the full path instead:", file=sys.stderr)
        for label, paths in dupes.items():
            print(f"    '{label}':", file=sys.stderr)
            for p in paths:
                print(f"        {p}", file=sys.stderr)

    df = pd.DataFrame(rows)

    if args.percent:
        for m in METRIC_COLS:
            if m in df.columns:
                df[m] = df[m] * 100

    # Sort (done with plain Python sorted() + reindex rather than df.sort_values,
    # since natural-sort keys are heterogeneous tuples that pandas' lexsort can't compare)
    if args.sort in ("natural", "alpha"):
        if args.sort == "natural":
            key_fn = lambda i: (natural_sort_key(df.at[i, "run"]), df.at[i, "dataset"] if args.per_dataset else "")
        else:
            key_fn = lambda i: (df.at[i, "run"], df.at[i, "dataset"] if args.per_dataset else "")
        new_order = sorted(df.index, key=key_fn)
        df = df.loc[new_order]
    df = df.reset_index(drop=True)

    # Column order
    id_cols = ["run"] + (["dataset"] if args.per_dataset else ["n_datasets"])
    metric_cols_present = [m for m in METRIC_COLS if m in df.columns]
    df = df[id_cols + metric_cols_present + ["source_file"]]

    display_df = df.drop(columns=["source_file"]) if args.hide_source else df
    for m in metric_cols_present:
        display_df[m] = display_df[m].round(4 if not args.percent else 2)

    mode_desc = "per-dataset" if args.per_dataset else "averaged across datasets"
    print(f"\nFound {len(used_labels)} metrics file(s) under {root}\nMode: {mode_desc}\n")
    try:
        print(display_df.to_string(index=False))
    except Exception:
        print(display_df)

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved CSV to: {out_path}")


if __name__ == "__main__":
    main()