"""
Latent space analysis for CLaRa compressor models (any training stage).

Covers the two action items:
  1. Visualise latent space    -> t-SNE / PCA plots of memory-token reps vs.
                                   document-token reps, faceted by compression
                                   ratio (CR).
  2. Orthogonal vectors per CR -> effective rank (via SVD) and a thresholded
                                   pairwise-orthogonality count of the memory
                                   token vectors, per CR.

Usage
-----
  # Stage 1 (pretraining baseline) — default behaviour:
    python latent_space_analysis.py \
        --base_dir /vol/biomedic3/vd425/clara-main \
        --stage_label stage1 \
        --crs 4 16 32 64 128 256 \
        --dataset musique \
        --n_docs 200

  # Stage 1_2 (instruction-tuned):
    python latent_space_analysis.py \
        --stage_label stage1_2 \
        --crs 4 16 32 64 128 256

  # Any arbitrary checkpoint tree (must contain {cr}):
    python latent_space_analysis.py \
        --stage_label my_experiment \
        --ckpt_root /path/to/checkpoints/cr{cr} \
        --out_dir   ./results/my_experiment

Configurable paths / labels (all optional)
------------------------------------------
  --stage_label   Human-readable tag used in plot titles and output filenames.
                  Also drives the default --ckpt_root and --out_dir when those
                  are not set explicitly.  Default: "stage1".
  --ckpt_root     Full template for per-CR checkpoint directories; must contain
                  the literal string {cr}.
                  Default: <base_dir>/checkpoints/<stage_label>/cr{cr}
  --out_dir       Directory for all outputs (plots, JSON, .npz caches).
                  Default: ./latent_space_results_<stage_label>

Notes
-----
- This script loads ONE checkpoint at a time, extracts embeddings, frees the
  model, then moves to the next CR. This keeps peak GPU memory to a single
  7B model rather than 6 of them.
- Embeddings are cached to disk per-CR so the (expensive) extraction step
  only needs to run once. Re-running the script reuses the cache unless
  --recompute is passed.
"""

import argparse
import gc
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_documents(base_dir: str, dataset: str, n_docs: int, seed: int = 42) -> List[str]:
    """
    Pull a sample of raw document strings from the existing eval parquet files
    so embeddings are computed on real, consistent data across all CRs.

    Falls back to the example jsonl shipped with ml-clara if the parquet
    eval data isn't found.
    """
    random.seed(seed)
    candidates = glob.glob(
        os.path.join(base_dir, "data", f"e2e_eval_{dataset}_with_pos", "*.parquet")
    )
    docs: List[str] = []

    if candidates:
        import pandas as pd

        df = pd.read_parquet(candidates[0])
        # Each row's "docs" column is expected to be a list of document
        # strings (top-k retrieved docs) per the evaluate.py DataLoader.
        for row_docs in df["docs"]:
            row_docs = list(row_docs)[:5]
            for d in row_docs:
                if isinstance(d, str) and len(d.strip()) > 0:
                    docs.append(d.strip())
            if len(docs) >= n_docs * 3:  # collect extra, then sample down
                break
    else:
        example_path = os.path.join(base_dir, "ml-clara", "example", "end_to_end_data.jsonl")
        with open(example_path, "r") as f:
            for line in f:
                item = json.loads(line)
                for d in item.get("docs", item.get("documents", []))[:5]:
                    if isinstance(d, str) and len(d.strip()) > 0:
                        docs.append(d.strip())

    if not docs:
        raise RuntimeError(
            f"No documents found for dataset='{dataset}'. Checked {candidates} "
            "and the ml-clara/example fallback."
        )

    # De-duplicate, then sample a fixed-size, reproducible subset.
    docs = list(dict.fromkeys(docs))
    if len(docs) > n_docs:
        docs = random.sample(docs, n_docs)
    return docs


# --------------------------------------------------------------------------
# Embedding extraction
# --------------------------------------------------------------------------

@dataclass
class CREmbeddings:
    cr: int
    num_mem_tokens: int
    mem_reps: np.ndarray       # [n_docs, l, hidden_dim]
    mem_reps_pooled: np.ndarray  # [n_docs, hidden_dim]  (mean over l)
    doc_reps_pooled: np.ndarray  # [n_docs, hidden_dim]  (mean-pooled doc tokens)


def extract_embeddings_for_cr(
    ckpt_path: str,
    documents: List[str],
    batch_size: int = 8,
    device: str = "cuda",
) -> CREmbeddings:
    """
    Load one CLaRa checkpoint, run `compress_documents` to get memory-token
    representations, and also grab mean-pooled raw document hidden states
    (final layer, before compression) for the visualisation comparison.
    """
    from transformers import AutoModel

    print(f"[load] {ckpt_path}")
    model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True).to(device)
    model.eval()

    cr = int(model.config.compr_rate)
    doc_max_length = int(getattr(model, "doc_max_length", model.config.doc_max_length))
    num_mem_tokens = doc_max_length // cr

    all_mem = []
    all_doc_pooled = []

    with torch.no_grad():
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            # Memory-token (compressed) representations via the model's own API.
            compressed_embs = model.compress_documents(batch)  # [B, l, H]
            if isinstance(compressed_embs, tuple):  # some paths also return mse_loss
                compressed_embs = compressed_embs[0]
            all_mem.append(compressed_embs.float().cpu().numpy())

            # Raw document-token representations for comparison: re-encode the
            # documents through the same encoder adapter and mean-pool the
            # non-memory positions. We reuse the model's internal encoder-input
            # builder so tokenisation/special tokens match exactly what
            # compress_documents() used.
            enc_inputs = model._prepare_encoder_inputs(batch, max_length=model.doc_max_length)
            enc_input_ids = enc_inputs["input_ids"].to(device)
            enc_attention_mask = enc_inputs["attention_mask"].to(device)

            if "encoder_adapter" in model.adapter_keys:
                model.decoder.set_adapter("encoder_adapter")
            hidden = model.decoder(
                input_ids=enc_input_ids,
                attention_mask=enc_attention_mask,
                output_hidden_states=True,
            ).hidden_states[-1]

            mem_token_mask = torch.isin(
                enc_input_ids, model.decoder_tokenizer.mem_token_ids_pt.to(device)
            )
            non_mem_mask = (~mem_token_mask) & enc_attention_mask.bool()
            non_mem_len = non_mem_mask.sum(dim=1).clamp(min=1)
            doc_pooled = (hidden * non_mem_mask.unsqueeze(-1)).sum(dim=1) / non_mem_len.unsqueeze(-1)
            all_doc_pooled.append(doc_pooled.float().cpu().numpy())

            # Reactivate full adapter set so subsequent compress_documents()
            # calls behave the same as the first one.
            if len(model.adapter_keys) > 0:
                model.decoder.set_adapter(model.adapter_keys)

    mem_reps = np.concatenate(all_mem, axis=0)            # [n_docs, l, H]
    doc_reps_pooled = np.concatenate(all_doc_pooled, axis=0)  # [n_docs, H]
    mem_reps_pooled = mem_reps.mean(axis=1)                # [n_docs, H]

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return CREmbeddings(
        cr=cr,
        num_mem_tokens=num_mem_tokens,
        mem_reps=mem_reps,
        mem_reps_pooled=mem_reps_pooled,
        doc_reps_pooled=doc_reps_pooled,
    )


# --------------------------------------------------------------------------
# (1) Visualisation
# --------------------------------------------------------------------------

def plot_tsne_per_cr(results: Dict[int, CREmbeddings], out_dir: str, perplexity: int = 30, stage_label: str = "stage1") -> None:
    """One t-SNE panel per CR: memory-token reps (pooled) vs. document reps."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    crs = sorted(results.keys())
    n = len(crs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for ax, cr in zip(axes[0], crs):
        res = results[cr]
        all_data = np.vstack([res.mem_reps_pooled, res.doc_reps_pooled])

        if all_data.shape[1] > 50:
            all_data = PCA(n_components=50, random_state=42).fit_transform(all_data)

        eff_perplexity = min(perplexity, max(5, len(all_data) // 4))
        proj = TSNE(
            n_components=2, random_state=42, perplexity=eff_perplexity, init="pca"
        ).fit_transform(all_data)

        n_mem = len(res.mem_reps_pooled)
        ax.scatter(
            proj[n_mem:, 0], proj[n_mem:, 1],
            c="#0066CC", alpha=0.6, s=20, label="Document reps",
        )
        ax.scatter(
            proj[:n_mem, 0], proj[:n_mem, 1],
            c="#FF3333", alpha=0.6, s=20, label="Memory tokens (pooled)",
        )
        ax.set_title(f"CR = {cr}x  (l = {res.num_mem_tokens})")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0][0].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Latent space: memory-token reps vs. document reps, {stage_label} model", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"tsne_by_cr_{stage_label}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_tsne_combined_across_cr(results: Dict[int, CREmbeddings], out_dir: str, perplexity: int = 30, stage_label: str = "stage1") -> None:
    """
    Single t-SNE with all CRs' memory-token reps overlaid (one colour per CR),
    to see how compression ratio reshapes the latent manifold relative to
    each other (not just vs. doc reps).
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    crs = sorted(results.keys())
    all_vecs = []
    labels = []
    for cr in crs:
        all_vecs.append(results[cr].mem_reps_pooled)
        labels += [cr] * len(results[cr].mem_reps_pooled)

    all_vecs = np.vstack(all_vecs)
    labels = np.array(labels)

    if all_vecs.shape[1] > 50:
        all_vecs = PCA(n_components=50, random_state=42).fit_transform(all_vecs)

    eff_perplexity = min(perplexity, max(5, len(all_vecs) // 4))
    proj = TSNE(n_components=2, random_state=42, perplexity=eff_perplexity, init="pca").fit_transform(all_vecs)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("viridis", len(crs))
    for idx, cr in enumerate(crs):
        mask = labels == cr
        ax.scatter(proj[mask, 0], proj[mask, 1], s=18, alpha=0.7, color=cmap(idx), label=f"CR {cr}x")

    ax.set_title(f"Memory-token latent space across compression ratios ({stage_label})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"tsne_across_cr_{stage_label}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# --------------------------------------------------------------------------
# (2) Orthogonal vector counting / effective rank
# --------------------------------------------------------------------------

def orthogonality_stats_for_cr(
    mem_reps: np.ndarray, cos_threshold: float = 0.1
) -> Dict[str, float]:
    """
    mem_reps: [n_docs, l, hidden_dim] memory-token vectors.

    Two complementary measures of "how many orthogonal vectors" the latent
    code uses:

    A) Per-document pairwise orthogonality: within each document's l memory
       tokens, what fraction of pairs are near-orthogonal (|cos sim| < thr)?
       Averaged across documents. Tells you whether individual memory slots
       within one document specialise into distinct directions.

    B) Effective rank (participation ratio) of the *set* of all memory
       vectors stacked together: effective_rank = (sum(s_i^2))^2 / sum(s_i^4)
       where s_i are singular values. This estimates how many independent
       (orthogonal) directions the whole pool of vectors actually spans,
       on a continuous 1..min(n,d) scale rather than a hard count.

    C) A hard count: number of singular values needed to explain 95% of
       variance (an alternate, more standard "effective dimensionality").
    """
    n_docs, l, hidden_dim = mem_reps.shape

    # --- (A) within-document pairwise orthogonality ---
    if l > 1:
        frac_orthogonal_per_doc = []
        for doc_idx in range(n_docs):
            v = mem_reps[doc_idx]  # [l, hidden_dim]
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
            gram = v @ v.T
            iu = np.triu_indices(l, k=1)
            off_diag = gram[iu]
            frac_orthogonal_per_doc.append(np.mean(np.abs(off_diag) < cos_threshold))
        within_doc_frac_orthogonal = float(np.mean(frac_orthogonal_per_doc))
        within_doc_n_pairs = l * (l - 1) // 2
    else:
        within_doc_frac_orthogonal = float("nan")  # only 1 memory token: no pairs
        within_doc_n_pairs = 0

    # --- (B) effective rank across the whole pool of memory vectors ---
    flat = mem_reps.reshape(n_docs * l, hidden_dim)
    flat_centered = flat - flat.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(flat_centered, compute_uv=False)
    sv_sq = singular_values ** 2
    effective_rank = float((sv_sq.sum() ** 2) / (sv_sq ** 2).sum())

    # --- (C) hard count: # components for 95% explained variance ---
    explained = np.cumsum(sv_sq) / sv_sq.sum()
    n_components_95 = int(np.searchsorted(explained, 0.95) + 1)

    return {
        "n_docs": n_docs,
        "num_mem_tokens_per_doc": l,
        "hidden_dim": hidden_dim,
        "within_doc_pairs_checked": within_doc_n_pairs,
        "within_doc_frac_orthogonal": within_doc_frac_orthogonal,
        "effective_rank_participation_ratio": effective_rank,
        "n_components_for_95pct_variance": n_components_95,
        "max_possible_rank": min(n_docs * l, hidden_dim),
    }


def run_orthogonality_analysis(
    results: Dict[int, CREmbeddings], out_dir: str, cos_threshold: float = 0.1, stage_label: str = "stage1"
) -> None:
    table = []
    for cr in sorted(results.keys()):
        stats = orthogonality_stats_for_cr(results[cr].mem_reps, cos_threshold=cos_threshold)
        stats["cr"] = cr
        table.append(stats)
        print(
            f"CR={cr:>4} | l={stats['num_mem_tokens_per_doc']:>3} | "
            f"eff_rank={stats['effective_rank_participation_ratio']:.2f} | "
            f"95%-var dims={stats['n_components_for_95pct_variance']:>3} | "
            f"within-doc frac orthogonal (|cos|<{cos_threshold})="
            f"{stats['within_doc_frac_orthogonal']:.3f}"
        )

    out_json = os.path.join(out_dir, f"orthogonality_stats_{stage_label}.json")
    with open(out_json, "w") as f:
        json.dump(table, f, indent=2)
    print(f"[saved] {out_json}")

    # Also a quick bar plot of effective rank vs CR, since that's the
    # single number most likely to go straight into the report.
    import matplotlib.pyplot as plt

    crs = [t["cr"] for t in table]
    eff_ranks = [t["effective_rank_participation_ratio"] for t in table]
    n95 = [t["n_components_for_95pct_variance"] for t in table]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.plot(crs, eff_ranks, "o-", color="#0066CC", label="Effective rank (participation ratio)")
    ax2.plot(crs, n95, "s--", color="#CC6600", label="# dims for 95% variance")

    ax1.set_xlabel("Compression ratio (CR)")
    ax1.set_ylabel("Effective rank", color="#0066CC")
    ax2.set_ylabel("# components (95% variance)", color="#CC6600")
    ax1.set_xscale("log", base=2)
    ax1.set_title(f"Effective dimensionality of memory-token latent space vs. CR ({stage_label})")
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"effective_rank_vs_cr_{stage_label}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# --------------------------------------------------------------------------
# Caching helpers
# --------------------------------------------------------------------------

def cache_path(out_dir: str, cr: int) -> str:
    return os.path.join(out_dir, f"embeddings_cr{cr}.npz")


def save_cache(out_dir: str, emb: CREmbeddings) -> None:
    np.savez_compressed(
        cache_path(out_dir, emb.cr),
        cr=emb.cr,
        num_mem_tokens=emb.num_mem_tokens,
        mem_reps=emb.mem_reps,
        mem_reps_pooled=emb.mem_reps_pooled,
        doc_reps_pooled=emb.doc_reps_pooled,
    )


def load_cache(out_dir: str, cr: int) -> CREmbeddings:
    data = np.load(cache_path(out_dir, cr))
    return CREmbeddings(
        cr=int(data["cr"]),
        num_mem_tokens=int(data["num_mem_tokens"]),
        mem_reps=data["mem_reps"],
        mem_reps_pooled=data["mem_reps_pooled"],
        doc_reps_pooled=data["doc_reps_pooled"],
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CLaRa latent space analysis")
    parser.add_argument("--base_dir", type=str, default="/vol/biomedic3/vd425/clara-main")
    parser.add_argument(
        "--stage_label", type=str, default="stage1",
        help="Human-readable tag for the training stage (e.g. stage1, stage1_2, e2e). "
             "Used in plot titles, output filenames, and as the default checkpoint "
             "sub-directory and output directory when those are not set explicitly.",
    )
    parser.add_argument(
        "--ckpt_root", type=str, default=None,
        help="Checkpoint root template; must contain '{cr}'. "
             "Default: <base_dir>/checkpoints/<stage_label>/cr{cr}",
    )
    parser.add_argument("--crs", type=int, nargs="+", default=[4, 16, 32, 64, 128, 256])
    parser.add_argument("--dataset", type=str, default="musique")
    parser.add_argument("--n_docs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cos_threshold", type=float, default=0.1)
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory for plots, JSON, and embedding caches. "
             "Default: ./latent_space_results_<stage_label>",
    )
    parser.add_argument("--recompute", action="store_true", help="Ignore cached embeddings.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Derive defaults from stage_label when not explicitly provided.
    ckpt_template = args.ckpt_root or os.path.join(
        args.base_dir, "checkpoints", args.stage_label, "cr{cr}"
    )
    out_dir = args.out_dir or f"./latent_space_results_{args.stage_label}"

    os.makedirs(out_dir, exist_ok=True)

    documents = load_documents(args.base_dir, args.dataset, args.n_docs)
    print(f"[data] loaded {len(documents)} documents from dataset='{args.dataset}'")

    results: Dict[int, CREmbeddings] = {}
    for cr in args.crs:
        if not args.recompute and os.path.exists(cache_path(out_dir, cr)):
            print(f"[cache] loading cached embeddings for CR={cr}")
            results[cr] = load_cache(out_dir, cr)
            continue

        ckpt_path = ckpt_template.format(cr=cr)
        if not os.path.exists(ckpt_path):
            print(f"[skip] checkpoint not found for CR={cr}: {ckpt_path}")
            continue

        emb = extract_embeddings_for_cr(
            ckpt_path, documents, batch_size=args.batch_size, device=args.device
        )
        save_cache(out_dir, emb)
        results[cr] = emb

    if not results:
        raise RuntimeError("No checkpoints were successfully processed. Check --ckpt_root / --crs.")

    # (1) Visualisation
    plot_tsne_per_cr(results, out_dir, stage_label=args.stage_label)
    plot_tsne_combined_across_cr(results, out_dir, stage_label=args.stage_label)

    # (2) Orthogonal vector counts
    run_orthogonality_analysis(results, out_dir, cos_threshold=args.cos_threshold, stage_label=args.stage_label)

    print(f"\nDone. All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()