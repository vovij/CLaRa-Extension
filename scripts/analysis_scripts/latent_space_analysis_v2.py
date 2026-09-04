"""
Latent space analysis for CLaRa — v2 (manifold-preserving, non-aggregated)
==========================================================================

Supervisor action items addressed
----------------------------------
1.  Compression plots WITHOUT aggregation
    Every individual memory-token vector  h_{m_j}  (j = 1..l) is plotted as
    its own point, coloured by slot index j.  This preserves the full manifold
    geometry rather than collapsing l vectors per document into one centroid.

    Mathematical framing
    ~~~~~~~~~~~~~~~~~~~~
    For document d_i, the compressor produces l hidden states:
        M_i = {h_{m_1}, ..., h_{m_l}} ⊂ R^H,   |M_i| = l = ceil(tokens / CR)
    We embed ALL n_docs × l vectors jointly under a projection φ: R^H → R^2.
    Crucially φ is fitted once on the full set so the inter-slot and
    inter-document structure is preserved; no per-document averaging.

    What it reveals
    ~~~~~~~~~~~~~~~
    • If slot j always occupies the same neighbourhood regardless of document
      content, the compressor has learned positional specialisation (each slot
      captures a different "aspect" of the document).
    • If slots scatter freely relative to document token reps, the model has
      learned content-driven compression rather than fixed-position summaries.
    • Comparing low-CR vs high-CR panels shows how the manifold contracts as
      the bottleneck tightens.

2.  Retrieval model (query reasoner θ_qr) plots WITHOUT aggregation
    Query vectors q_i (also l memory tokens, same shape as document M_i) are
    embedded in the same projection space as the document vectors.  Cosine
    similarity s_i = cos(q_i, M_i) drives retrieval (Eq. 3.1 in CLaRa paper).

    Mathematical framing
    ~~~~~~~~~~~~~~~~~~~~
    A well-trained query reasoner should satisfy:
        φ(q_i)  ≈ φ(M_i^+)  for the positive document M_i^+
    while remaining separated from non-relevant document clusters.
    Plotting both without pooling shows whether query and document manifolds
    share the same intrinsic geometry — a necessary condition for cosine
    similarity to be a meaningful retrieval signal.

3.  Projection choices (manifold-preserving, non-linear where needed)
    • PCA  — linear, globally distance-preserving; used as pre-whitening and
              as a standalone plot. Preserves variance structure exactly.
    • UMAP — non-linear, preserves local AND global topology better than
              t-SNE; recommended for large N.  Run on PCA-50 whitened data.
    • t-SNE — preserves local neighbourhood only; included for comparison.

    All projections share the SAME fit (fitted on the union of all point sets
    for a given panel) so relative positions are meaningful.

Usage
-----
  # Compression-only analysis (compressor model, stage1):
    python latent_space_analysis_v2.py \
        --base_dir /vol/biomedic3/vd425/clara-main \
        --stage_label stage1 \
        --mode compression \
        --crs 4 16 32 64 128 256 \
        --dataset musique --n_docs 200

  # Retrieval analysis (e2e checkpoint — has both compressor + query reasoner):
    python latent_space_analysis_v2.py \
        --stage_label e2e \
        --mode retrieval \
        --ckpt_root /path/to/e2e/cr{cr} \
        --crs 4 32 128

  # Both together:
    python latent_space_analysis_v2.py \
        --stage_label e2e --mode both --crs 4 16 32 64 128 256

CLI flags
---------
  --mode          compression | retrieval | both   (default: compression)
  --projection    pca | umap | tsne | all          (default: all)
  --n_slot_sample Max memory-token slots to plot per CR (avoids overplotting).
                  Default: 3000 total points (sampled uniformly across docs).
  --recompute     Ignore embedding cache and recompute from model.
  [all other flags same as v1]
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# Data structures
# ============================================================

@dataclass
class CREmbeddings:
    """
    All embeddings for one compression ratio CR.

    Shapes
    ------
    mem_reps       : [n_docs, l, H]  — raw per-slot memory vectors (NOT pooled)
    doc_reps_pooled: [n_docs, H]     — mean-pooled document token hidden states
    query_reps     : [n_queries, l, H] | None  — per-slot query vectors from θ_qr
    """
    cr: int
    num_mem_tokens: int
    mem_reps: np.ndarray                        # [n, l, H]
    doc_reps_pooled: np.ndarray                 # [n, H]
    query_reps: Optional[np.ndarray] = None    # [n_q, l, H] or None


# ============================================================
# Data loading
# ============================================================

def load_documents(base_dir: str, dataset: str, n_docs: int, seed: int = 42) -> List[str]:
    """
    Load a fixed random sample of documents from the eval parquets.
    Uses the same path pattern as latent_space_analysis.py (v1):
        <base_dir>/data/e2e_eval_{dataset}_with_pos/*.parquet
    Each row's 'docs' column is a list of retrieved document strings.
    Falls back to the ml-clara example jsonl if parquets are not found.
    """
    random.seed(seed)

    candidates = glob.glob(
        os.path.join(base_dir, "data", f"e2e_eval_{dataset}_with_pos", "*.parquet")
    )
    docs: List[str] = []

    if candidates:
        import pandas as pd
        df = pd.read_parquet(candidates[0])
        for row_docs in df["docs"]:
            for d in list(row_docs)[:5]:
                if isinstance(d, str) and len(d.strip()) > 0:
                    docs.append(d.strip())
            if len(docs) >= n_docs * 3:
                break
    else:
        example_path = os.path.join(base_dir, "ml-clara", "example", "end_to_end_data.jsonl")
        if os.path.exists(example_path):
            with open(example_path) as f:
                for line in f:
                    item = json.loads(line)
                    for d in item.get("docs", item.get("documents", []))[:5]:
                        if isinstance(d, str) and len(d.strip()) > 0:
                            docs.append(d.strip())

    if not docs:
        raise RuntimeError(
            f"No documents found for dataset='{dataset}'. "
            f"Checked: {base_dir}/data/e2e_eval_{dataset}_with_pos/*.parquet\n"
            "Make sure --base_dir points to the clara-main root."
        )

    docs = list(dict.fromkeys(docs))  # deduplicate, preserve order
    if len(docs) > n_docs:
        docs = random.sample(docs, n_docs)
    return docs


def load_queries(base_dir: str, dataset: str, n_queries: int, seed: int = 42) -> List[str]:
    """
    Load queries from the same eval parquets (column: 'query' or 'question').
    Same path pattern as load_documents.
    """
    random.seed(seed + 1)

    candidates = glob.glob(
        os.path.join(base_dir, "data", f"e2e_eval_{dataset}_with_pos", "*.parquet")
    )
    queries: List[str] = []

    if candidates:
        import pandas as pd
        df = pd.read_parquet(candidates[0])
        col = next((c for c in ("query", "question") if c in df.columns), None)
        if col:
            queries = df[col].dropna().tolist()

    if not queries:
        return []

    if len(queries) > n_queries:
        queries = random.sample(queries, n_queries)
    return queries


# ============================================================
# Embedding extraction — compressor (θ_c)
# ============================================================

def extract_compression_embeddings(
    ckpt_dir: str,
    documents: List[str],
    cr: int,
    device: str = "cuda",
    batch_size: int = 8,
) -> CREmbeddings:
    """
    Load the compressor checkpoint and extract:
      • mem_reps       [n, l, H] — ALL l memory-token hidden states (unflattened)
      • doc_reps_pooled [n, H]   — mean of document token hidden states

    No pooling is applied to mem_reps here; that is left to the plotting
    functions so they can choose whether to aggregate or not.
    """
    from transformers import AutoTokenizer
    # CLaRa uses a custom ClaraModel; fall back gracefully for dev environments
    try:
        from clara.models.compressor import ClaraCompressor as Model
    except ImportError:
        try:
            from ml_clara.models.compressor import ClaraCompressor as Model
        except ImportError:
            raise ImportError(
                "Cannot import ClaraCompressor. Make sure ml-clara is on PYTHONPATH."
            )

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = Model.from_pretrained(ckpt_dir, trust_remote_code=True)
    model.to(device).eval()

    # Infer num_mem_tokens from config or model attribute
    num_mem_tokens: int = getattr(model.config, "num_mem_tokens", cr)

    all_mem: List[np.ndarray] = []
    all_doc_pooled: List[np.ndarray] = []

    for batch_start in range(0, len(documents), batch_size):
        batch = documents[batch_start: batch_start + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(device)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        # Memory token hidden states — shape [B, l, H]
        mem_hidden = out.memory_hidden_states          # attribute from ClaraCompressor
        # Document token hidden states — shape [B, T, H]; pool over T
        doc_hidden = out.hidden_states[-1][:, : enc.input_ids.shape[1], :]
        doc_pooled = doc_hidden.mean(dim=1)            # [B, H]

        all_mem.append(mem_hidden.cpu().float().numpy())
        all_doc_pooled.append(doc_pooled.cpu().float().numpy())

    del model
    gc.collect()
    torch.cuda.empty_cache()

    mem_reps = np.concatenate(all_mem, axis=0)          # [n, l, H]
    doc_reps_pooled = np.concatenate(all_doc_pooled, axis=0)  # [n, H]

    return CREmbeddings(
        cr=cr,
        num_mem_tokens=num_mem_tokens,
        mem_reps=mem_reps,
        doc_reps_pooled=doc_reps_pooled,
    )


# ============================================================
# Embedding extraction — query reasoner (θ_qr)
# ============================================================

def extract_query_embeddings(
    ckpt_dir: str,
    queries: List[str],
    device: str = "cuda",
    batch_size: int = 8,
) -> np.ndarray:
    """
    Load the query-reasoner checkpoint (same architecture as compressor,
    different LoRA weights) and return query memory-token vectors.

    Returns
    -------
    query_reps : [n_queries, l, H]
    """
    from transformers import AutoTokenizer
    try:
        from clara.models.query_reasoner import ClaraQueryReasoner as QModel
    except ImportError:
        try:
            from ml_clara.models.query_reasoner import ClaraQueryReasoner as QModel
        except ImportError:
            raise ImportError(
                "Cannot import ClaraQueryReasoner. Needed for --mode retrieval."
            )

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = QModel.from_pretrained(ckpt_dir, trust_remote_code=True)
    model.to(device).eval()

    all_q: List[np.ndarray] = []

    for batch_start in range(0, len(queries), batch_size):
        batch = queries[batch_start: batch_start + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        all_q.append(out.memory_hidden_states.cpu().float().numpy())

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return np.concatenate(all_q, axis=0)   # [n_queries, l, H]


# ============================================================
# Projection helpers
# ============================================================

def pca_whiten(X: np.ndarray, n_components: int = 50) -> np.ndarray:
    """PCA whitening to n_components. Used as pre-processing for UMAP/t-SNE."""
    from sklearn.decomposition import PCA
    if X.shape[1] <= n_components:
        return X
    return PCA(n_components=n_components, random_state=42).fit_transform(X)


def project_2d(
    X: np.ndarray,
    method: str,
    perplexity: int = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Project high-dimensional vectors to 2D.

    Parameters
    ----------
    X       : [N, D] input (should already be PCA-whitened for t-SNE/UMAP)
    method  : 'pca' | 'tsne' | 'umap'

    Notes on manifold preservation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    PCA  — linear projection maximising variance. Globally distance-preserving
           but cannot capture non-linear structure.
    UMAP — optimises a fuzzy topological representation; preserves both local
           neighbourhood AND global cluster relationships.  Faster than t-SNE
           and scales to large N.  Recommended for ≥ 5 000 points.
    t-SNE — minimises KL divergence between pairwise similarity distributions.
            Excellent local neighbourhood preservation; distorts global distances.
    """
    if method == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2, random_state=random_state).fit_transform(X)
    elif method == "umap":
        try:
            import umap  # pip install umap-learn
        except ImportError:
            raise ImportError(
                "umap-learn is required for --projection umap. "
                "Install with: pip install umap-learn"
            )
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",  # matches CLaRa retrieval cosine similarity
            random_state=random_state,
        )
        proj = reducer.fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        eff_perp = min(perplexity, max(5, X.shape[0] // 4))
        proj = TSNE(
            n_components=2,
            perplexity=eff_perp,
            init="pca",
            random_state=random_state,
        ).fit_transform(X)
    else:
        raise ValueError(f"Unknown projection method: {method}")
    return proj


# ============================================================
# Sampling helper  (avoid overplotting with tens-of-thousands of points)
# ============================================================

def subsample_slots(
    mem_reps: np.ndarray, max_total: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten [n_docs, l, H] → [n_docs*l, H] then subsample to max_total.

    Returns
    -------
    vecs      : [N, H]  subsampled vectors
    doc_ids   : [N]     which document each vector came from
    slot_ids  : [N]     which memory-slot index (0..l-1)
    """
    n, l, H = mem_reps.shape
    vecs = mem_reps.reshape(n * l, H)
    doc_ids = np.repeat(np.arange(n), l)
    slot_ids = np.tile(np.arange(l), n)

    if len(vecs) > max_total:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(vecs), size=max_total, replace=False)
        vecs, doc_ids, slot_ids = vecs[idx], doc_ids[idx], slot_ids[idx]

    return vecs, doc_ids, slot_ids


# ============================================================
# Plot 1 — Compression model: individual memory slots (no pooling)
# ============================================================

def plot_compression_no_aggregate(
    results: Dict[int, CREmbeddings],
    out_dir: str,
    projection: str = "umap",
    max_pts: int = 3000,
    stage_label: str = "stage1",
) -> None:
    """
    Plot every individual memory-token vector and document-token vector
    in the same 2-D projection space WITHOUT any aggregation/pooling.

    Design
    ------
    Each panel = one CR.
    Points:
      • Blue gradient (slot 0 → l-1) : individual memory-token vectors h_{m_j}
      • Red           : mean-pooled document token reps (one per doc)
    Colour by slot index reveals whether the compressor learns positional
    specialisation (each slot consistently captures different semantics).

    Mathematical framing
    --------------------
    Let Φ = {h_{m_j}^{(i)} : i=1..n, j=1..l} ∪ {ĥ_d^{(i)} : i=1..n}
    where  ĥ_d^{(i)} = (1/|d_i|) Σ_{t∈d_i} h_t   is the document rep.
    We apply φ: R^H → R^2 to all of Φ simultaneously (shared projection),
    then colour memory slots by j and document reps in red.

    What it says about the model
    ----------------------------
    • Tight per-slot clusters (same j, different i) → positional specialisation
    • Memory slots overlapping document reps → high semantic fidelity (MSE
      alignment objective L_MSE is being respected at the individual slot level)
    • Clear separation between CR panels → compression ratio reshapes the
      intrinsic manifold (a sign of non-trivial information routing)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    crs = sorted(results.keys())
    n_cr = len(crs)
    fig, axes = plt.subplots(1, n_cr, figsize=(5 * n_cr, 5), squeeze=False)

    for ax, cr in zip(axes[0], crs):
        res = results[cr]
        l = res.num_mem_tokens

        # --- Flatten memory slots: [n*l, H] ---
        mem_vecs, _, slot_ids = subsample_slots(res.mem_reps, max_pts, seed=42)
        doc_vecs = res.doc_reps_pooled   # [n, H]

        # Limit doc points to same order of magnitude
        n_doc_show = min(len(doc_vecs), max_pts // 4)
        rng = np.random.default_rng(0)
        doc_idx = rng.choice(len(doc_vecs), n_doc_show, replace=False)
        doc_vecs_sub = doc_vecs[doc_idx]

        all_data = np.vstack([mem_vecs, doc_vecs_sub])

        # PCA whiten before non-linear projection
        all_w = pca_whiten(all_data, n_components=50)
        proj = project_2d(all_w, method=projection)

        n_mem = len(mem_vecs)
        proj_mem = proj[:n_mem]
        proj_doc = proj[n_mem:]

        # Colour memory slots by slot index
        cmap = cm.get_cmap("plasma", l)
        slot_colours = cmap(slot_ids / max(l - 1, 1))

        ax.scatter(
            proj_doc[:, 0], proj_doc[:, 1],
            c="#CC2222", alpha=0.5, s=16, label="Doc reps (pooled)", zorder=2,
        )
        sc = ax.scatter(
            proj_mem[:, 0], proj_mem[:, 1],
            c=slot_ids, cmap="plasma", vmin=0, vmax=l - 1,
            alpha=0.55, s=12, label="Memory slots", zorder=3,
        )
        ax.set_title(f"CR={cr}x   (l={l} slots)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        # Mini colourbar for slot index
        plt.colorbar(sc, ax=ax, label="slot index j", fraction=0.046, pad=0.04)

    axes[0][0].legend(loc="upper left", fontsize=8, markerscale=1.5)
    fig.suptitle(
        f"Compression model — individual memory slots vs. document reps\n"
        f"Projection: {projection.upper()}  |  stage: {stage_label}  "
        f"(no pooling, n_pts≤{max_pts} per CR)",
        y=1.03, fontsize=11,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"compression_slots_{projection}_{stage_label}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ============================================================
# Plot 2 — Compression: slots across all CRs on one canvas
# ============================================================

def plot_compression_across_cr(
    results: Dict[int, CREmbeddings],
    out_dir: str,
    projection: str = "umap",
    max_pts_total: int = 6000,
    stage_label: str = "stage1",
) -> None:
    """
    Single 2-D projection of ALL CRs' memory-token vectors, coloured by CR.
    Fits ONE global projection on the union so CR manifolds are directly
    comparable in the same coordinate frame.

    Mathematical framing
    --------------------
    Let Φ_k = {h_{m_j}^{(i)} : i=1..n, j=1..l_k} for each CR k.
    We project Φ = ∪_k Φ_k jointly under φ: R^H → R^2.
    Separation of Φ_k clusters indicates that the compressor maps to
    different regions of R^H for different bottleneck sizes — evidence
    that the model genuinely adapts its representation per CR rather than
    learning a single representation and truncating.

    What it says about the model
    ----------------------------
    • Overlapping clusters → CRs share similar feature directions; the model
      may be over-compressing early ratios (wasteful capacity).
    • Progressive separation along a gradient → ordered latent geometry
      consistent with hierarchical compression.
    """
    import matplotlib.pyplot as plt

    crs = sorted(results.keys())
    pts_per_cr = max_pts_total // len(crs)

    all_vecs, all_cr_labels = [], []
    for cr in crs:
        mem_v, _, _ = subsample_slots(results[cr].mem_reps, pts_per_cr, seed=cr)
        all_vecs.append(mem_v)
        all_cr_labels.extend([cr] * len(mem_v))

    all_vecs = np.vstack(all_vecs)
    all_cr_labels = np.array(all_cr_labels)

    all_w = pca_whiten(all_vecs, 50)
    proj = project_2d(all_w, method=projection)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("viridis", len(crs))
    for idx, cr in enumerate(crs):
        mask = all_cr_labels == cr
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            s=10, alpha=0.6, color=cmap(idx), label=f"CR {cr}x",
        )

    ax.set_title(
        f"Memory-slot manifold across CRs — {projection.upper()}\n"
        f"stage: {stage_label}  (no aggregation)",
        fontsize=11,
    )
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, markerscale=2)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"compression_across_cr_{projection}_{stage_label}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ============================================================
# Plot 3 — Retrieval model: query vs. document slot vectors
# ============================================================

def plot_retrieval_no_aggregate(
    results: Dict[int, CREmbeddings],
    out_dir: str,
    projection: str = "umap",
    max_pts: int = 3000,
    stage_label: str = "stage1",
) -> None:
    """
    Plot individual query-reasoner memory-token vectors against individual
    document compressor memory-token vectors — WITHOUT pooling either.

    Mathematical framing
    --------------------
    Retrieval score: s_i = cos(q_i, M_i)  where both q_i ∈ R^{l×H} and
    M_i ∈ R^{l×H}.  In CLaRa the cosine is computed on the FLATTENED
    concatenation q ∈ R^{lH}, but the per-slot view here lets us ask:

      "Do individual query slots align with corresponding document slots?"

    If  cos(q_{(j)}, M_{(j)}) > cos(q_{(j)}, M_{(k≠j)})  on average,
    slots are positionally aligned (slot j in the query encodes the same
    semantic 'aspect' as slot j in the document).

    A joint 2-D projection of {q_{m_j}} ∪ {h_{m_j}} reveals:
      • Whether query and document manifolds are co-located (necessary for
        cosine similarity to work as retrieval signal).
      • Whether query slots 'reach into' the positive-document cluster or
        scatter broadly (the latter = poor retrieval alignment).

    What it says about the model
    ----------------------------
    • Query slots clustered inside document manifold → strong retrieval
      alignment; θ_qr has learned to anticipate document content.
    • Query slots forming a separate cloud → retrieval relies on embedding
      direction alone; harder for fine-grained multi-hop retrieval.
    • Per-slot analysis helps diagnose whether certain slots are 'wasted'
      (consistently near-zero cosine with all document slots).
    """
    import matplotlib.pyplot as plt

    crs = sorted(cr for cr, res in results.items() if res.query_reps is not None)
    if not crs:
        print("[skip] No query_reps found — run with --mode retrieval or both.")
        return

    n_cr = len(crs)
    fig, axes = plt.subplots(1, n_cr, figsize=(5 * n_cr, 5), squeeze=False)

    for ax, cr in zip(axes[0], crs):
        res = results[cr]
        l = res.num_mem_tokens

        doc_vecs, _, doc_slot_ids = subsample_slots(res.mem_reps, max_pts, seed=42)
        q_vecs, _, q_slot_ids = subsample_slots(res.query_reps, max_pts // 2, seed=99)

        all_data = np.vstack([doc_vecs, q_vecs])
        all_w = pca_whiten(all_data, 50)
        proj = project_2d(all_w, method=projection)

        n_doc = len(doc_vecs)
        proj_doc = proj[:n_doc]
        proj_q = proj[n_doc:]

        ax.scatter(
            proj_doc[:, 0], proj_doc[:, 1],
            c=doc_slot_ids, cmap="Blues", vmin=0, vmax=l - 1,
            alpha=0.4, s=12, label="Doc memory slots",
        )
        ax.scatter(
            proj_q[:, 0], proj_q[:, 1],
            c=q_slot_ids, cmap="Oranges", vmin=0, vmax=l - 1,
            alpha=0.7, s=18, marker="^", label="Query memory slots (θ_qr)",
        )
        ax.set_title(f"CR={cr}x   (l={l} slots)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    axes[0][0].legend(loc="upper left", fontsize=8, markerscale=1.5)
    fig.suptitle(
        f"Retrieval model — query θ_qr vs. document compressor slots\n"
        f"Projection: {projection.upper()}  |  stage: {stage_label}  (no pooling)",
        y=1.03, fontsize=11,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"retrieval_slots_{projection}_{stage_label}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ============================================================
# Plot 4 — PCA variance explained (linear structure diagnostic)
# ============================================================

def plot_pca_variance(
    results: Dict[int, CREmbeddings],
    out_dir: str,
    stage_label: str = "stage1",
    max_components: int = 64,
) -> None:
    """
    For each CR, plot cumulative explained variance ratio of PCA on the
    flattened (unagregated) memory-slot vectors.

    Mathematical framing
    --------------------
    Given the matrix V ∈ R^{N×H}  (N = n_docs × l, H = hidden dim),
    its singular value decomposition  V = U Σ W^T  gives explained variance:
        EVR_k = σ_k^2 / Σ_j σ_j^2
    Cumulative EVR reveals the intrinsic dimensionality of the compression
    manifold as a function of CR.

    What it says about the model
    ----------------------------
    • A steep EVR curve (90% variance in few PCs) → low intrinsic rank;
      the compressor collapses information onto a narrow linear subspace.
      This is pathological for high CR (over-compression) but expected.
    • Flatter EVR (requires many PCs for 90%) → high intrinsic rank;
      the compressor distributes information across many orthogonal
      directions, exploiting the full capacity of the l × H representation.
    • Comparing CR=4 vs CR=256 should show a rank decrease proportional
      to the reduction in l (since l ∝ 1/CR and rank ≤ min(N, l, H)).
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    crs = sorted(results.keys())
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis", len(crs))

    for idx, cr in enumerate(crs):
        res = results[cr]
        n, l, H = res.mem_reps.shape
        vecs = res.mem_reps.reshape(n * l, H)
        n_comp = min(max_components, vecs.shape[0] - 1, vecs.shape[1])
        pca = PCA(n_components=n_comp, random_state=42).fit(vecs)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(range(1, n_comp + 1), cumvar, color=cmap(idx), label=f"CR {cr}x")

    ax.axhline(0.9, color="grey", linestyle="--", linewidth=0.8, label="90% threshold")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(
        f"PCA explained variance of memory-slot vectors by CR\n"
        f"stage: {stage_label}  (no aggregation — all n×l vectors)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(1, max_components)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"pca_variance_{stage_label}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ============================================================
# Existing orthogonality / effective-rank analysis (kept from v1)
# ============================================================

def orthogonality_stats(
    mem_reps: np.ndarray, cos_threshold: float = 0.1
) -> Dict:
    """
    Intra-document pairwise orthogonality + global effective rank.
    See v1 docstring for full mathematical framing.
    mem_reps : [n, l, H]
    """
    n, l, H = mem_reps.shape
    frac_ortho_list = []
    for i in range(n):
        slots = mem_reps[i]                              # [l, H]
        norms = np.linalg.norm(slots, axis=1, keepdims=True) + 1e-9
        normed = slots / norms
        cos_sim = normed @ normed.T                      # [l, l]
        # count off-diagonal pairs
        pairs = l * (l - 1)
        if pairs == 0:
            frac_ortho_list.append(1.0)
            continue
        near_orth = np.sum(np.abs(cos_sim) < cos_threshold) - l  # exclude diagonal
        frac_ortho_list.append(near_orth / pairs)

    # Effective rank on full flattened set
    vecs = mem_reps.reshape(n * l, H)
    U, s, Vt = np.linalg.svd(vecs, full_matrices=False)
    s2 = s ** 2
    eff_rank = (s2.sum() ** 2) / (s2 ** 2).sum()
    cumvar = np.cumsum(s2) / s2.sum()
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1

    return {
        "mean_intra_doc_ortho_fraction": float(np.mean(frac_ortho_list)),
        "effective_rank_participation_ratio": float(eff_rank),
        "n_components_for_95pct_variance": int(n95),
    }


# ============================================================
# Cache helpers
# ============================================================

def cache_path(out_dir: str, cr: int) -> str:
    return os.path.join(out_dir, f"embeddings_cr{cr}.npz")


def save_cache(out_dir: str, emb: CREmbeddings) -> None:
    arrays = dict(
        cr=emb.cr,
        num_mem_tokens=emb.num_mem_tokens,
        mem_reps=emb.mem_reps,
        doc_reps_pooled=emb.doc_reps_pooled,
    )
    if emb.query_reps is not None:
        arrays["query_reps"] = emb.query_reps
    np.savez_compressed(cache_path(out_dir, emb.cr), **arrays)


def load_cache(out_dir: str, cr: int) -> CREmbeddings:
    data = np.load(cache_path(out_dir, cr))
    return CREmbeddings(
        cr=int(data["cr"]),
        num_mem_tokens=int(data["num_mem_tokens"]),
        mem_reps=data["mem_reps"],
        doc_reps_pooled=data["doc_reps_pooled"],
        query_reps=data["query_reps"] if "query_reps" in data else None,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLaRa latent space analysis v2 — manifold-preserving, no aggregation"
    )
    parser.add_argument("--base_dir", default="/vol/biomedic3/vd425/clara-main")
    parser.add_argument(
        "--stage_label", default="stage1",
        help="Tag for checkpoint sub-directory and output filenames.",
    )
    parser.add_argument(
        "--ckpt_root", default=None,
        help="Template with {cr} for per-CR checkpoint dirs. "
             "Default: <base_dir>/checkpoints/<stage_label>/cr{cr}",
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory. Default: ./latent_space_v2_<stage_label>",
    )
    parser.add_argument("--crs", nargs="+", type=int, default=[4, 16, 32, 64, 128, 256])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_docs", type=int, default=200)
    parser.add_argument("--n_queries", type=int, default=100,
                        help="Number of queries for retrieval mode.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mode", choices=["compression", "retrieval", "both"], default="compression",
        help="Which plots to produce.",
    )
    parser.add_argument(
        "--projection", choices=["pca", "umap", "tsne", "all"], default="all",
        help="Which 2-D projection(s) to use. 'all' produces all three.",
    )
    parser.add_argument(
        "--max_pts", type=int, default=3000,
        help="Max individual slot points to plot per CR panel (avoids overplotting).",
    )
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore cached embeddings and re-extract from model.")
    args = parser.parse_args()

    # Resolve defaults
    if args.ckpt_root is None:
        args.ckpt_root = os.path.join(
            args.base_dir, "checkpoints", args.stage_label, "cr{cr}"
        )
    if args.out_dir is None:
        args.out_dir = f"./latent_space_v2_{args.stage_label}"
    os.makedirs(args.out_dir, exist_ok=True)

    projections = (
        ["pca", "umap", "tsne"] if args.projection == "all"
        else [args.projection]
    )

    # ── Load documents (and queries if needed) ──────────────────────────
    print(f"Loading documents (dataset={args.dataset}, n={args.n_docs}) …")
    documents = load_documents(args.base_dir, args.dataset, args.n_docs)

    queries = []
    if args.mode in ("retrieval", "both"):
        print(f"Loading queries (n={args.n_queries}) …")
        queries = load_queries(args.base_dir, args.dataset, args.n_queries)
        if not queries:
            print("[warn] No queries found — retrieval plots will be skipped.")

    # ── Extract / load cached embeddings ────────────────────────────────
    results: Dict[int, CREmbeddings] = {}
    for cr in args.crs:
        cp = cache_path(args.out_dir, cr)
        if not args.recompute and os.path.exists(cp):
            print(f"  [cache] CR={cr}x — loading from {cp}")
            results[cr] = load_cache(args.out_dir, cr)
        else:
            ckpt_dir = args.ckpt_root.replace("{cr}", str(cr))
            print(f"  [extract] CR={cr}x — {ckpt_dir}")
            emb = extract_compression_embeddings(
                ckpt_dir, documents, cr, args.device, args.batch_size
            )
            if args.mode in ("retrieval", "both") and queries:
                print(f"    → extracting query embeddings …")
                emb.query_reps = extract_query_embeddings(
                    ckpt_dir, queries, args.device, args.batch_size
                )
            save_cache(args.out_dir, emb)
            results[cr] = emb

    # ── Produce plots ────────────────────────────────────────────────────
    print("\nGenerating plots …")

    # PCA variance (linear diagnostic, no projection choice needed)
    plot_pca_variance(results, args.out_dir, args.stage_label)

    for proj in projections:
        if args.mode in ("compression", "both"):
            print(f"  Compression slots — {proj.upper()} …")
            plot_compression_no_aggregate(
                results, args.out_dir, proj, args.max_pts, args.stage_label
            )
            plot_compression_across_cr(
                results, args.out_dir, proj,
                max_pts_total=args.max_pts * len(args.crs),
                stage_label=args.stage_label,
            )

        if args.mode in ("retrieval", "both"):
            print(f"  Retrieval slots — {proj.upper()} …")
            plot_retrieval_no_aggregate(
                results, args.out_dir, proj, args.max_pts, args.stage_label
            )

    # Orthogonality table (kept from v1)
    print("\nComputing orthogonality statistics …")
    table = []
    for cr in sorted(results.keys()):
        stats = orthogonality_stats(results[cr].mem_reps)
        stats["cr"] = cr
        table.append(stats)
        print(
            f"  CR={cr:4d}x  eff_rank={stats['effective_rank_participation_ratio']:.1f}"
            f"  n95={stats['n_components_for_95pct_variance']}"
            f"  intra_ortho={stats['mean_intra_doc_ortho_fraction']:.3f}"
        )

    out_json = os.path.join(args.out_dir, f"orthogonality_{args.stage_label}.json")
    with open(out_json, "w") as fh:
        json.dump(table, fh, indent=2)
    print(f"[saved] {out_json}")
    print(f"\nAll outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()