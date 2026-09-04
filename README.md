## CLaRa Extension: SLURM Pipeline & Temporal Link Matrix

<div align="center">
  <img src="figs/clara_logo.jpg" width="300"/>
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2511.18659"><img src="https://img.shields.io/badge/arXiv-2511.18659-b31b1b.svg" alt="Original CLaRa paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apple-blue" alt="License"></a>
  <a href="https://huggingface.co/apple/CLaRa-7B-E2E"><img src="https://img.shields.io/badge/Hugging%20Face-CLaRa_E2E-FFEB3B" alt="deploy"></a>
  <a href="https://huggingface.co/datasets/apple/CLaRa_multi_stage"><img src="https://img.shields.io/badge/Hugging%20Face-CLaRa_Data-FFEB3B" alt="data"></a>
</div>

This repository is an MSc Individual Project extension of Apple's [**CLaRa**](https://arxiv.org/abs/2511.18659) (Continuous Latent Reasoning), carried out in the Department of Computing at Imperial College London under the supervision of Dr Avinash Kori. The full write-up is in the accompanying dissertation, *"Extending CLaRa for Multi-Hop Reasoning in Retrieval-Augmented Generation"* (Drobot, 2026).

It extends [Apple's original release](https://github.com/apple/ml-clara) in two ways:

1. **A reproducible SLURM training pipeline** for Imperial's BioMedIA cluster: the full three-stage CLaRa pipeline (compressor pretraining → compression instruction tuning → end-to-end joint training), retrained from scratch across six compression rates, CR ∈ {4, 16, 32, 64, 128, 256}, plus evaluation on the held-out [FRAMES](https://arxiv.org/abs/2409.12941) multi-hop benchmark.
2. **The Temporal Link Matrix (TLM)** — a differentiable, query-conditioned link structure over a widened retrieval candidate pool, propagated via personalized PageRank before CLaRa's existing differentiable top-*k* selection. It's designed to give the model an explicit mechanism for chaining evidence across multi-hop questions, something the original single-step cosine retrieval doesn't do.

---

### What's new here vs. the original CLaRa repo

- SLURM job scripts and array configs for Imperial's `gpus`/`gpus24`/`gpus48` partitions, including fixes for a DeepSpeed ZeRO-2 learning-rate scheduler crash, a shell glob-expansion bug that silently truncated full-dataset training, and a ZeRO-3 initialisation hang resolved by switching to ZeRO-2 with CPU Adam offload.
- `prepare_frames.py`, which converts the [FRAMES](https://huggingface.co/datasets/google/frames-benchmark) dataset into CLaRa's expected 20-document input format, padding gold Wikipedia articles with BM25-retrieved distractors.
- A latent-space analysis pipeline (`analysis_scripts/latent_space_analysis*.py`) using t-SNE, UMAP, PCA, and per-document effective rank to inspect how memory-token representations vary with compression rate and training stage.
- `temporal_link_matrix.py` — the TLM module itself, plus its integration into `modeling_clara.py` and the config flags (`use_kg_retrieval`, `kg_rank`, `kg_hops`, `kg_damping`) needed to turn it on.
- Rank and candidate-pool-width ablation scripts, run to test whether the TLM's dataset-dependent behaviour was a capacity artefact of its untuned defaults.

---

### Repository Structure

```
├── scripts/
│   ├── training_scripts/         # Stage 1 / 1.5 / 2 launch scripts (vanilla + TLM), sweeps
│   │   ├── stage1_array.sh
│   │   ├── stage1_2_array.sh
│   │   ├── stage2_array.sh
│   │   ├── stage2_cr4_kg.sh
│   │   ├── stage2_kg_array.sh
│   │   ├── stage2_kg_rank_ablation.sh
│   │   └── sweep_hops_damping_*.sh
│   ├── eval_scripts/             # End-to-end, FRAMES, and ablation evaluation
│   │   ├── eval_e2e_kg_array.sh
│   │   ├── eval_frames_array.sh
│   │   └── eval_e2e_kg_rank_ablation_cr4.sh
│   ├── analysis_scripts/         # Latent space analysis, TLM module, metrics tooling
│   │   ├── latent_space_analysis.py
│   │   ├── latent_space_analysis_v2.py
│   │   ├── temporal_link_matrix.py
│   │   ├── test_temporal_link_matrix.py
│   │   └── build_metrics_table.py
│   └── misc_scripts/             # Downloads, environment setup
│       ├── download_clara_data.sh
│       └── download_models.sh
├── openrlhf/                     # Core training framework (forked from OpenRLHF)
│   ├── models/
│   │   └── modeling_clara.py     # CLaRa model definition + TLM integration
│   ├── datasets/
│   │   └── prepare_frames.py     # FRAMES dataset preparation
│   ├── trainer/
│   └── cli/
├── evaluation/                   # Evaluation framework (F1, EM, recall@k)
├── checkpoints/                  # Stage 1 / 1.5 / 2 checkpoints per compression rate
├── results/                      # Latent space analysis outputs, metrics
└── README.md
```

---

### Three-Stage Training (+ the TLM Extension)

**Stage 1 — Compressor Pretraining**
Trains the compressor adapter via the SCP framework (QA + paraphrase supervision) plus an MSE alignment loss. Run across all six compression rates with:
```bash
sbatch scripts/training_scripts/stage1_array.sh
```

**Stage 1.5 — Compression Instruction Tuning**
Jointly fine-tunes compressor and generator on instruction-following QA:
```bash
sbatch scripts/training_scripts/stage1_2_array.sh
```

**Stage 2 — End-to-End Joint Training**
Introduces the query reasoner and trains it jointly with the generator via the unified language-modelling loss, using CLaRa's straight-through top-*k* estimator:
```bash
sbatch scripts/training_scripts/stage2_array.sh          # vanilla
sbatch scripts/training_scripts/stage2_kg_array.sh       # with the TLM extension
```

**The Temporal Link Matrix**, enabled via `--use_kg_retrieval`, replaces the raw cosine retrieval scores with scores propagated across a learned, query-conditioned link matrix over a widened candidate pool (`stage2_retrieval_top_n=100` by default, generation still uses `top_k=5`). Key flags:

| Flag | Meaning | Default |
|---|---|---|
| `--kg_rank` | Link-matrix projection rank *r* | 64 |
| `--kg_hops` | Propagation steps *T* | 2 |
| `--kg_damping` | Damping factor *a* (0 = unmodified pipeline) | 0.5 |

See `analysis_scripts/temporal_link_matrix.py` for the full implementation and `test_temporal_link_matrix.py` for the correctness/differentiability test suite.

---

### Getting Started

```bash
# Environment (uv, not conda — see the dissertation for the migration rationale)
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements_uv.txt --break-system-packages

export PYTHONPATH=/path/to/clara-extension:$PYTHONPATH
export CUDA_HOME=/vol/cuda/12.9.0   # or your cluster's CUDA 12.x install
```

Download the base model and Apple's released checkpoints:
```bash
bash scripts/misc_scripts/download_models.sh
bash scripts/misc_scripts/download_clara_data.sh
```

Prepare the FRAMES benchmark:
```bash
python openrlhf/datasets/prepare_frames.py
```

Run the full pipeline for a single compression rate (example: CR = 4):
```bash
sbatch scripts/training_scripts/stage1_cr4_mutiple_gpus.sh
sbatch scripts/training_scripts/stage1_2_cr4_mutiple_gpus.sh
sbatch scripts/training_scripts/stage2_cr4_mutiple_gpus.sh      # vanilla
sbatch scripts/training_scripts/stage2_cr4_kg.sh                # + TLM
```

Evaluate:
```bash
bash scripts/eval_scripts/eval_e2e_cr4.sh
bash scripts/eval_scripts/eval_e2e_kg_cr4.sh
bash scripts/eval_scripts/eval_frames_array.sh
```

---

### Key Findings

**Latent space.** Compression ratio, not document content, dominates the structure of CLaRa's latent space — memory-token embeddings from different compression rates form near-disjoint clusters when projected jointly. Effective rank collapses sharply as CR grows: only ~17% of available slot capacity is used at CR = 4, falling toward rank-1 by CR ≈ 64.

**Compression is cheap — except on MuSiQue.** Across a 64× compression range, NQ/HotpotQA/2Wiki lose only 3–4.5 F1 points. MuSiQue barely moves at all (F1 15.9–17.0 across the whole sweep), suggesting its bottleneck is evidence composition, not representation capacity.

**CR = 4, vanilla vs. TLM extension:**

| Dataset | Vanilla F1 / EM | + TLM F1 / EM |
|---|---|---|
| MuSiQue | 16.24 / 7.07 | **18.96 / 10.40** |
| HotpotQA | 43.78 / 32.85 | 44.94 / 34.00 |
| NQ | 48.31 / 38.22 | 47.45 / 37.80 |
| 2Wiki | **44.06 / 39.68** | 41.58 / 36.60 |
| FRAMES | **20.15 / 13.19** | 17.62 / 11.36 |

The TLM improves every MuSiQue metric at CR = 4, but consistently regresses recall on HotpotQA/2Wiki across the compression-rate sweep, and on FRAMES at every CR with no exceptions. Rank and candidate-pool-width ablations (`stage2_kg_rank_ablation.sh`, `stage2_cr4_kg_d20.sh`) don't close this gap, pointing toward the propagation mechanism itself — redistributing relevance mass toward bridge documents even when no bridging was needed — rather than under-tuned capacity.

Full derivations, all six compression rates, and the complete discussion are in the dissertation report.

---

## Acknowledgments

This project builds directly on:
- [**CLaRa**](https://arxiv.org/abs/2511.18659) (He et al., Apple, 2025) and its [official release](https://github.com/apple/ml-clara), which this repository forks and extends.
- The [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) framework CLaRa itself is built on.
- Imperial College London's Department of Computing, for access to the BioMedIA SLURM cluster.
- My supervisor, Dr Avinash Kori, for guidance throughout the project.
