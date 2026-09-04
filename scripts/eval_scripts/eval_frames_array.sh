#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:4
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/eval_frames_rank_ablation_%a_%j.log
#SBATCH --time=0-11:59:00
#SBATCH --job-name=eval_frames_rank_ablation
#SBATCH --array=0-2

set -ex

# ─────────────────────────────────────────
# Environment — matches the .venv-based convention used by every other
# Stage 2 / KG eval script.
# ─────────────────────────────────────────
source /vol/cuda/12.9.0/setup.sh
export CUDA_HOME=/vol/cuda/12.9.0
export PATH=$CUDA_HOME/bin:$PATH
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_HOME=/vol/biomedic3/vd425/.cache/huggingface
export HF_DATASETS_CACHE=/vol/biomedic3/vd425/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/vol/biomedic3/vd425/.cache/triton
export TORCH_EXTENSIONS_DIR=/vol/biomedic3/vd425/.cache/torch_extensions
export TMPDIR=/vol/biomedic3/vd425/.tmp
export MPLCONFIGDIR=/vol/biomedic3/vd425/.cache/matplotlib
export PYTHONUNBUFFERED=1
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $TMPDIR $MPLCONFIGDIR

source /vol/biomedic3/vd425/clara-main/ml-clara/.venv/bin/activate

BASE_DIR=/vol/biomedic3/vd425/clara-main

# ─────────────────────────────────────────
# Config — FRAMES eval for the KG matrix-rank ablation checkpoints
# (cr16_r32, cr16_r128, cr16_r256). 3 array tasks, one per rank.
#
# Task ID -> rank mapping:
#   0=r32   1=r128   2=r256
# ─────────────────────────────────────────
RANKS=(32 128 256)
RANK=${RANKS[$SLURM_ARRAY_TASK_ID]}

CKPT_PATH=$BASE_DIR/checkpoints/stage2_kg_rank_ablation/cr16_r${RANK}

unset PYTHONPATH
export PYTHONPATH=$BASE_DIR/ml-clara:$CKPT_PATH

echo "=== Starting FRAMES eval: task ${SLURM_ARRAY_TASK_ID} -> cr16_r${RANK} (${CKPT_PATH}) ==="

# One-line pre-flight check, cheap and fast (no GPU touched), confirming the
# checkpoint exists before spending a 4-GPU allocation on it.
if [ ! -f "$CKPT_PATH/config.json" ]; then
    echo "ERROR: $CKPT_PATH/config.json not found -- aborting task ${SLURM_ARRAY_TASK_ID} (cr16_r${RANK})."
    exit 1
fi

python -c "
import json
cfg = json.load(open('$CKPT_PATH/config.json'))
use_kg = cfg.get('use_kg_retrieval', False)
assert use_kg, f'use_kg_retrieval={use_kg} but this is a KG rank-ablation checkpoint (cr16_r${RANK}) -- aborting'
print('OK: KG config confirmed, stage2_retrieval_top_n=%s kg_rank=%s kg_hops=%s kg_damping=%s' % (
    cfg.get('stage2_retrieval_top_n'), cfg.get('kg_rank'), cfg.get('kg_hops'), cfg.get('kg_damping')))
"

# Distinct port per array task to avoid collisions if multiple tasks land on
# the same node concurrently.
MASTER_PORT=$((29650 + SLURM_ARRAY_TASK_ID))

# ─────────────────────────────────────────
# IMPORTANT: evaluate.py always writes to the checkpoint directory using a
# fixed naming scheme -- {dataset}_{stage}_{gold_retrieval}_{top_k}.jsonl and
# results_metrics_{stage}_{gold_retrieval}_{top_k}.json. For frames+stage2+
# False+5 that's exactly "results_metrics_stage2_False_5.json", the SAME
# filename the existing in-distribution (nq/hotpotqa/musique/2wiki) eval
# already wrote into each cr16_r* checkpoint dir. Running frames straight
# into $CKPT_PATH would silently clobber that file.
#
# Fix: point evaluate.py at a separate output directory (a "frames_eval"
# subfolder inside the checkpoint dir) that mirrors just the files it needs
# to run (config.json, tokenizer files, weights) via symlinks, so the
# fixed-name outputs land there instead of in $CKPT_PATH itself. This keeps
# evaluate.py completely untouched.
# ─────────────────────────────────────────
OUT_DIR=$CKPT_PATH/frames_eval
mkdir -p $OUT_DIR
for f in "$CKPT_PATH"/*; do
    fname=$(basename "$f")
    # Don't re-symlink into ourselves, and don't pull in prior eval outputs
    # (jsonl / json results, or a previous frames_eval dir) as "model files".
    case "$fname" in
        frames_eval|*.jsonl|results_metrics_*.json|efficiency_*.json)
            continue
            ;;
    esac
    ln -sf "$f" "$OUT_DIR/$fname"
done

accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --main_process_port=${MASTER_PORT} \
    $BASE_DIR/ml-clara/evaluation/evaluate.py \
        --model_path  "$OUT_DIR" \
        --stage       stage2 \
        --dataset     frames \
        --generation_top_k 5 \
        --batch_size  4 \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}

# ─────────────────────────────────────────
# Rename the fixed-name outputs to carry "frames" in the filename, then move
# them up into $CKPT_PATH (next to the existing in-distribution results, not
# overwriting them) and clean up the symlink scratch dir.
# ─────────────────────────────────────────
if [ -f "$OUT_DIR/results_metrics_stage2_False_5.json" ]; then
    mv "$OUT_DIR/results_metrics_stage2_False_5.json" \
       "$CKPT_PATH/results_metrics_frames_stage2_False_5.json"
fi
if [ -f "$OUT_DIR/frames_stage2_False_5.jsonl" ]; then
    mv "$OUT_DIR/frames_stage2_False_5.jsonl" \
       "$CKPT_PATH/frames_stage2_False_5.jsonl"
fi
if [ -f "$OUT_DIR/efficiency_frames_stage2_False_5.json" ]; then
    mv "$OUT_DIR/efficiency_frames_stage2_False_5.json" \
       "$CKPT_PATH/efficiency_frames_stage2_False_5.json"
fi

rm -rf "$OUT_DIR"

echo ""
echo "========================================"
echo "Done — FRAMES eval, cr16_r${RANK} (rank ablation). Results in $CKPT_PATH/"
echo "  -> $CKPT_PATH/frames_stage2_False_5.jsonl"
echo "  -> $CKPT_PATH/results_metrics_frames_stage2_False_5.json"
echo "  (existing results_metrics_stage2_False_5.json left untouched)"
echo "========================================"