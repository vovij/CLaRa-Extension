#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:4
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/eval_frames_cr4_d20_%j.log
#SBATCH --time=0-11:59:00
#SBATCH --job-name=eval_frames_cr4_d20
# Waits for the D=20 training job (clara_stage2_cr4_kg_d20, job 74978, node
# loki) to finish and exit cleanly before this eval starts. Update the job ID
# below if you resubmit training and get a new one.
#SBATCH --dependency=afterok:74978

set -ex

# ─────────────────────────────────────────
# Environment — same .venv convention as eval_frames_cr4_vanilla_vs_kg.sh
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
# Config — CR4, KG, D=20 candidate pool.
# This checkpoint's training-time pool width (20) matches FRAMES's actual
# fixed pool width (20, per prepare_frames.py), unlike the D=100 checkpoint
# already evaluated in eval_frames_cr4_vanilla_vs_kg.sh. Comparing this
# result against that run's stage2_kg/cr4 (D=100) number is what lets us
# separate "TLM mechanism doesn't help on FRAMES" from "TLM's train/eval
# pool-width mismatch hurt the D=100 FRAMES result" (see chat discussion,
# Section 5.5.1 / 5.7 open items).
# ─────────────────────────────────────────
CKPT_PATH=$BASE_DIR/checkpoints/stage2_kg/cr4_d20

unset PYTHONPATH
export PYTHONPATH=$BASE_DIR/ml-clara:$CKPT_PATH

echo "=== Starting FRAMES eval: CR4 kg d20 (${CKPT_PATH}) ==="

# Pre-flight check: confirm use_kg_retrieval=True AND stage2_retrieval_top_n
# is actually 20 (not 100) before spending a 4-GPU allocation on it -- that's
# the entire point of running this checkpoint on FRAMES.
python -c "
import json
cfg = json.load(open('$CKPT_PATH/config.json'))
assert cfg.get('use_kg_retrieval') is True, f'use_kg_retrieval={cfg.get(\"use_kg_retrieval\")} -- aborting'
assert cfg.get('stage2_retrieval_top_n') == 20, f'stage2_retrieval_top_n={cfg.get(\"stage2_retrieval_top_n\")} (expected 20) -- aborting'
print('OK: KG D=20 config confirmed, stage2_retrieval_top_n=%s kg_rank=%s kg_hops=%s kg_damping=%s' % (
    cfg.get('stage2_retrieval_top_n'), cfg.get('kg_rank'), cfg.get('kg_hops'), cfg.get('kg_damping')))
"

accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --main_process_port=29612 \
    $BASE_DIR/ml-clara/evaluation/evaluate.py \
        --model_path  "$CKPT_PATH" \
        --stage       stage2 \
        --dataset     frames \
        --generation_top_k 5 \
        --batch_size  4 \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}

echo ""
echo "========================================"
echo "Done — FRAMES eval, CR4 kg d20. Results in $CKPT_PATH/"
echo "========================================"
