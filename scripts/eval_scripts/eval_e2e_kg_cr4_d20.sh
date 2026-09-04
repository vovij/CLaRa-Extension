#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:4
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/eval_stage2_cr4_kg_d20_%j.log
#SBATCH --time=23:59:59
#SBATCH --job-name=eval_stage2_cr4_kg_d20
# Waits for the D=20 training job (clara_stage2_cr4_kg_d20, node loki) to
# finish and exit cleanly before this eval starts. Update the job ID below
# if you resubmit training and get a new one.
#SBATCH --dependency=afterok:74978

set -ex

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
# Points at the D=20 pool-width ablation checkpoint, NOT the D=100 default
# (checkpoints/stage2_kg/cr4). Training script: stage2_cr4_kg_d20.sh
CKPT_PATH=$BASE_DIR/checkpoints/stage2_kg/cr4_d20

unset PYTHONPATH
export PYTHONPATH=$BASE_DIR/ml-clara:$CKPT_PATH

# One-line pre-flight check, cheap and fast (no GPU touched), confirming
# use_kg_retrieval AND the expected D=20 candidate pool actually made it into
# this checkpoint's config.json before spending a 4-GPU allocation on it.
# NOTE: stage2_retrieval_top_n is asserted == 20 here (not 100) -- that is
# the whole point of this ablation (Section 5.5.1 pool-width sweep). An
# accidental match against a D=100 checkpoint would fail this assert.
python -c "
import json
cfg = json.load(open('$CKPT_PATH/config.json'))
assert cfg.get('use_kg_retrieval') is True, f'use_kg_retrieval={cfg.get(\"use_kg_retrieval\")} -- aborting'
assert cfg.get('stage2_retrieval_top_n') == 20, f'stage2_retrieval_top_n={cfg.get(\"stage2_retrieval_top_n\")} (expected 20) -- aborting'
print('OK: KG config confirmed, stage2_retrieval_top_n=%s kg_rank=%s kg_hops=%s kg_damping=%s' % (
    cfg.get('stage2_retrieval_top_n'), cfg.get('kg_rank'), cfg.get('kg_hops'), cfg.get('kg_damping')))
"

# Distinct port from the D=100 eval script, in case both are ever queued/run
# around the same time on the same node.
accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --main_process_port=29605 \
    $BASE_DIR/ml-clara/evaluation/evaluate.py \
        --model_path  "$CKPT_PATH" \
        --stage       stage2 \
        --dataset     nq,hotpotqa,musique,2wiki \
        --generation_top_k 5 \
        --batch_size  4 \
        ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}

echo ""
echo "========================================"
echo "Done — CR=4, D=20 KG ablation. Results in $CKPT_PATH/"
echo "========================================"