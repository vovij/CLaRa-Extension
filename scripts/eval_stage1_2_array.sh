#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:4
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/eval_stage1_2_cr%a_%j.log
#SBATCH --time=4:00:00
#SBATCH --job-name=eval_stage1_2
#SBATCH --array=4

set -ex

# ─────────────────────────────────────────
# Environment  (mirrors stage1_2 training script)
# ─────────────────────────────────────────
source /vol/cuda/12.9.0/setup.sh

export CUDA_HOME=/vol/cuda/12.9.0
export UV_CACHE_DIR=/vol/biomedic3/vd425/.cache/uv
export PATH=$CUDA_HOME/bin:$PATH
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_HOME=/vol/biomedic3/vd425/.cache/huggingface
export HF_DATASETS_CACHE=/vol/biomedic3/vd425/.cache/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/vol/biomedic3/vd425/.cache/triton
export TORCH_EXTENSIONS_DIR=/vol/biomedic3/vd425/.cache/torch_extensions
export TMPDIR=/vol/biomedic3/vd425/.tmp
export PYTHONUNBUFFERED=1
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $TMPDIR

source /vol/biomedic3/vd425/clara-main/ml-clara/.venv/bin/activate

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
CR=${SLURM_ARRAY_TASK_ID}
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
BASE_DIR=/vol/biomedic3/vd425/clara-main
CKPT_PATH=$BASE_DIR/checkpoints/stage1_2/cr${CR}
MAX_SAMPLES=500   # set to "" for full eval

# modeling_clara.py lives in the checkpoint dir — must be on PYTHONPATH
# (same pattern as apple eval scripts)
unset PYTHONPATH
export PYTHONPATH=$BASE_DIR/ml-clara:$CKPT_PATH:$PYTHONPATH

echo "========================================"
echo "Evaluating stage1_2  CR=${CR}x  GPUs=${NUM_GPUS}"
echo "Checkpoint : $CKPT_PATH"
echo "========================================"
nvcc --version
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)"
nvidia-smi

# ─────────────────────────────────────────
# Helper
# ─────────────────────────────────────────
run_eval() {
    local DATASET=$1
    local GOLD_FLAG=$2   # "--gold_retrieval" or ""
    local LABEL=$3       # "with_pos" or "no_pos"

    echo ""
    echo ">>> dataset=${DATASET}  retrieval=${LABEL}"

    accelerate launch \
        --num_processes=${NUM_GPUS} \
        --num_machines=1 \
        --mixed_precision=bf16 \
        $BASE_DIR/ml-clara/evaluation/evaluate.py \
            --model_path  "$CKPT_PATH" \
            --stage       stage1_2 \
            --dataset     "$DATASET" \
            --generation_top_k 5 \
            --batch_size  4 \
            ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
            $GOLD_FLAG
}

# ─────────────────────────────────────────
# All 4 datasets × normal + oracle
# ─────────────────────────────────────────
for DATASET in nq hotpotqa musique 2wiki; do
    run_eval "$DATASET" ""                 "no_pos"
    run_eval "$DATASET" "--gold_retrieval" "with_pos"
done

echo ""
echo "========================================"
echo "Done — CR=${CR}x.  Results in $CKPT_PATH/"
echo "========================================"