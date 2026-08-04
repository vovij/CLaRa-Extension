#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:2
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/sweep_%a_%j.log
#SBATCH --time=6:00:00
#SBATCH --job-name=kg_sweep_12
#SBATCH --exclude=lora
#SBATCH --array=0-11

set -ex

# Stagger task starts by 60s per array index to reduce the chance of two
# tasks hitting rendezvous/GPU-init at the exact same instant when they
# land on the same node. This is a supplement to --main_process_port and
# the %4 throttle above, not a replacement for either.

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

echo "SLURM_JOB_ID=$SLURM_JOB_ID SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES (as set by SLURM)=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS  SLURM_STEP_GPUS=$SLURM_STEP_GPUS"
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv
if [ -n "$SLURM_JOB_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
elif [ -n "$SLURM_STEP_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES=$SLURM_STEP_GPUS
fi
echo "CUDA_VISIBLE_DEVICES (final)=$CUDA_VISIBLE_DEVICES"

BASE_DIR=/vol/biomedic3/vd425/clara-main
SWEEP_ROOT=$BASE_DIR/checkpoints/stage2_kg_sweep

HOPS_LIST=(2 2 2 2 2 2 1 1 3 3 4 4)
DAMP_LIST=(0.1 0.3 0.5 0.7 0.8 0.9 0.5 0.9 0.5 0.9 0.5 0.9)
HOPS=${HOPS_LIST[$SLURM_ARRAY_TASK_ID]}
DAMP=${DAMP_LIST[$SLURM_ARRAY_TASK_ID]}

CKPT_PATH=$SWEEP_ROOT/hops${HOPS}_damp${DAMP}

unset PYTHONPATH
export PYTHONPATH=$BASE_DIR/ml-clara:$CKPT_PATH

python -c "
import json
cfg = json.load(open('$CKPT_PATH/config.json'))
assert cfg.get('use_kg_retrieval') is True
assert cfg.get('kg_hops') == $HOPS
assert cfg.get('kg_damping') == $DAMP
print('OK: sweep variant confirmed, hops=%s damping=%s' % (cfg.get('kg_hops'), cfg.get('kg_damping')))
"

# NUM_GPUS is derived from SLURM's own allocation (matches #SBATCH --gres=gpu:N
# above automatically) rather than being hardcoded separately, so this can
# never drift out of sync with the requested GPU count again.
NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}
echo "NUM_GPUS (from SLURM_GPUS_ON_NODE, matches --gres=gpu:N)=$NUM_GPUS"

# Unique rendezvous port per array task -- prevents EADDRINUSE when two tasks
# land on the same physical node at the same time (SLURM does not guarantee
# array tasks get separate nodes).
PORT=$((29500 + SLURM_ARRAY_TASK_ID))
echo "Using main_process_port=$PORT for array task $SLURM_ARRAY_TASK_ID"

accelerate launch \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --main_process_port=$PORT \
    $BASE_DIR/ml-clara/evaluation/evaluate.py \
        --model_path  "$CKPT_PATH" \
        --stage       stage2 \
        --dataset     nq,hotpotqa,musique,2wiki \
        --generation_top_k 5 \
        --batch_size  4 \
        --max_samples 500

echo "Done sweep variant hops=$HOPS damping=$DAMP"
