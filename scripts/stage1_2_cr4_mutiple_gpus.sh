#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:4
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/stage1_2_cr4_%j.log
#SBATCH --time=4-11:59:00
#SBATCH --job-name=clara_stage1_2_cr4

set -ex

# ─────────────────────────────────────────
# Environment
# ─────────────────────────────────────────
source /vol/cuda/12.9.0/setup.sh

export CUDA_HOME=/vol/cuda/12.9.0
export UV_CACHE_DIR=/vol/biomedic3/vd425/.cache/uv
export PATH=$CUDA_HOME/bin:$PATH
export NCCL_DEBUG=INFO

#Zero3 Debug
export NCCL_TIMEOUT=1200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/vol/biomedic3/vd425/clara-main/logs/nccl_trace_  # writable dir, not /tmp (avoid quota issues there too)
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export HF_HOME=/vol/biomedic3/vd425/.cache/huggingface
export HF_DATASETS_CACHE=/vol/biomedic3/vd425/.cache/huggingface/datasets
export TRITON_CACHE_DIR=/vol/biomedic3/vd425/.cache/triton
export TORCH_EXTENSIONS_DIR=/vol/biomedic3/vd425/.cache/torch_extensions
export TMPDIR=/vol/biomedic3/vd425/.tmp
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $TMPDIR

source /vol/biomedic3/vd425/clara-main/ml-clara/.venv/bin/activate
export PYTHONPATH=/vol/biomedic3/vd425/clara-main/ml-clara:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvcc --version
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)"

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
CR=4
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
BASE_DIR=/vol/biomedic3/vd425/clara-main
MODEL_PATH=$BASE_DIR/models/Mistral-7B-Instruct-v0.2
DATA_PATH="$BASE_DIR/data/instruction_tuning/train-*.parquet"

# Stage 1 checkpoint to continue from
PRETRAIN_CKPT=$BASE_DIR/checkpoints/stage1/cr${CR}

SAVE_PATH=$BASE_DIR/checkpoints/stage1_2/cr${CR}
mkdir -p $SAVE_PATH $BASE_DIR/logs

echo "Using ${NUM_GPUS} GPUs"
echo "Loading Stage 1 checkpoint from: $PRETRAIN_CKPT"

# ─────────────────────────────────────────
# Training command
# ─────────────────────────────────────────
training_commands=(
    openrlhf.cli.train_sft
    --pretrain "$MODEL_PATH"
    --pretrain_checkpoint "$PRETRAIN_CKPT"
    --stage stage1_2
    --compress_rate "$CR"
    --doc_max_length 256
    --dataset "$DATA_PATH"
    --max_len 2048
    --train_batch_size 128
    --micro_train_batch_size 1
    --save_path "$SAVE_PATH"
    --ckpt_path "$SAVE_PATH/checkpoints"
    --save_steps -2
    --save_hf_ckpt
    --logging_steps 1
    --eval_steps 30
    --zero_stage 2
    --adam_offload
    --max_epochs 1
    --bf16
    --flash_attn
    --learning_rate 1e-4
    --lr_scheduler cosine_with_min_lr
    --gradient_checkpointing
    --generation_top_k 5
    --mse_loss
    --do_eval_gen
)

# ─────────────────────────────────────────
# Launch
# ─────────────────────────────────────────
DISTRIBUTED_ARGS="--nproc_per_node ${NUM_GPUS} \
    --nnodes 1 \
    --rdzv_id 102 \
    --rdzv_backend c10d \
    --rdzv_endpoint 127.0.0.1:29502 \
    --master_addr 127.0.0.1 \
    --master_port 29502 \
    --node_rank 0"

nvidia-smi
echo "Starting Stage 1_2 (instruction tuning) — compression rate ${CR}x on ${NUM_GPUS} GPUs"
torchrun $DISTRIBUTED_ARGS -m "${training_commands[@]}"
echo "Done!"