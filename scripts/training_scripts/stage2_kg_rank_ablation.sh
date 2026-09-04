#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:4
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/stage2_kg_cr16_rank%a_%j.log
#SBATCH --time=4-11:59:00
#SBATCH --job-name=clara_stage2_kg_rank_ablation_cr16
#SBATCH --array=32,128,256

set -ex

# ─────────────────────────────────────────
# Environment
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
export TRITON_CACHE_DIR=/vol/biomedic3/vd425/.cache/triton
export TORCH_EXTENSIONS_DIR=/vol/biomedic3/vd425/.cache/torch_extensions
export TMPDIR=/vol/biomedic3/vd425/.tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $TMPDIR

source /vol/biomedic3/vd425/clara-main/ml-clara/.venv/bin/activate
export PYTHONPATH=/vol/biomedic3/vd425/clara-main/ml-clara:$PYTHONPATH

nvcc --version
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)"

# Fail fast if the new module isn't where it needs to be, rather than 40 minutes
# into model loading on 4 GPUs.
python -c "from openrlhf.models.temporal_link_matrix import TemporalLinkMatrix, multi_hop_propagate" \
    || { echo "ERROR: openrlhf/models/temporal_link_matrix.py not found/importable. Aborting."; exit 1; }

# ─────────────────────────────────────────
# Config — kg_rank (r) comes from the array task ID.
#
# CR is FIXED at 16 for this ablation. Per the vanilla-vs-KG delta table,
# CR=16 is the single worst-performing setting for stage2_kg across every
# metric and every dataset (mean F1 delta -2.88, mean R@5 delta -14.74 --
# both the largest-magnitude regressions in the sweep). It's the cleanest
# case to test whether r is the culprit: if a different rank recovers
# performance here, that's the strongest possible signal; if nothing helps
# at CR=16, that points away from r and toward D (stage2_retrieval_top_n)
# or the KG mechanism itself.
#
# r=64 is the existing baseline (already run, not repeated here). This
# array covers r=32 and r=128 (bracketing 64) plus r=256, since the
# corrected E at CR=16 (mem_tokens=16, E=16*4096=65,536) makes even r=256
# well within low-rank-bottleneck territory (<1% of E) -- no oversizing
# concern.
# ─────────────────────────────────────────
KG_RANK=${SLURM_ARRAY_TASK_ID}
CR_FIXED=16
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
BASE_DIR=/vol/biomedic3/vd425/clara-main

MODEL_PATH=$BASE_DIR/models/Mistral-7B-Instruct-v0.2
PRETRAIN_CHECKPOINT=$BASE_DIR/checkpoints/stage1_2/cr${CR_FIXED}

# Same data as the baseline/KG runs -- only kg_rank changes across array tasks,
# so this is a clean single-variable ablation against the existing r=64 KG run.
DATA_PATH="$BASE_DIR/data/e2e_training_ours_*_with_pos/train-*.parquet"

SAVE_PATH=$BASE_DIR/checkpoints/stage2_kg_rank_ablation/cr${CR_FIXED}_r${KG_RANK}
mkdir -p $SAVE_PATH $BASE_DIR/logs

echo "Using ${NUM_GPUS} GPUs — CR=${CR_FIXED} (fixed), kg_rank=${KG_RANK} (swept)"
echo "Pretrain checkpoint: ${PRETRAIN_CHECKPOINT}  (stage1_2 -- link_module has no prior"
echo "  weights here and will train from a fresh random init during this run, same as"
echo "  the original stage2_kg_array.sh and r=64 baseline runs)"
echo "Train data: ${DATA_PATH}"

# ─────────────────────────────────────────
# Training command
# ─────────────────────────────────────────
# Identical to stage2_kg_array.sh in every respect except:
#   --compress_rate      now fixed at 16 (was swept via array)
#   --kg_rank              now swept via array at 32/128/256 (was fixed at 64)
#
# stage2_retrieval_top_n (D) and everything else (data, epochs, LR, batch
# sizes, precision, kg_hops, kg_damping, checkpointing cadence) are left
# UNCHANGED on purpose -- this ablation isolates r alone. D gets its own
# sweep in a follow-up script once the best r from this run is picked.
training_commands=(
    openrlhf.cli.train_sft
    --pretrain "$MODEL_PATH"
    --pretrain_checkpoint "$PRETRAIN_CHECKPOINT"
    --stage stage2
    --compress_rate "$CR_FIXED"
    --doc_max_length 256
    --dataset "$DATA_PATH"
    --max_len 1024
    --train_batch_size 32
    --micro_train_batch_size 1
    --save_path "$SAVE_PATH"
    --ckpt_path "$SAVE_PATH/checkpoints"
    --save_steps -2
    --save_hf_ckpt
    --logging_steps 1
    --eval_steps 100
    --zero_stage 2
    --max_epochs 1
    --bf16
    --flash_attn
    --learning_rate 5e-6
    --lr_scheduler constant
    --gradient_checkpointing
    --stage2_retrieval_top_n 100
    --generation_top_k 5
    --use_kg_retrieval
    --kg_rank "$KG_RANK"
    --kg_hops 2
    --kg_damping 0.5
    --qa_loss
    --do_eval_gen
)

# ─────────────────────────────────────────
# Launch
# ─────────────────────────────────────────
# Unique port per array task (keyed on KG_RANK) to avoid collisions if jobs
# land on the same node.
MASTER_PORT=$((29504 + KG_RANK))

DISTRIBUTED_ARGS="--nproc_per_node ${NUM_GPUS} \
    --nnodes 1 \
    --rdzv_id ${KG_RANK} \
    --rdzv_backend c10d \
    --rdzv_endpoint 127.0.0.1:${MASTER_PORT} \
    --master_addr 127.0.0.1 \
    --master_port ${MASTER_PORT} \
    --node_rank 0"

nvidia-smi
echo "Starting Stage 2 (end-to-end, KG-augmented) rank ablation — CR=${CR_FIXED}, kg_rank=${KG_RANK} on ${NUM_GPUS} GPUs"
torchrun $DISTRIBUTED_ARGS -m "${training_commands[@]}"

# Sanity check: confirm the link module actually got saved, not just the usual files.
if [ -f "$SAVE_PATH/link_module.pth" ]; then
    echo "OK: link_module.pth found in $SAVE_PATH"
else
    echo "WARNING: link_module.pth NOT found in $SAVE_PATH -- KG weights were not persisted."
fi

# Copy model definition (and the new module) alongside checkpoint, matching
# the baseline/KG scripts' existing convention.
cp $BASE_DIR/ml-clara/openrlhf/models/modeling_clara.py $SAVE_PATH
cp $BASE_DIR/ml-clara/openrlhf/models/temporal_link_matrix.py $SAVE_PATH

echo "Done — CR=${CR_FIXED}, kg_rank=${KG_RANK}!"