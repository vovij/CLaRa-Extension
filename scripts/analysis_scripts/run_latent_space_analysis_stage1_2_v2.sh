#!/bin/bash
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/biomedic3/vd425/clara-main/logs/latent_space_v2_stage1_2_%j.log
#SBATCH --time=0-02:00:00
#SBATCH --job-name=clara_latent_v2_stage1_2

set -ex

# ─────────────────────────────────────────
# Environment  (identical to run_latent_space_analysis_stage1.sh)
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
export MPLCONFIGDIR=/vol/biomedic3/vd425/.cache/matplotlib
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $TMPDIR $MPLCONFIGDIR

source /vol/biomedic3/vd425/clara-main/ml-clara/.venv/bin/activate
export PYTHONPATH=/vol/biomedic3/vd425/clara-main/ml-clara:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvcc --version
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)"

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
BASE_DIR=/vol/biomedic3/vd425/clara-main
DATASET=musique
N_DOCS=200
N_QUERIES=100
BATCH_SIZE=8
CRS="4 16 32 64 128 256"

CKPT_ROOT="$BASE_DIR/checkpoints/stage1_2/cr{cr}"

OUT_DIR=$BASE_DIR/results/latent_space_v2/stage1_2
mkdir -p $OUT_DIR $BASE_DIR/logs

echo "Running latent space analysis v2 (no aggregation)"
echo "  dataset:   $DATASET"
echo "  n_docs:    $N_DOCS"
echo "  CRs:       $CRS"
echo "  ckpt_root: $CKPT_ROOT"
echo "  out_dir:   $OUT_DIR"

# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────
python $BASE_DIR/scripts/eval/latent_space_analysis_v2.py \
    --base_dir    "$BASE_DIR"    \
    --ckpt_root   "$CKPT_ROOT"  \
    --stage_label stage1_2      \
    --mode        compression   \
    --projection  all           \
    --crs         $CRS          \
    --dataset     "$DATASET"    \
    --n_docs      $N_DOCS       \
    --n_queries   $N_QUERIES    \
    --batch_size  $BATCH_SIZE   \
    --max_pts     3000          \
    --out_dir     "$OUT_DIR"    \
    --device      cuda

echo "Done! Results saved to $OUT_DIR"