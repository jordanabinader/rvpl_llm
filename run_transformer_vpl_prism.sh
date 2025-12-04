#!/bin/bash

#########################
# Slurm job parameters  #
#########################

#SBATCH -J transformer_vpl_prism            # Job name
#SBATCH -p mit_normal_gpu                   # GPU partition
#SBATCH -c 4                                # CPU cores
#SBATCH --mem=32G                           # Memory
#SBATCH -t 4:00:00                          # Time limit
#SBATCH -G 1                                # 1 GPU
#SBATCH -o logs/transformer_vpl_prism_%j.out # STDOUT log
#SBATCH -e logs/transformer_vpl_prism_%j.err # STDERR log

#########################
# Environment setup     #
#########################

# Create logs directory
mkdir -p logs

# W&B Configuration
export WANDB_MODE=online
export WANDB_PROJECT="streaming-vpl-prism"
export WANDB_LOG_MODEL="false"
export WANDB_WATCH="false"

# 1. Load the base system python
module load miniforge

# 2. Create the virtual environment (if it doesn't exist)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# 3. Activate the environment
source venv/bin/activate

# 4. Ensure we rely ONLY on this venv
export PYTHONNOUSERSITE=1

# 5. Upgrade pip
python -m pip install --upgrade pip

echo "Node: $(hostname)"
echo "GPUs visible to this job:"
nvidia-smi || echo "nvidia-smi not found"

# Verify versions
echo "Package versions in venv:"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}')"

# base slurm id (we'll append hyperparams per run)
RUN_ID_BASE="slurm_${SLURM_JOB_ID:-local}"

#########################
# Base configuration    #
#########################

SEQ_LENGTH=${1:-6}
LATENT_DIM=${2:-64}
HIDDEN_DIM=${3:-256}
# These single values are now just defaults; grid below overrides them
BETA_MAX_DEFAULT=${4:-0.05}
NUM_HEADS=${5:-4}
NUM_LAYERS=${6:-2}
SEED=${7:-0}
EPOCH_SIZE=${8:-4000}
FREE_BITS_DEFAULT=${9:-3.2}
DATA_SUBSET=${10:-both}
ALLOW_KL_GRADIENT_FLOW=${11:-True}
USE_CONTRASTIVE=${12:-True}

# Fixed hyperparameters
CYCLES=4
GAMMA=1.1

#########################
# Grid specification    #
#########################
# Edit these arrays to change the grid
BETA_GRID=(0.08 0.10 0.12)
FREE_BITS_GRID=(4.0 6.4 8.0)

echo "========================================="
echo "Transformer VPL Training on PRISM (Grid Search)"
echo "========================================="
echo "Base Configuration (non-grid params):"
echo "  Architecture: TRANSFORMER (Self-Attention)"
echo "  Sequence Length: $SEQ_LENGTH"
echo "  Latent Dimension: $LATENT_DIM"
echo "  Hidden Dimension: $HIDDEN_DIM"
echo "  Attention Heads: $NUM_HEADS"
echo "  Transformer Layers: $NUM_LAYERS"
echo "  Seed: $SEED"
echo "  Epoch Size: $EPOCH_SIZE"
echo "  Data Subset: $DATA_SUBSET"
echo "  Cycles: $CYCLES"
echo "  Gamma: $GAMMA"
echo "========================================="
echo "Grid:"
echo "  Beta Max grid: ${BETA_GRID[@]}"
echo "  Free Bits grid: ${FREE_BITS_GRID[@]}"
echo "========================================="
echo ""

#########################
# Grid search loop      #
#########################

for BETA_MAX in "${BETA_GRID[@]}"; do
  for FREE_BITS in "${FREE_BITS_GRID[@]}"; do

    # Normalize FREE_BITS for filenames (replace '.' with 'p')
    FREE_BITS_TAG=${FREE_BITS/./p}
    BETA_TAG=${BETA_MAX/./p}

    # Experiment name with "transformer" tag
    EXP_NAME="${DATA_SUBSET}_seq${SEQ_LENGTH}_latent${LATENT_DIM}_hidden${HIDDEN_DIM}_beta${BETA_MAX}_freebits${FREE_BITS}_cycles${CYCLES}_gamma${GAMMA}_seed${SEED}"

    # Per-run RUN_ID (avoid collisions within same Slurm job)
    RUN_ID="${RUN_ID_BASE}_b${BETA_TAG}_fb${FREE_BITS_TAG}"

    # W&B Tags for easier filtering
    export WANDB_TAGS="transformer,seq${SEQ_LENGTH},latent${LATENT_DIM},heads${NUM_HEADS},layers${NUM_LAYERS},beta${BETA_MAX},freebits${FREE_BITS},seed${SEED}"

    echo ""
    echo "========================================="
    echo "Starting run:"
    echo "  BETA_MAX = ${BETA_MAX}"
    echo "  FREE_BITS = ${FREE_BITS}"
    echo "  RUN_ID = ${RUN_ID}"
    echo "  EXP_NAME = ${EXP_NAME}"
    echo "========================================="
    echo ""

    #########################
    # Training              #
    #########################

    python -m hidden_context.train_streaming_vpl \
        --data_path data_release/prism/gpt2 \
        --data_subset "${DATA_SUBSET}" \
        --seq_length "${SEQ_LENGTH}" \
        --latent_dim "${LATENT_DIM}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --encoder_embed_dim 768 \
        --decoder_embed_dim 768 \
        --use_transformer True \
        --num_attention_heads "${NUM_HEADS}" \
        --num_transformer_layers "${NUM_LAYERS}" \
        --transformer_dropout 0.1 \
        --beta_max "${BETA_MAX}" \
        --beta_cycles "${CYCLES}" \
        --temporal_gamma "${GAMMA}" \
        --free_bits "${FREE_BITS}" \
        --epoch_size "${EPOCH_SIZE}" \
        --learning_rate 2e-4 \
        --num_train_epochs 10 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --eval_steps 500 \
        --save_steps 500 \
        --log_dir experiments/streaming_vpl_prism \
        --seed "${SEED}" \
        --bf16 True \
        --run_id "${RUN_ID}" \
        --allow_kl_gradient_flow "${ALLOW_KL_GRADIENT_FLOW}" \
        --use_contrastive "${USE_CONTRASTIVE}"

    echo ""
    echo "========================================="
    echo "Step 2: Evaluation (BETA_MAX=${BETA_MAX}, FREE_BITS=${FREE_BITS})"
    echo "=========================================="

    # Find the trained model
    echo "Searching for model checkpoint..."
    MODEL_PATH="experiments/streaming_vpl_prism/${EXP_NAME}/${RUN_ID}/final_checkpoint/model.pt"

    echo "Found model at: $MODEL_PATH"
    EVAL_DIR="$(dirname "$(dirname "${MODEL_PATH}")")/evaluation"

    python -m hidden_context.evaluate_streaming_vpl \
        --model_path "${MODEL_PATH}" \
        --data_path data_release/prism/gpt2 \
        --data_subset "${DATA_SUBSET}" \
        --seq_length "${SEQ_LENGTH}" \
        --latent_dim "${LATENT_DIM}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --encoder_embed_dim 768 \
        --decoder_embed_dim 768 \
        --use_transformer True \
        --num_attention_heads "${NUM_HEADS}" \
        --num_transformer_layers "${NUM_LAYERS}" \
        --num_eval_episodes 200 \
        --output_dir "${EVAL_DIR}" \
        --seed "${SEED}" \
        --use_contrastive "${USE_CONTRASTIVE}"

    echo ""
    echo "========================================="
    echo "Run complete (BETA_MAX=${BETA_MAX}, FREE_BITS=${FREE_BITS})"
    echo "Model: ${MODEL_PATH}"
    echo "Evaluation: ${EVAL_DIR}"
    echo "========================================="
    echo ""

  done
done

echo "========================================="
echo "All grid runs complete!"
echo "View on W&B:"
echo "  https://wandb.ai/$(whoami)/streaming-vpl-prism"
echo "========================================="
