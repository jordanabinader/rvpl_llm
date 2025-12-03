#!/bin/bash

#########################
# Slurm job parameters  #
#########################

#SBATCH -J transformer_vpl_prism       # Job name
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

#########################
# Configuration         #
#########################

SEQ_LENGTH=${1:-10}
LATENT_DIM=${2:-512}
HIDDEN_DIM=${3:-512}
BETA_MAX=${4:-0.1}
NUM_HEADS=${5:-4}
NUM_LAYERS=${6:-2}
SEED=${7:-0}

# Fixed hyperparameters
CYCLES=4
GAMMA=1.1

# Experiment name with "transformer" tag
EXP_NAME="transformer_prism_seq${SEQ_LENGTH}_latent${LATENT_DIM}_hidden${HIDDEN_DIM}_beta${BETA_MAX}_heads${NUM_HEADS}_layers${NUM_LAYERS}_seed${SEED}"

echo "========================================="
echo "Transformer VPL Training on PRISM"
echo "========================================="
echo "Configuration:"
echo "  Architecture: TRANSFORMER (Self-Attention)"
echo "  Sequence Length: $SEQ_LENGTH"
echo "  Latent Dimension: $LATENT_DIM"
echo "  Hidden Dimension: $HIDDEN_DIM"
echo "  Attention Heads: $NUM_HEADS"
echo "  Transformer Layers: $NUM_LAYERS"
echo "  Beta Max: $BETA_MAX"
echo "  Seed: $SEED"
echo "========================================="
echo ""

#########################
# Training              #
#########################

python -m hidden_context.train_streaming_vpl \
    --data_path data_release/prism/gpt2 \
    --seq_length ${SEQ_LENGTH} \
    --latent_dim ${LATENT_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --encoder_embed_dim 768 \
    --decoder_embed_dim 768 \
    --use_transformer True \
    --num_attention_heads ${NUM_HEADS} \
    --num_transformer_layers ${NUM_LAYERS} \
    --transformer_dropout 0.1 \
    --beta_max ${BETA_MAX} \
    --beta_cycles ${CYCLES} \
    --temporal_gamma ${GAMMA} \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --eval_steps 500 \
    --save_steps 500 \
    --log_dir experiments/streaming_vpl_prism \
    --seed ${SEED} \
    --bf16 True

echo ""
echo "========================================="
echo "Step 2: Evaluation"
echo "=========================================="

# Find the trained model
echo "Searching for model checkpoint..."
MODEL_PATH=$(find experiments/streaming_vpl_prism -name "model.pt" -path "*/final_checkpoint/*" -type f 2>/dev/null | grep -E "seq${SEQ_LENGTH}" | tail -1)

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Could not find model checkpoint"
    echo "Available models:"
    find experiments/streaming_vpl_prism -name "model.pt" -path "*/final_checkpoint/*" 2>/dev/null
    exit 1
fi

echo "Found model at: $MODEL_PATH"
EVAL_DIR=$(dirname $(dirname ${MODEL_PATH}))/evaluation

python -m hidden_context.evaluate_streaming_vpl \
    --model_path ${MODEL_PATH} \
    --data_path data_release/prism/gpt2 \
    --seq_length ${SEQ_LENGTH} \
    --latent_dim ${LATENT_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --encoder_embed_dim 768 \
    --decoder_embed_dim 768 \
    --use_transformer True \
    --num_attention_heads ${NUM_HEADS} \
    --num_transformer_layers ${NUM_LAYERS} \
    --num_eval_episodes 200 \
    --output_dir ${EVAL_DIR} \
    --seed ${SEED}

echo ""
echo "========================================="
echo "Complete!"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Evaluation: ${EVAL_DIR}"
echo ""
echo "View on W&B:"
echo "  https://wandb.ai/$(whoami)/streaming-vpl-prism"
echo "========================================="

