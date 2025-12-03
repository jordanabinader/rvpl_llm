#!/bin/bash

#########################
# Slurm job parameters  #
#########################

#SBATCH -J streaming_vpl_prism         # Job name
#SBATCH -p mit_normal_gpu                   # GPU partition
#SBATCH -c 4                                # CPU cores
#SBATCH --mem=32G                           # Memory
#SBATCH -t 4:00:00                          # Time limit (4 hours for larger dataset)
#SBATCH -G 1                                # 1 GPU
#SBATCH -o logs/streaming_vpl_prism_%j.out # STDOUT log
#SBATCH -e logs/streaming_vpl_prism_%j.err # STDERR log

#########################
# Environment setup     #
#########################

# Create logs directory
mkdir -p logs

# W&B Configuration for comprehensive tracking
export WANDB_MODE=online
export WANDB_PROJECT="streaming-vpl-prism"
export WANDB_LOG_MODEL="false"  # Don't upload model checkpoints to wandb
export WANDB_WATCH="false"      # Don't watch gradients (can be slow)

# 1. Load the base system python
module load miniforge

# 2. Create the virtual environment (if it doesn't exist)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# 3. Activate the environment
source venv/bin/activate

# 4. CRITICAL: Ensure we rely ONLY on this venv, ignoring local user packages
export PYTHONNOUSERSITE=1

# 5. Upgrade pip inside the virtual environment
python -m pip install --upgrade pip

# 6. Install packages
# echo "Installing dependencies..."
# python -m pip install --upgrade --ignore-installed --no-cache-dir \
#     "torch>=2.0" \
#     "transformers==4.40.0" \
#     "tokenizers==0.19.1" \
#     "peft==0.10.0" \
#     "accelerate>=0.27.0" \
#     "numpy" \
#     "sentencepiece" \
#     "datasets" \
#     "wandb" \
#     "matplotlib" \
#     "scipy" \
#     "scikit-learn" 

echo "Node: $(hostname)"
echo "GPUs visible to this job:"
nvidia-smi || echo "nvidia-smi not found"

# Verify versions
echo "Package versions in venv:"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import peft; print(f'peft: {peft.__version__}')"
python -c "import tokenizers; print(f'tokenizers: {tokenizers.__version__}')"
python -c "import accelerate; print(f'accelerate: {accelerate.__version__}')"

#########################
# Configuration         #
#########################

SEQ_LENGTH=${1:-10}
LATENT_DIM=${2:-512}
HIDDEN_DIM=${3:-512}
BETA_MAX=${4:-0.1}
SEED=${5:-0}

# Fixed hyperparameters
CYCLES=4
GAMMA=1.1

# Experiment name
EXP_NAME="prism_seq${SEQ_LENGTH}_latent${LATENT_DIM}_hidden${HIDDEN_DIM}_beta${BETA_MAX}_cycles${CYCLES}_gamma${GAMMA}_seed${SEED}"
OUTPUT_DIR="experiments/streaming_vpl_prism/${EXP_NAME}"

echo "Configuration:"
echo "  Sequence Length: $SEQ_LENGTH"
echo "  Latent Dimension: $LATENT_DIM"
echo "  Hidden Dimension: $HIDDEN_DIM"
echo "  Beta Max: $BETA_MAX"
echo "  Seed: $SEED"
echo "  Output: $OUTPUT_DIR"
echo ""

#########################
# Training              #
#########################

echo "========================================="
echo "Step 1: Training Streaming VPL on PRISM"
echo "========================================="

python -m hidden_context.train_streaming_vpl \
    --data_path data_release/prism/gpt2 \
    --seq_length ${SEQ_LENGTH} \
    --latent_dim ${LATENT_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --encoder_embed_dim 768 \
    --decoder_embed_dim 768 \
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
echo "Step 2: Evaluation (With Thompson Sampling)"
echo "=========================================="

# Re-enable W&B for evaluation (it logs separately)
export WANDB_MODE=online

# Find the actual model checkpoint (training creates nested directory structure)
echo "Searching for model checkpoint..."
MODEL_PATH=$(find experiments/streaming_vpl_prism -name "model.pt" -path "*/final_checkpoint/*" -type f 2>/dev/null | grep -E "(${EXP_NAME}|both_seq${SEQ_LENGTH})" | tail -1)

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Could not find model.pt for experiment"
    echo "Searched for patterns: ${EXP_NAME} or both_seq${SEQ_LENGTH}"
    echo ""
    echo "Available model checkpoints:"
    find experiments/streaming_vpl_prism -name "model.pt" -path "*/final_checkpoint/*" 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "Found model at: $MODEL_PATH"

# Determine evaluation output directory (same level as final_checkpoint)
EVAL_DIR=$(dirname $(dirname ${MODEL_PATH}))/evaluation
echo "Evaluation results will be saved to: $EVAL_DIR"

python -m hidden_context.evaluate_streaming_vpl \
    --model_path ${MODEL_PATH} \
    --data_path data_release/prism/gpt2 \
    --seq_length ${SEQ_LENGTH} \
    --latent_dim ${LATENT_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --encoder_embed_dim 768 \
    --decoder_embed_dim 768 \
    --num_eval_episodes 200 \
    --output_dir ${EVAL_DIR} \
    --seed ${SEED}

echo ""
echo "========================================="
echo "Complete!"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "Evaluation results: ${EVAL_DIR}"
echo ""
echo "To view results:"
echo "  cat ${EVAL_DIR}/results.json"
echo "  open ${EVAL_DIR}/adaptation_curve.png"
echo ""
echo "View on W&B:"
echo "  https://wandb.ai/$(whoami)/streaming-vpl-prism"
echo "========================================="

