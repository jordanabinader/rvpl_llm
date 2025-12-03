#!/bin/bash

#########################
# Slurm job parameters  #
#########################

#SBATCH -J streaming_vpl_prism             # Job name
#SBATCH -p mit_normal_gpu                   # GPU partition
#SBATCH -c 4                                # CPU cores
#SBATCH --mem=32G                           # Memory
#SBATCH -t 2:00:00                         # Time limit
#SBATCH -G 1                                # 1 GPU
#SBATCH -o logs/streaming_vpl_prism_%j.out # STDOUT log
#SBATCH -e logs/streaming_vpl_prism_%j.err # STDERR log
#########################
# Environment setup     #
#########################

<<<<<<< Updated upstream
# Create logs directory
mkdir -p logs
export WANDB_MODE=disabled

# 1. Load the base system python
module load miniforge

# 2. Create the virtual environment (if it doesn't exist)
# We use 'venv' (standard spelling), not 'vnev'
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
=======
# W&B Configuration
export WANDB_MODE=online
export WANDB_PROJECT="streaming-vpl-prism"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source vnev/bin/activate
>>>>>>> Stashed changes
fi

# 3. Activate the environment (Fixing the typo here: vnev -> venv)
source venv/bin/activate

# 4. CRITICAL: Ensure we rely ONLY on this venv, ignoring local user packages
export PYTHONNOUSERSITE=1

# 5. Upgrade pip inside the virtual environment
# python -m pip install --upgrade pip

# 6. Install packages
# NOTE: We REMOVED 'pip uninstall'. You cannot uninstall system packages.
# Instead, we use --ignore-installed to force installing a fresh copy 
# into your local venv, shadowing the system version.

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
# Training              #
#########################

# Configuration
DATA_PATH="data_release/prism/gpt2"
SEQ_LENGTH=${1:-10}
LATENT_DIM=${2:-512}
BETA=${3:-0.1}
SEED=${4:-0}

# Check if data exists
if [ ! -f "$DATA_PATH/train.jsonl" ]; then
    echo "Error: PRISM data not found at $DATA_PATH"
    echo "Please run: bash generate_prism_embeddings.sh first"
    exit 1
fi

# Create experiment name
EXP_NAME="prism_seq${SEQ_LENGTH}_latent${LATENT_DIM}_beta${BETA}_seed${SEED}"
OUTPUT_DIR="experiments/streaming_vpl_prism/$EXP_NAME"

echo "Configuration:"
echo "  Data Path: $DATA_PATH"
echo "  Sequence Length: $SEQ_LENGTH"
echo "  Latent Dimension: $LATENT_DIM"
echo "  Beta: $BETA"
echo "  Seed: $SEED"
echo "  Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
python -m hidden_context.train_streaming_vpl \
    --data_path $DATA_PATH \
    --seq_length $SEQ_LENGTH \
    --latent_dim $LATENT_DIM \
    --hidden_dim 512 \
    --encoder_embed_dim 768 \
    --decoder_embed_dim 768 \
    --beta_max $BETA \
    --beta_cycles 4 \
    --temporal_gamma 1.1 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 10 \
    --eval_steps 500 \
    --save_steps 500 \
    --log_dir $OUTPUT_DIR \
    --seed $SEED

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To evaluate, run:"
echo "  python -m hidden_context.evaluate_streaming_vpl --checkpoint_dir $OUTPUT_DIR/final_checkpoint"

