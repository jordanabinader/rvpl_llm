#!/bin/bash

#########################
# Slurm job parameters  #
#########################

#SBATCH -J streaming_vpl_hh             # Job name
#SBATCH -p mit_normal_gpu                   # GPU partition
#SBATCH -c 4                                # CPU cores
#SBATCH --mem=32G                           # Memory
#SBATCH -t 2:00:00                         # Time limit
#SBATCH -G 1                                # 1 GPU
#SBATCH -o logs/streaming_vpl_hh_%j.out # STDOUT log
#SBATCH -e logs/streaming_vpl_hh_%j.err # STDERR log
#########################
# Environment setup     #
#########################

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
fi

# 3. Activate the environment (Fixing the typo here: vnev -> venv)
source venv/bin/activate

# 4. CRITICAL: Ensure we rely ONLY on this venv, ignoring local user packages
export PYTHONNOUSERSITE=1

# 5. Upgrade pip inside the virtual environment
python -m pip install --upgrade pip

# 6. Install packages
# NOTE: We REMOVED 'pip uninstall'. You cannot uninstall system packages.
# Instead, we use --ignore-installed to force installing a fresh copy 
# into your local venv, shadowing the system version.

echo "Installing dependencies..."
python -m pip install --upgrade --ignore-installed --no-cache-dir \
    "torch>=2.0" \
    "transformers==4.40.0" \
    "tokenizers==0.19.1" \
    "peft==0.10.0" \
    "accelerate>=0.27.0" \
    "numpy" \
    "sentencepiece" \
    "datasets" \
    "wandb" \
    "matplotlib" \
    "scipy" \
    "scikit-learn" 

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
DATA_SUBSET="both"
SEQ_LENGTH=10
LATENT_DIM=64      # ← CHANGED from 512 (Fix #4: Inverted Funnel)
HIDDEN_DIM=256     # ← ADDED (Fix #4: Inverted Funnel)
SEED=0

# NEW HYPERPARAMETERS
BETA_MAX=0.1
CYCLES=4
GAMMA=1.1

# Updated Experiment Name to track new params
EXP_NAME="${DATA_SUBSET}_seq${SEQ_LENGTH}_latent${LATENT_DIM}_hidden${HIDDEN_DIM}_beta${BETA_MAX}_cycles${CYCLES}_gamma${GAMMA}_seed${SEED}"
OUTPUT_DIR="experiments/streaming_vpl_hh/${EXP_NAME}"

echo "========================================="
echo "Step 1: Training Streaming VPL (With Fixes)"
echo "========================================="

python -m hidden_context.train_streaming_vpl \
    --data_path data_release/hh_rlhf/gpt2 \
    --data_subset ${DATA_SUBSET} \
    --seq_length ${SEQ_LENGTH} \
    --latent_dim ${LATENT_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --beta_max ${BETA_MAX} \
    --beta_cycles ${CYCLES} \
    --temporal_gamma ${GAMMA} \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --eval_steps 100 \
    --save_steps 500 \
    --log_dir experiments/streaming_vpl_hh \
    --seed ${SEED} \
    --bf16 True

echo ""
echo "========================================="
echo "Step 2: Evaluation (With Thompson Sampling)"
echo "=========================================="

python -m hidden_context.evaluate_streaming_vpl \
    --model_path ${OUTPUT_DIR}/final_checkpoint/model.pt \
    --data_path data_release/hh_rlhf/gpt2 \
    --data_subset ${DATA_SUBSET} \
    --seq_length ${SEQ_LENGTH} \
    --latent_dim ${LATENT_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --num_eval_episodes 200 \
    --output_dir ${OUTPUT_DIR}/evaluation \
    --seed ${SEED}
echo ""
echo "========================================="
echo "Complete!"
echo "Results saved to: ${OUTPUT_DIR}/evaluation/"
echo "========================================="