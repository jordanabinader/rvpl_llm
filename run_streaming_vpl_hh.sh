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
module load miniforge

# Activate your Python environment
source vnev/bin/activate
# Avoid Rust build failures for tokenizers
python -m pip install "tokenizers==0.13.3"

# Core HF stack
python -m pip install "torch" "transformers[torch]" accelerate

# Extra utilities
python -m pip install numpy sentencepiece datasets

# Logging
python -m pip install wandb

echo "Node: $(hostname)"
echo "GPUs visible to this job:"
nvidia-smi || echo "nvidia-smi not found"
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