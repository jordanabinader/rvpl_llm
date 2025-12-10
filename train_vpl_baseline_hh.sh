#!/bin/bash
#SBATCH -J train_vpl_baseline_hh             # Job name
#SBATCH -p mit_normal_gpu                 # GPU partition
#SBATCH -c 4                              # CPU cores
#SBATCH --mem=32G                         # Memory
#SBATCH -t 0:30:00                        # Time limit (30 min)
#SBATCH -G 1                              # 1 GPU
#SBATCH -o logs/train_vpl_baseline_hh_%j.out # STDOUT log
#SBATCH -e logs/train_vpl_baseline_hh_%j.err # STDERR log


# Environment setup
export WANDB_MODE=online
export WANDB_PROJECT=vpl-baseline-hh

# Activate virtual environment
source venv/bin/activate

# Model and data configuration
MODEL_NAME='gpt2'
DATA_PATH="data_release/hh_rlhf/gpt2/"
DATA_SUBSET="both"  # 'harmless', 'helpful', or 'both'
LOG_DIR="experiments/vpl_baseline_hh"

# Train original VPL baseline on HH-RLHF dataset
python -m hidden_context.train_llm_vae_preference_model \
    --model_name=${MODEL_NAME} \
    --data_path=${DATA_PATH} \
    --data_subset=${DATA_SUBSET} \
    --num_train_epochs=2 \
    --reward_model_type=vae \
    --log_dir=${LOG_DIR} \
    --bf16 True \
    --fp16 False \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --latent_dim 512 \
    --hidden_dim 512 \
    --learning_rate 1e-4 \
    --use_annealing True \
    --kl_loss_weight 0 \
    --seed 0

echo "Training complete!"

