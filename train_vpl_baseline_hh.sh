#!/bin/bash
#SBATCH --job-name=vpl_baseline_prism
#SBATCH --output=logs/vpl_baseline_prism_%j.out
#SBATCH --error=logs/vpl_baseline_prism_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

# Environment setup
export WANDB_MODE=online
export WANDB_PROJECT=streaming-vpl-prism
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Activate virtual environment
source venv/bin/activate

# Model and data configuration
MODEL_NAME='gpt2'
DATA_PATH="data_release/prism/gpt2"
DATA_SUBSET="all"
LOG_DIR="experiments/vpl_baseline_prism"

# Train original VPL baseline on PRISM dataset
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

