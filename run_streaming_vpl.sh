#!/bin/bash

# Shell script for running Streaming VPL experiments
# Trains a RecurrentVAEModel on Pets (Divergent) dataset and evaluates adaptation

set -e  # Exit on error

# Determine Python executable
if [ -f "vnev/bin/python" ]; then
    PYTHON="vnev/bin/python"
    echo "Using Python from vnev..."
elif [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
    echo "Using Python from venv..."
else
    PYTHON="python3"
    echo "Warning: No virtual environment found, using system Python"
fi

echo "=========================================="
echo "Streaming VPL Experiment"
echo "=========================================="

# Configuration
DATA_PATH="data_release/simple_pets/gpt2"
DATA_SUBSET="both"  # "both" = Dog + Cat Lovers (required for adaptation!), "harmless" = Cat only, "helpful" = Dog only
MODEL_NAME="gpt2"
SEQ_LENGTH=10
LATENT_DIM=512
HIDDEN_DIM=512
EMBED_DIM=768  # GPT-2 embedding dimension

# KL Annealing parameters (critical for preventing posterior collapse)
BETA_START=0.0
BETA_END=0.1
BETA_ANNEAL_EPOCHS=5

# Training hyperparameters
LEARNING_RATE=1e-4
NUM_EPOCHS=20
BATCH_SIZE=8
GRADIENT_ACCUM=1
EPOCH_SIZE=1000

# Output directory
LOG_DIR="experiments/streaming_vpl"
SEED=0

# Create output directory
mkdir -p ${LOG_DIR}

echo ""
echo "Configuration:"
echo "  Data: ${DATA_PATH}/${DATA_SUBSET}"
echo "  Seq Length: ${SEQ_LENGTH}"
echo "  Latent Dim: ${LATENT_DIM}"
echo "  KL Annealing: ${BETA_START} → ${BETA_END} over ${BETA_ANNEAL_EPOCHS} epochs"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Seed: ${SEED}"
echo ""

# Training
echo "=========================================="
echo "Step 1: Training"
echo "=========================================="

$PYTHON -m hidden_context.train_streaming_vpl \
    --data_path ${DATA_PATH} \
    --data_subset ${DATA_SUBSET} \
    --encoder_embed_dim ${EMBED_DIM} \
    --decoder_embed_dim ${EMBED_DIM} \
    --latent_dim ${LATENT_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --seq_length ${SEQ_LENGTH} \
    --epoch_size ${EPOCH_SIZE} \
    --beta_start ${BETA_START} \
    --beta_end ${BETA_END} \
    --beta_anneal_epochs ${BETA_ANNEAL_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUM} \
    --eval_steps 100 \
    --save_steps 500 \
    --log_dir ${LOG_DIR} \
    --seed ${SEED} \
    --bf16 True \
    --gradient_checkpointing False

# Find the trained model path
TRAINED_MODEL_DIR="${LOG_DIR}/${DATA_SUBSET}_seq${SEQ_LENGTH}_latent${LATENT_DIM}_beta${BETA_END}_seed${SEED}"
MODEL_PATH="${TRAINED_MODEL_DIR}/final_checkpoint/model.pt"

echo ""
echo "Training complete! Model saved to: ${MODEL_PATH}"
echo ""

# Evaluation
echo "=========================================="
echo "Step 2: Evaluation"
echo "=========================================="

EVAL_OUTPUT_DIR="${TRAINED_MODEL_DIR}/evaluation"

$PYTHON -m hidden_context.evaluate_streaming_vpl \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --data_subset ${DATA_SUBSET} \
    --encoder_embed_dim ${EMBED_DIM} \
    --decoder_embed_dim ${EMBED_DIM} \
    --latent_dim ${LATENT_DIM} \
    --hidden_dim ${HIDDEN_DIM} \
    --seq_length ${SEQ_LENGTH} \
    --num_eval_episodes 200 \
    --batch_size 16 \
    --output_dir ${EVAL_OUTPUT_DIR} \
    --seed ${SEED}

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Results saved to: ${EVAL_OUTPUT_DIR}"
echo "  - adaptation_curve.png: Visualization of adaptation over time"
echo "  - results.json: Numerical results"
echo ""
echo "Check the adaptation curve to verify that accuracy improves"
echo "from ~55% at t=0 to ~75%+ at t=${SEQ_LENGTH}."
echo ""

