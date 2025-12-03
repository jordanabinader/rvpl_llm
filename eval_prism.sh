#!/bin/bash

# Quick evaluation script for PRISM models

if [ -z "$1" ]; then
    echo "Usage: bash eval_prism.sh <experiment_dir>"
    echo ""
    echo "Example:"
    echo "  bash eval_prism.sh experiments/streaming_vpl_prism/prism_seq10_latent512_hidden512_beta0.1_cycles4_gamma1.1_seed0"
    echo ""
    echo "Or auto-find latest experiment:"
    echo "  bash eval_prism.sh auto"
    exit 1
fi

# Activate virtual environment if needed
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d "vnev" ]; then
        source vnev/bin/activate
    fi
fi

# W&B Configuration
export WANDB_MODE=online
export WANDB_PROJECT="streaming-vpl-prism"

# Find experiment directory
if [ "$1" == "auto" ]; then
    EXP_DIR=$(ls -dt experiments/streaming_vpl_prism/prism_* 2>/dev/null | head -1)
    if [ -z "$EXP_DIR" ]; then
        echo "Error: No PRISM experiments found in experiments/streaming_vpl_prism/"
        exit 1
    fi
    echo "Auto-detected experiment: $EXP_DIR"
else
    EXP_DIR="$1"
fi

# Find the model checkpoint
MODEL_PATH=$(find "$EXP_DIR" -name "model.pt" -path "*/final_checkpoint/*" | head -1)

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Could not find model.pt in $EXP_DIR/*/final_checkpoint/"
    echo ""
    echo "Available checkpoints:"
    find "$EXP_DIR" -name "model.pt" 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "========================================="
echo "PRISM Model Evaluation"
echo "========================================="
echo "Model: $MODEL_PATH"
echo "Data: data_release/prism/gpt2"
echo "========================================="
echo ""

# Extract parameters from directory name (or use defaults)
SEQ_LENGTH=10
LATENT_DIM=512
HIDDEN_DIM=512

# Create output directory
EVAL_DIR="$(dirname $(dirname $MODEL_PATH))/evaluation"
mkdir -p "$EVAL_DIR"

# Run evaluation
python -m hidden_context.evaluate_streaming_vpl \
    --model_path "$MODEL_PATH" \
    --data_path data_release/prism/gpt2 \
    --seq_length $SEQ_LENGTH \
    --latent_dim $LATENT_DIM \
    --hidden_dim $HIDDEN_DIM \
    --encoder_embed_dim 768 \
    --decoder_embed_dim 768 \
    --num_eval_episodes 200 \
    --output_dir "$EVAL_DIR" \
    --seed 0

echo ""
echo "========================================="
echo "Evaluation Complete!"
echo "========================================="
echo "Results saved to: $EVAL_DIR"
echo ""
echo "View results:"
echo "  cat $EVAL_DIR/results.json"
echo "  open $EVAL_DIR/adaptation_curve.png"
echo ""
echo "View on W&B:"
echo "  https://wandb.ai/your-username/streaming-vpl-prism"
echo "========================================="

