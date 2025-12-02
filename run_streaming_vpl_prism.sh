#!/bin/bash

# Train Streaming VPL on PRISM alignment dataset
# This uses real user conversations with natural sequential interactions

echo "========================================="
echo "Streaming VPL Training on PRISM Dataset"
echo "========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source vnev/bin/activate
fi

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
    --dataset_type prism \
    --seq_length $SEQ_LENGTH \
    --latent_dim $LATENT_DIM \
    --hidden_dim 512 \
    --embed_dim 1024 \
    --num_layers 2 \
    --beta $BETA \
    --learning_rate 1e-4 \
    --batch_size 32 \
    --num_epochs 10 \
    --eval_every 500 \
    --save_every 500 \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --wandb_project "streaming-vpl-prism" \
    --wandb_name $EXP_NAME

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To evaluate, run:"
echo "  python -m hidden_context.evaluate_streaming_vpl --checkpoint_dir $OUTPUT_DIR/final_checkpoint"

