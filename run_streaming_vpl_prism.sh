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

