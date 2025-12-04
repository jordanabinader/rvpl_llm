#!/bin/bash
#
# Training script for Streaming VPL on UltraFeedback Dataset
# 
# This script runs experiments on the P_4_survey_100 dataset with different configurations.
# The dataset has 4 user types (1, 2, 4, 8) representing different preference patterns.
#

set -e  # Exit on error

# Default arguments
DATA_PATH="data_release/P_4_survey_100/gpt2"
DATA_SUBSET="all"  # Can be: 1, 2, 4, 8, or all
SEQ_LENGTH=10
LATENT_DIM=64
HIDDEN_DIM=256
BETA_MAX=0.1
BETA_CYCLES=4
TEMPORAL_GAMMA=1.1
FREE_BITS=6.4
NUM_EPOCHS=20
BATCH_SIZE=8
LEARNING_RATE=1e-4
SEED=0
LOG_DIR="experiments/streaming_vpl_ultrafeedback"

# Use transformer or recurrent architecture
USE_TRANSFORMER=false
NUM_ATTENTION_HEADS=4
NUM_TRANSFORMER_LAYERS=2

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_subset)
            DATA_SUBSET="$2"
            shift 2
            ;;
        --seq_length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --latent_dim)
            LATENT_DIM="$2"
            shift 2
            ;;
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --beta_max)
            BETA_MAX="$2"
            shift 2
            ;;
        --beta_cycles)
            BETA_CYCLES="$2"
            shift 2
            ;;
        --temporal_gamma)
            TEMPORAL_GAMMA="$2"
            shift 2
            ;;
        --free_bits)
            FREE_BITS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --use_transformer)
            USE_TRANSFORMER=true
            shift
            ;;
        --num_attention_heads)
            NUM_ATTENTION_HEADS="$2"
            shift 2
            ;;
        --num_transformer_layers)
            NUM_TRANSFORMER_LAYERS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Build run name
if [ "$USE_TRANSFORMER" = true ]; then
    ARCH="transformer"
else
    ARCH="recurrent"
fi

RUN_NAME="${DATA_SUBSET}_seq${SEQ_LENGTH}_latent${LATENT_DIM}_hidden${HIDDEN_DIM}_beta${BETA_MAX}_fb${FREE_BITS}_cycles${BETA_CYCLES}_gamma${TEMPORAL_GAMMA}_${ARCH}_seed${SEED}"

echo "="
echo "Starting Streaming VPL Training on UltraFeedback"
echo "="
echo "Data subset: $DATA_SUBSET"
echo "Sequence length: $SEQ_LENGTH"
echo "Architecture: $ARCH"
echo "Latent dim: $LATENT_DIM"
echo "Hidden dim: $HIDDEN_DIM"
echo "Beta max: $BETA_MAX"
echo "Free bits: $FREE_BITS"
echo "Cyclical cycles: $BETA_CYCLES"
echo "Temporal gamma: $TEMPORAL_GAMMA"
echo "Seed: $SEED"
echo "Run name: $RUN_NAME"
echo "="

# Build command
CMD="python -m hidden_context.train_streaming_vpl \
    --data_path $DATA_PATH \
    --data_subset $DATA_SUBSET \
    --seq_length $SEQ_LENGTH \
    --latent_dim $LATENT_DIM \
    --hidden_dim $HIDDEN_DIM \
    --beta_max $BETA_MAX \
    --beta_cycles $BETA_CYCLES \
    --temporal_gamma $TEMPORAL_GAMMA \
    --free_bits $FREE_BITS \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --seed $SEED \
    --log_dir $LOG_DIR \
    --bf16 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 1000"

# Add transformer-specific arguments if needed
if [ "$USE_TRANSFORMER" = true ]; then
    CMD="$CMD --use_transformer --num_attention_heads $NUM_ATTENTION_HEADS --num_transformer_layers $NUM_TRANSFORMER_LAYERS"
fi

# Run training
echo "Running: $CMD"
eval $CMD

echo ""
echo "Training complete! Results saved to: $LOG_DIR/$RUN_NAME"
echo ""

