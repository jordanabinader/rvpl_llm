#!/bin/bash
#
#SBATCH -J streaming_vpl_ultrafeedback            # Job name
#SBATCH -p mit_normal_gpu                   # GPU partition
#SBATCH -c 4                                # CPU cores
#SBATCH --mem=32G                           # Memory
#SBATCH -t 4:00:00                          # Time limit
#SBATCH -G 1                                # 1 GPU
#SBATCH -o logs/streaming_vpl_ultrafeedback_%j.out # STDOUT log
#SBATCH -e logs/streaming_vpl_ultrafeedback_%j.err # STDERR log


# Create logs directory
mkdir -p logs

# W&B Configuration
export WANDB_MODE=online
export WANDB_PROJECT="streaming-vpl-prism"
export WANDB_LOG_MODEL="false"
export WANDB_WATCH="false"

# 1. Load the base system python
module load miniforge

# 2. Create the virtual environment (if it doesn't exist)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# 3. Activate the environment
source venv/bin/activate

# 4. Ensure we rely ONLY on this venv
export PYTHONNOUSERSITE=1

# 5. Upgrade pip
python -m pip install --upgrade pip

echo "Node: $(hostname)"
echo "GPUs visible to this job:"
nvidia-smi || echo "nvidia-smi not found"

# Verify versions
echo "Package versions in venv:"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}')"


# Default arguments
DATA_PATH="data_release/P_4_survey_100/gpt2"
DATA_SUBSET="all"  # Can be: 1, 2, 4, 8, or all
SEQ_LENGTH=10
LATENT_DIM=64
HIDDEN_DIM=256
BETA_MAX=0.12
BETA_CYCLES=4
TEMPORAL_GAMMA=1.1
FREE_BITS=8.0
NUM_EPOCHS=20
BATCH_SIZE=8
LEARNING_RATE=1e-4
SEED=0
LOG_DIR="experiments/streaming_vpl_ultrafeedback"

# Use transformer or recurrent architecture
USE_TRANSFORMER=true
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

# Build run name (must match train_streaming_vpl.py output_dir construction)
# Note: We don't include architecture in the path - train_streaming_vpl.py doesn't either
RUN_NAME="${DATA_SUBSET}_seq${SEQ_LENGTH}_latent${LATENT_DIM}_hidden${HIDDEN_DIM}_beta${BETA_MAX}_freebits${FREE_BITS}_cycles${BETA_CYCLES}_gamma${TEMPORAL_GAMMA}_seed${SEED}"

# Determine architecture string for logging
if [ "$USE_TRANSFORMER" = true ]; then
    ARCH="transformer"
else
    ARCH="recurrent"
fi

# Create unique run ID (uses SLURM job ID if available, otherwise "local")
RUN_ID_BASE="slurm_${SLURM_JOB_ID:-local}"

# Full output directory
OUTPUT_DIR="${LOG_DIR}/${RUN_NAME}/${RUN_ID_BASE}"

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
echo "Run ID: $RUN_ID_BASE"
echo "Output dir: $OUTPUT_DIR"
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
    --run_id $RUN_ID_BASE \
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
echo "================================"
echo "Training complete!"
echo "================================"
echo ""
echo "Model saved to: ${OUTPUT_DIR}/final_checkpoint/model.pt"
echo ""
echo "To evaluate this model, run:"
echo ""
if [ "$USE_TRANSFORMER" = true ]; then
    python -m hidden_context.evaluate_streaming_vpl  \
    --model_path ${OUTPUT_DIR}/final_checkpoint/model.pt \
    --data_path $DATA_PATH \
    --data_subset $DATA_SUBSET \
    --seq_length $SEQ_LENGTH \
    --latent_dim $LATENT_DIM \
    --hidden_dim $HIDDEN_DIM \
    --use_transformer \
    --num_attention_heads $NUM_ATTENTION_HEADS \
    --num_transformer_layers $NUM_TRANSFORMER_LAYERS \
    --num_eval_episodes 500 \
    --seed $SEED
else
    python -m hidden_context.evaluate_streaming_vpl \
    --model_path ${OUTPUT_DIR}/final_checkpoint/model.pt \
    --data_path $DATA_PATH \
    --data_subset $DATA_SUBSET \
    --seq_length $SEQ_LENGTH \
    --latent_dim $LATENT_DIM \
    --hidden_dim $HIDDEN_DIM \
    --num_eval_episodes 500 \
    --seed $SEED
fi
echo ""
echo "================================"
echo ""