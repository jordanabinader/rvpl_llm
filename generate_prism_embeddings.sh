#!/bin/bash

# Generate embeddings for PRISM alignment dataset
# This script processes the PRISM dataset and creates train/test splits with embeddings

echo "========================================="
echo "PRISM Dataset Preprocessing"
echo "========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source vnev/bin/activate
fi

# Model type (gpt2 or llama)
MODEL_TYPE=${1:-"gpt2"}
EMBED_DIM=${2:-1024}
MIN_RATING_DIFF=${3:-10.0}

echo "Model Type: $MODEL_TYPE"
echo "Embedding Dimension: $EMBED_DIM"
echo "Minimum Rating Difference: $MIN_RATING_DIFF"
echo ""

# Run preprocessing
python generate_prism_embeddings.py \
    --output_dir data_release/prism \
    --model_type $MODEL_TYPE \
    --embed_dim $EMBED_DIM \
    --max_length 1024 \
    --train_test_split 0.8 \
    --min_rating_diff $MIN_RATING_DIFF \
    --seed 0

echo ""
echo "========================================="
echo "Preprocessing Complete!"
echo "========================================="
echo "Output: data_release/prism/$MODEL_TYPE/"
echo ""
echo "To train with this data, run:"
echo "  bash run_streaming_vpl_prism.sh"

