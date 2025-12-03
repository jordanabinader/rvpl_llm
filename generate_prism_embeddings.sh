#!/bin/bash


#SBATCH -J generate_prism_embeddings             # Job name
#SBATCH -p mit_normal_gpu                   # GPU partition
#SBATCH -c 4                                # CPU cores
#SBATCH --mem=32G                           # Memory
#SBATCH -t 2:00:00                         # Time limit
#SBATCH -G 1                                # 1 GPU
#SBATCH -o logs/generate_prism_embeddings_%j.out # STDOUT log
#SBATCH -e logs/generate_prism_embeddings_%j.err # STDERR log
#########################
# Environment setup     #
#########################

echo "========================================="
echo "PRISM Dataset Preprocessing"
echo "========================================="
echo ""

export WANDB_MODE=disabled
# Load miniforge module
module load miniforge

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Ensure we rely ONLY on this venv, ignoring local user packages
export PYTHONNOUSERSITE=1

# Upgrade pip
python -m pip install --upgrade pip


echo "Installing dependencies..."
python -m pip install --upgrade --ignore-installed --no-cache-dir \
    "torch>=2.0" \
    "transformers==4.40.0" \
    "tokenizers==0.19.1" \
    "peft==0.10.0" \
    "accelerate>=0.27.0" \
    "numpy" \
    "sentencepiece" \
    "datasets" \
    "wandb" \
    "matplotlib" \
    "scipy" \
    "scikit-learn" 

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

