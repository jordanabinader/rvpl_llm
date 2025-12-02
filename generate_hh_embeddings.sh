#!/bin/bash

#########################
# Slurm job parameters  #
#########################

#SBATCH -J hh_embed_gpu                 # Job name
#SBATCH -p mit_normal_gpu               # GPU partition
#SBATCH -c 4                            # CPU cores
#SBATCH --mem=32G                       # Memory (tune if needed)
#SBATCH -t 04:00:00                     # Time limit (HH:MM:SS)
#SBATCH -G 1                            # 1 GPU (default: L40S)

#SBATCH -o logs/hh_embed_gpu_%j.out     # STDOUT
#SBATCH -e logs/hh_embed_gpu_%j.err     # STDERR

#########################
# Environment setup     #
#########################

module load miniforge                   # or your Python module
# conda activate rvpl_llm               # if you have a conda env

echo "Node: $(hostname)"
echo "GPUs visible to this job:"
nvidia-smi || echo "nvidia-smi not found"

#########################
# Actual workload       #
#########################

echo "========================================="
echo "Step 1: Generate embeddings for 'both' (helpful + harmless)"
echo "========================================="

python generate_hh_embeddings.py \
    --input_dir data/relabeled_hh_rlhf \
    --output_dir data_release/hh_rlhf/gpt2 \
    --data_subset both \
    --embed_dim 768 \
    --max_length 512 \
    --batch_size 8

echo ""
echo "========================================="
echo "Done! Data saved to: data_release/hh_rlhf/gpt2/both/"
echo "========================================="
