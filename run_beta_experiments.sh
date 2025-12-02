#!/bin/bash

#########################
# Slurm job parameters  #
#########################

#SBATCH -J beta_experiments             # Job name
#SBATCH -p mit_normal_gpu                   # GPU partition
#SBATCH -c 4                                # CPU cores
#SBATCH --mem=32G                           # Memory
#SBATCH -t 8:00:00                         # Time limit (longer for 2 experiments)
#SBATCH -G 1                                # 1 GPU
#SBATCH -o logs/beta_experiments_%j.out # STDOUT log
#SBATCH -e logs/beta_experiments_%j.err # STDERR log

#########################
# Beta Experiments      #
# Test beta=0.001 and   #
# beta=0.005 with       #
# proper free bits      #
#########################

# Create logs directory
mkdir -p logs
export WANDB_MODE=disabled

# 1. Load the base system python
module load miniforge

# 2. Create the virtual environment (if it doesn't exist)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# 3. Activate the environment
source venv/bin/activate

# 4. CRITICAL: Ensure we rely ONLY on this venv, ignoring local user packages
export PYTHONNOUSERSITE=1

# 5. Upgrade pip inside the virtual environment
python -m pip install --upgrade pip

# 6. Install packages
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
    "scikit-learn" \
    "Pillow"

echo "Node: $(hostname)"
echo "GPUs visible to this job:"
nvidia-smi || echo "nvidia-smi not found"

# Verify versions
echo "Package versions in venv:"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import peft; print(f'peft: {peft.__version__}')"
python -c "import torch; print(f'torch: {torch.__version__}')"

#########################
# Experiment Parameters #
#########################
DATA_SUBSET="both"
SEQ_LENGTH=10
LATENT_DIM=64
HIDDEN_DIM=256
SEED=0
FREE_BITS=6.4
CYCLES=4
GAMMA=1.1

# Loop over beta values
for BETA_MAX in 0.001 0.005; do
    echo ""
    echo "========================================================================"
    echo "EXPERIMENT: BETA = ${BETA_MAX}"
    echo "========================================================================"
    
    # Updated Experiment Name
    EXP_NAME="${DATA_SUBSET}_seq${SEQ_LENGTH}_latent${LATENT_DIM}_hidden${HIDDEN_DIM}_beta${BETA_MAX}_freebits${FREE_BITS}_cycles${CYCLES}_gamma${GAMMA}_seed${SEED}"
    OUTPUT_DIR="experiments/streaming_vpl_hh/${EXP_NAME}"
    
    echo "Output directory: ${OUTPUT_DIR}"
    
    echo ""
    echo "========================================="
    echo "Step 1: Training with Beta=${BETA_MAX}"
    echo "========================================="
    
    python -m hidden_context.train_streaming_vpl \
        --data_path data_release/hh_rlhf/gpt2 \
        --data_subset ${DATA_SUBSET} \
        --seq_length ${SEQ_LENGTH} \
        --latent_dim ${LATENT_DIM} \
        --hidden_dim ${HIDDEN_DIM} \
        --beta_max ${BETA_MAX} \
        --beta_cycles ${CYCLES} \
        --temporal_gamma ${GAMMA} \
        --free_bits ${FREE_BITS} \
        --learning_rate 1e-4 \
        --num_train_epochs 20 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --eval_steps 100 \
        --save_steps 1000 \
        --log_dir experiments/streaming_vpl_hh \
        --seed ${SEED} \
        --bf16 True
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for beta=${BETA_MAX}"
        continue
    fi
    
    echo ""
    echo "========================================="
    echo "Step 2: Diagnosing Checkpoints"
    echo "========================================="
    
    # Diagnose each checkpoint
    for CKPT_DIR in ${OUTPUT_DIR}/checkpoint-* ${OUTPUT_DIR}/final_checkpoint; do
        if [ -d "${CKPT_DIR}" ] && [ -f "${CKPT_DIR}/model.pt" ]; then
            CKPT_NAME=$(basename ${CKPT_DIR})
            echo "Diagnosing ${CKPT_NAME}..."
            
            python diagnose_adaptation.py \
                --model_path ${CKPT_DIR}/model.pt \
                --data_path data_release/hh_rlhf/gpt2 \
                --latent_dim ${LATENT_DIM} \
                --hidden_dim ${HIDDEN_DIM} \
                --seq_length ${SEQ_LENGTH}
            
            # Rename the output to include checkpoint name
            if [ -f "experiments/latent_diagnosis_hh.png" ]; then
                mv experiments/latent_diagnosis_hh.png ${OUTPUT_DIR}/latent_diagnosis_${CKPT_NAME}.png
                echo "Saved: ${OUTPUT_DIR}/latent_diagnosis_${CKPT_NAME}.png"
            fi
        fi
    done
    
    echo ""
    echo "========================================="
    echo "Step 3: Full Evaluation with Thompson Sampling"
    echo "========================================="
    
    python -m hidden_context.evaluate_streaming_vpl \
        --model_path ${OUTPUT_DIR}/final_checkpoint/model.pt \
        --data_path data_release/hh_rlhf/gpt2 \
        --data_subset ${DATA_SUBSET} \
        --seq_length ${SEQ_LENGTH} \
        --latent_dim ${LATENT_DIM} \
        --hidden_dim ${HIDDEN_DIM} \
        --num_eval_episodes 200 \
        --output_dir ${OUTPUT_DIR}/evaluation \
        --seed ${SEED}
    
    echo ""
    echo "========================================="
    echo "COMPLETED: Beta=${BETA_MAX}"
    echo "Results saved to: ${OUTPUT_DIR}/evaluation/"
    echo "========================================="
done

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================================================"
echo "Run 'python compare_beta_experiments.py' to generate comparison plots"
echo "Run 'python generate_beta_report.py' to generate summary report"
echo "========================================================================"

