#!/bin/bash

##############################################################################
# Transformer VPL Parameter Sweep
# 
# This script runs multiple Transformer experiments with different configurations
# and logs each to W&B for comparison.
##############################################################################

# Configuration arrays
SEQ_LENGTHS=(5 10 15)
LATENT_DIMS=(256 512)
NUM_HEADS=(4 8)
NUM_LAYERS=(2 3)
BETA_VALUES=(0.05 0.1 0.2)
SEEDS=(0 1 2)

# Function to submit a single experiment
submit_experiment() {
    local seq=$1
    local latent=$2
    local heads=$3
    local layers=$4
    local beta=$5
    local seed=$6
    
    # Hidden dim = latent dim for simplicity
    local hidden=$latent
    
    # Create unique experiment name
    local exp_name="transformer_seq${seq}_lat${latent}_h${heads}_l${layers}_b${beta}_s${seed}"
    
    echo "Submitting: $exp_name"
    
    # Submit SLURM job
    sbatch --job-name=$exp_name run_transformer_vpl_prism.sh $seq $latent $hidden $beta $heads $layers $seed
}

# Main experiment sweep
echo "========================================"
echo "Transformer VPL Parameter Sweep"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Sequence Lengths: ${SEQ_LENGTHS[@]}"
echo "  Latent Dimensions: ${LATENT_DIMS[@]}"
echo "  Attention Heads: ${NUM_HEADS[@]}"
echo "  Transformer Layers: ${NUM_LAYERS[@]}"
echo "  Beta Values: ${BETA_VALUES[@]}"
echo "  Seeds: ${SEEDS[@]}"
echo ""

# Count total experiments
total_exps=0
for seq in "${SEQ_LENGTHS[@]}"; do
    for latent in "${LATENT_DIMS[@]}"; do
        for heads in "${NUM_HEADS[@]}"; do
            for layers in "${NUM_LAYERS[@]}"; do
                for beta in "${BETA_VALUES[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        ((total_exps++))
                    done
                done
            done
        done
    done
done

echo "Total experiments to run: $total_exps"
echo ""
read -p "Proceed? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Submitting jobs..."
echo ""

# Submit all experiments
submitted=0
for seq in "${SEQ_LENGTHS[@]}"; do
    for latent in "${LATENT_DIMS[@]}"; do
        for heads in "${NUM_HEADS[@]}"; do
            for layers in "${NUM_LAYERS[@]}"; do
                for beta in "${BETA_VALUES[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        submit_experiment $seq $latent $heads $layers $beta $seed
                        ((submitted++))
                        
                        # Add small delay to avoid overwhelming scheduler
                        sleep 0.5
                    done
                done
            done
        done
    done
done

echo ""
echo "========================================"
echo "Submitted $submitted experiments"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "View results on W&B:"
echo "  https://wandb.ai/\$(whoami)/streaming-vpl-prism"
echo ""

