#!/bin/bash

##############################################################################
# Focused Transformer VPL Parameter Sweep
# 
# Runs a focused set of experiments to test key hypotheses:
# 1. Effect of sequence length (5, 10, 15)
# 2. Effect of model capacity (small, medium, large)
# 3. Effect of KL regularization (0.05, 0.1, 0.2)
# 
# Total: ~18 experiments with 2 seeds each = 36 jobs
##############################################################################

echo "========================================"
echo "Focused Transformer VPL Parameter Sweep"
echo "========================================"
echo ""

# Create logs directory
mkdir -p logs

# Experiment 1: Sequence Length Ablation (fix other params)
echo "Experiment Set 1: Sequence Length (6 jobs)"
echo "  Testing: seq_length = 5, 10, 15"
echo "  Fixed: latent=512, hidden=512, heads=4, layers=2, beta=0.1"
echo ""
for seq in 5 10 15; do
    for seed in 0 1; do
        echo "  Submitting: seq=$seq, seed=$seed"
        sbatch --job-name="T_seq${seq}_s${seed}" run_transformer_vpl_prism.sh $seq 512 512 0.1 4 2 $seed
        sleep 0.3
    done
done

echo ""
echo "Experiment Set 2: Model Capacity (6 jobs)"
echo "  Testing: (heads=2,layers=1), (heads=4,layers=2), (heads=8,layers=3)"
echo "  Fixed: seq=10, latent=512, hidden=512, beta=0.1"
echo ""
# Small model
for seed in 0 1; do
    echo "  Submitting: small (heads=2, layers=1), seed=$seed"
    sbatch --job-name="T_small_s${seed}" run_transformer_vpl_prism.sh 10 512 512 0.1 2 1 $seed
    sleep 0.3
done
# Medium model (default)
for seed in 0 1; do
    echo "  Submitting: medium (heads=4, layers=2), seed=$seed"
    sbatch --job-name="T_medium_s${seed}" run_transformer_vpl_prism.sh 10 512 512 0.1 4 2 $seed
    sleep 0.3
done
# Large model
for seed in 0 1; do
    echo "  Submitting: large (heads=8, layers=3), seed=$seed"
    sbatch --job-name="T_large_s${seed}" run_transformer_vpl_prism.sh 10 512 512 0.1 8 3 $seed
    sleep 0.3
done

echo ""
echo "Experiment Set 3: KL Regularization (6 jobs)"
echo "  Testing: beta = 0.05, 0.1, 0.2"
echo "  Fixed: seq=10, latent=512, hidden=512, heads=4, layers=2"
echo ""
for beta in 0.05 0.1 0.2; do
    for seed in 0 1; do
        echo "  Submitting: beta=$beta, seed=$seed"
        sbatch --job-name="T_beta${beta}_s${seed}" run_transformer_vpl_prism.sh 10 512 512 $beta 4 2 $seed
        sleep 0.3
    done
done

echo ""
echo "Experiment Set 4: Latent Dimension (6 jobs)"
echo "  Testing: latent = 256, 512, 1024"
echo "  Fixed: seq=10, hidden=latent, heads=4, layers=2, beta=0.1"
echo ""
for latent in 256 512 1024; do
    for seed in 0 1; do
        echo "  Submitting: latent=$latent, seed=$seed"
        sbatch --job-name="T_lat${latent}_s${seed}" run_transformer_vpl_prism.sh 10 $latent $latent 0.1 4 2 $seed
        sleep 0.3
    done
done

echo ""
echo "Experiment Set 5: LSTM Baseline for Comparison (6 jobs)"
echo "  Running LSTM with same settings as medium Transformer"
echo "  This provides direct comparison"
echo ""
for seq in 5 10 15; do
    for seed in 0 1; do
        echo "  Submitting: LSTM seq=$seq, seed=$seed"
        sbatch --job-name="LSTM_seq${seq}_s${seed}" run_streaming_vpl_prism_slurm.sh $seq 512 512 0.1 $seed
        sleep 0.3
    done
done

echo ""
echo "========================================"
echo "Submitted 30 experiments (24 Transformer + 6 LSTM)"
echo "========================================"
echo ""
echo "Experiment Groups:"
echo "  - Sequence Length: 6 jobs"
echo "  - Model Capacity: 6 jobs"
echo "  - KL Regularization: 6 jobs"
echo "  - Latent Dimension: 6 jobs"
echo "  - LSTM Baseline: 6 jobs"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'squeue -u \$USER'"
echo ""
echo "View results on W&B:"
echo "  https://wandb.ai/\$(whoami)/streaming-vpl-prism"
echo ""
echo "Compare runs:"
echo "  - Group by 'seq_length' tag to see sequence length effect"
echo "  - Group by 'num_heads' to see capacity effect"
echo "  - Group by 'beta_max' to see regularization effect"
echo "  - Filter 'transformer' vs 'lstm' to compare architectures"
echo ""

