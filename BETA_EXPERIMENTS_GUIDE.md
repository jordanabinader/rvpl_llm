# Beta Experiments Implementation Guide

## Overview

This guide explains the implementation of reduced KL penalty (beta) experiments with free bits thresholding to fix the "empty latent space" problem identified in the diagnosis.

## What Was Implemented

### 1. Core Code Changes

#### `hidden_context/recurrent_vae_utils.py`
- **Added `free_bits` parameter** to `RecurrentVAETrainer.__init__`
  - Default value: `6.4` (0.1 per dimension × 64 latent dimensions)
  - Replaces hardcoded `kl_threshold = 0.05`
- **Updated KL loss computation** to use configurable threshold
  - Line ~440: Now uses `self.free_bits` instead of hardcoded value

#### `hidden_context/train_streaming_vpl.py`
- **Added `free_bits` argument** to `ScriptArguments`
  - Command-line configurable
  - Default: `6.4`
- **Updated `save_steps`** from 500 → 1000
  - Ensures checkpoints at 1000-step intervals
- **Passed `free_bits`** to `RecurrentVAETrainer` initialization

### 2. New Scripts Created

#### `run_beta_experiments.sh`
**Purpose:** Sequential experiment runner for testing multiple beta values

**Features:**
- Tests beta=0.001 and beta=0.005 sequentially
- Uses free_bits=6.4 for both experiments
- Saves checkpoints every 1000 steps
- Automatically runs diagnosis on each checkpoint
- Runs full evaluation on final model
- Properly handles virtual environment setup

**Usage:**
```bash
# Submit as Slurm job
sbatch run_beta_experiments.sh

# Or run locally (CPU)
bash run_beta_experiments.sh
```

**Output:**
- `experiments/streaming_vpl_hh/both_seq10_latent64_hidden256_beta0.001_freebits6.4_cycles4_gamma1.1_seed0/`
- `experiments/streaming_vpl_hh/both_seq10_latent64_hidden256_beta0.005_freebits6.4_cycles4_gamma1.1_seed0/`

#### `diagnose_checkpoints.py`
**Purpose:** Visualize latent space evolution during training

**Features:**
- Processes all checkpoints in an experiment directory
- Generates t-SNE visualizations for each checkpoint
- Creates multi-panel comparison showing progression
- Tracks metrics (accuracy, latent separation) over training
- Saves results to JSON for downstream analysis

**Usage:**
```bash
python diagnose_checkpoints.py \
  --exp_dir experiments/streaming_vpl_hh/both_seq10_latent64_hidden256_beta0.001_freebits6.4_cycles4_gamma1.1_seed0 \
  --data_path data_release/hh_rlhf/gpt2 \
  --latent_dim 64 \
  --hidden_dim 256 \
  --seq_length 10
```

**Outputs:**
- `{exp_dir}/checkpoint_progression.png` - Multi-panel t-SNE evolution
- `{exp_dir}/checkpoint_metrics.png` - Accuracy and separation over training
- `{exp_dir}/checkpoint_metrics.json` - Metrics data

#### `compare_beta_experiments.py`
**Purpose:** Compare results across different beta values

**Features:**
- Automatically finds experiment directories for each beta
- Generates comprehensive comparison visualizations:
  - Training curves (accuracy, separation over steps)
  - Side-by-side adaptation curves
  - Comparison table with key metrics
  - t-SNE progression for each beta
- Creates both detailed and summary views

**Usage:**
```bash
python compare_beta_experiments.py \
  --base_dir experiments/streaming_vpl_hh \
  --beta_values 0.001 0.005 \
  --output_dir experiments
```

**Outputs:**
- `experiments/beta_comparison_report.png` - Comprehensive multi-panel comparison
- `experiments/beta_tsne_comparison.png` - Side-by-side t-SNE comparison

#### `generate_beta_report.py`
**Purpose:** Create markdown report summarizing experiment results

**Features:**
- Generates comprehensive markdown report with:
  - Executive summary and quick comparison table
  - Embedded visualizations
  - Detailed metrics for each experiment
  - Training progression tables
  - Analysis and recommendations based on results
  - Automatic success/failure assessment
- Compares to baseline (β=0.1) results

**Usage:**
```bash
python generate_beta_report.py \
  --base_dir experiments/streaming_vpl_hh \
  --beta_values 0.001 0.005 \
  --output BETA_EXPERIMENT_RESULTS.md
```

**Output:**
- `BETA_EXPERIMENT_RESULTS.md` - Comprehensive report with embedded images

## Complete Workflow

### Step 1: Run Experiments
```bash
# Submit to Slurm (recommended for GPU)
sbatch run_beta_experiments.sh

# This will take 2-4 hours total
# Runs both beta=0.001 and beta=0.005 sequentially
```

### Step 2: Process Checkpoints (if not done automatically)
```bash
# For beta=0.001
python diagnose_checkpoints.py \
  --exp_dir experiments/streaming_vpl_hh/both_seq10_latent64_hidden256_beta0.001_freebits6.4_cycles4_gamma1.1_seed0 \
  --latent_dim 64 --hidden_dim 256

# For beta=0.005
python diagnose_checkpoints.py \
  --exp_dir experiments/streaming_vpl_hh/both_seq10_latent64_hidden256_beta0.005_freebits6.4_cycles4_gamma1.1_seed0 \
  --latent_dim 64 --hidden_dim 256
```

### Step 3: Generate Comparisons
```bash
python compare_beta_experiments.py
```

### Step 4: Generate Report
```bash
python generate_beta_report.py
```

### Step 5: Review Results
```bash
# View the main report
cat BETA_EXPERIMENT_RESULTS.md

# View comparison plots
open experiments/beta_comparison_report.png
open experiments/beta_tsne_comparison.png

# View individual experiment results
open experiments/streaming_vpl_hh/both_seq10_latent64_hidden256_beta0.001_*/checkpoint_progression.png
```

## Expected Outcomes

### Success Criteria
- ✅ **t-SNE shows clear separation** by checkpoint-2000 or checkpoint-3000
- ✅ **Deterministic accuracy improves** from ~59% to 70%+
- ✅ **Latent separation increases** from 0.002 to 0.02+ (10x improvement)
- ✅ **Evaluation shows positive or stable adaptation** (not negative -12%)

### Interpretation Guide

#### If β=0.001 succeeds:
- Latent space learns meaningful user type representations
- Positive adaptation (+5% to +15%)
- Clear cluster separation in t-SNE
- **Recommendation:** Use this beta for production

#### If β=0.005 succeeds but β=0.001 doesn't:
- β=0.001 may be too permissive (KL collapse)
- β=0.005 strikes better balance
- **Recommendation:** Use β=0.005 or try β=0.003

#### If both fail:
- Problem may be in observation encoder capacity
- **Next steps:**
  1. Increase `obs_dim` from 256 → 512
  2. Further reduce beta (try 0.0001)
  3. Check if observations contain sufficient signal
  4. Consider adding attention mechanism to pair encoder

## Key Parameters Explained

### Beta (KL Penalty Weight)
- **Current baseline:** 0.1
- **New values tested:** 0.001, 0.005
- **Effect:** Lower beta allows latent space to diverge from prior
- **Too high:** Latent collapse (all vectors near 0)
- **Too low:** KL collapse (no regularization, overfitting)

### Free Bits Threshold
- **Value:** 6.4 (total across all 64 dimensions)
- **Effect:** No penalty for first 0.1 "units of movement" per dimension
- **Purpose:** Removes disincentive for initial exploration
- **Standard technique:** Used in successful recurrent VAE papers

### Cyclical Annealing
- **Cycles:** 4
- **Purpose:** Repeatedly "warm up" the latent space
- **Effect:** Helps avoid local minima in KL/reconstruction tradeoff

## Files Modified

- `hidden_context/recurrent_vae_utils.py` - Added free_bits parameter
- `hidden_context/train_streaming_vpl.py` - Added free_bits argument, updated save_steps

## Files Created

- `run_beta_experiments.sh` - Experiment runner
- `diagnose_checkpoints.py` - Checkpoint diagnosis tool
- `compare_beta_experiments.py` - Comparison visualization tool
- `generate_beta_report.py` - Report generator
- `BETA_EXPERIMENTS_GUIDE.md` - This guide

## Existing Diagnostic Tools

These were created earlier and work with the new experiments:

- `diagnose_adaptation.py` - Single-checkpoint t-SNE visualization
- `diagnose_variance.py` - Thompson sampling vs deterministic analysis
- `DIAGNOSIS_REPORT.md` - Original problem diagnosis

## Troubleshooting

### "No experiments found"
- Check that experiments have completed
- Verify experiment names match pattern `*beta{value}*`
- Use `--base_dir` to specify correct directory

### "No checkpoint metrics found"
- Run `diagnose_checkpoints.py` first
- Ensure checkpoints exist in experiment directory

### Import errors
- Activate virtual environment: `source vnev/bin/activate`
- Install dependencies: `pip install torch numpy matplotlib scikit-learn pillow`

### Out of memory
- Reduce batch size in diagnose scripts
- Use `--max_samples` parameter to limit dataset size
- Run on CPU instead of GPU (slower but more memory)

## Next Steps

After reviewing the beta experiment results:

1. **If successful:** Update `run_streaming_vpl_hh.sh` with optimal beta value
2. **If needs tuning:** Adjust beta/free_bits and re-run
3. **If architectural changes needed:** Increase `obs_dim` or `hidden_dim`
4. **Document findings:** Update `DIAGNOSIS_REPORT.md` with new results

## References

- Original diagnosis: `DIAGNOSIS_REPORT.md`
- Training guide: `STREAMING_VPL_README.md`
- Implementation notes: `IMPLEMENTATION_SUMMARY.md`
- Bugfixes applied: `BUGFIXES_APPLIED.md`

---

**Implementation Date:** December 2, 2025  
**Author:** AI Assistant  
**Purpose:** Fix negative adaptation via reduced KL penalty and free bits

