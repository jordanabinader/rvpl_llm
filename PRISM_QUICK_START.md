# PRISM Dataset - Quick Start Guide

## Summary

I've integrated the **PRISM alignment dataset** into your Streaming VPL system. PRISM provides:

- **8,011 real conversations** from **1,396 users**
- **~5.7 conversations per user** on average  
- **Cardinal ratings (1-100)** instead of binary preferences
- **Multi-turn conversations (2-22 turns)** with sequential context
- **User identification** across multiple conversations

## What Was Created

### 1. Dataset Loader
**File**: `hidden_context/data_utils/sequential_prism_dataset.py`
- `SequentialPRISMDataset`: Loads and organizes PRISM data by user
- `sequential_prism_collate_fn`: Batches episodes for training
- Supports turn-order preservation and hard negative sampling

### 2. Preprocessing Script
**File**: `generate_prism_embeddings.py`
- Downloads PRISM from HuggingFace (`HannahRoseKirk/prism-alignment`)
- Extracts preference pairs from multi-turn conversations
- Generates embeddings using GPT-2 or Llama
- Splits into train/test (80/20)

### 3. Training Scripts
**Files**: 
- `run_streaming_vpl_prism.sh`: Main training script
- `generate_prism_embeddings.sh`: Preprocessing script

### 4. Test Scripts
**Files**:
- `test_prism_setup.py`: Comprehensive dataset analysis
- `test_prism_simple.py`: Quick connectivity test

### 5. Documentation
**File**: `PRISM_SETUP.md` - Complete setup and usage guide

## Quick Start (3 Steps)

### Step 1: Test Dataset Access
```bash
python test_prism_simple.py
```

**Expected**: Download sample and verify structure

### Step 2: Preprocess Data
```bash
bash generate_prism_embeddings.sh
```

**What it does**:
- Downloads full PRISM dataset (8K conversations)
- Extracts ~50-100K preference pairs
- Generates GPT-2 embeddings
- Saves to `data_release/prism/gpt2/`

**Time**: ~1-2 hours with GPU

### Step 3: Train Model
```bash
bash run_streaming_vpl_prism.sh
```

**Configuration** (all optional):
```bash
bash run_streaming_vpl_prism.sh SEQ_LENGTH LATENT_DIM BETA SEED
# Example: bash run_streaming_vpl_prism.sh 10 512 0.1 0
```

**Output**: `experiments/streaming_vpl_prism/prism_seq10_latent512_beta0.1_seed0/`

## Key Differences from HH-RLHF

| Feature | HH-RLHF | PRISM |
|---------|---------|-------|
| Users | 2 types (synthetic) | 1,396 real users |
| User Tracking | Implicit | Explicit (`user_id`) |
| Ratings | Binary | Cardinal (1-100) |
| Conversations | Single turn | Multi-turn (2-22) |
| Scale | ~170K pairs | ~50-100K pairs |

## Integration Points

The PRISM dataset is automatically detected in `train_streaming_vpl.py`:

```python
if "prism" in script_args.data_path.lower():
    # Uses SequentialPRISMDataset
    # Preserves turn order
    # Samples by user_id
```

No code changes needed - just point to PRISM data directory!

## Expected Results

With PRISM, you should see:

1. **Clear User Diversity**: Different adaptation patterns per user
2. **Turn-Order Effects**: Performance improves with sequential turns
3. **Hard Negative Challenge**: Close ratings (75 vs 72) are harder than extreme pairs (90 vs 40)
4. **Realistic Preferences**: Complex, multidimensional user preferences vs synthetic splits

## Troubleshooting

### "Config name is missing"
**Fix**: The scripts now use `load_dataset("HannahRoseKirk/prism-alignment", "conversations")`

### "Data not found"
**Fix**: Run preprocessing first: `bash generate_prism_embeddings.sh`

### Argument errors in training
**Fix**: Updated to use correct argument names:
- `--beta_max` (not `--beta`)
- `--per_device_train_batch_size` (not `--batch_size`)
- `--num_train_epochs` (not `--num_epochs`)

## Citation

```bibtex
@misc{kirk2024PRISM,
  title={The PRISM Alignment Project}, 
  author={Hannah Rose Kirk and others},
  year={2024},
  eprint={2404.16019},
  archivePrefix={arXiv}
}
```

## Next Steps

1. **Run preprocessing** on your compute cluster
2. **Start training** with default settings
3. **Compare results** to HH-RLHF using `compare_beta_experiments.py`
4. **Analyze user-specific adaptation** patterns

For detailed documentation, see `PRISM_SETUP.md`.

