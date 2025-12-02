# Using HH-RLHF Data for Streaming VPL

This guide explains how to use the HH-RLHF dataset instead of the Pets dataset for Streaming VPL experiments. The HH-RLHF dataset should provide more realistic preference learning scenarios with better adaptation curves.

## Why Use HH-RLHF Instead of Pets?

### Problem with Pets Dataset
- **100% accuracy at t=0**: User types (dog-lover vs cat-lover) are perfectly identifiable from a single comparison
- **No adaptation curve**: Since preferences are explicit keywords ("dog" vs "cat"), the model doesn't need to learn over time
- **Trivial task**: One observation is sufficient to classify users

### Benefits of HH-RLHF Dataset
- **Subtle preferences**: "Helpful" vs "Harmless" preferences are not based on simple keywords
- **Real-world data**: Actual human preference judgments from Anthropic's HH-RLHF dataset
- **Requires adaptation**: User types should require multiple observations to identify
- **More challenging**: Model must learn abstract preference patterns, not keyword matching

## Quick Start

### Step 1: Generate Embeddings

First, you need to add embeddings to the HH-RLHF data:

```bash
./generate_hh_embeddings.sh
```

This will:
- Load the relabeled HH-RLHF data from `data/relabeled_hh_rlhf/`
- Generate GPT-2 embeddings for all samples
- Save processed data to `data_release/hh_rlhf/gpt2/`

**Time estimate**: ~10-30 minutes depending on GPU and dataset size

**Requirements**:
- GPU (CUDA) recommended for faster embedding generation
- ~5GB disk space for processed data

### Step 2: Train and Evaluate

Run the complete training and evaluation pipeline:

```bash
./run_streaming_vpl_hh.sh
```

This will:
1. Train a Streaming VPL model on HH-RLHF data (20 epochs, ~30 mins with GPU)
2. Evaluate adaptation performance on test set
3. Generate adaptation curves and save results

**Output location**: `experiments/streaming_vpl_hh/both_seq10_latent512_beta0.1_seed0/`

## Expected Results

### What You Should See (If Working Correctly)

With HH-RLHF data, you should observe:

```
t=0: 55-65% accuracy  (model doesn't know user type yet)
t=2: 65-75% accuracy  (starting to learn)
t=5: 75-85% accuracy  (good adaptation)
t=9: 85-95% accuracy  (well-adapted)

Improvement: +20-35%
```

This demonstrates **true sequential adaptation** - the model learns user preferences over time!

### If You Still See 100% at t=0

If you still see very high accuracy at t=0, possible causes:
1. **Data is too easy**: HH-RLHF might still have some obvious patterns
2. **Embeddings too informative**: The GPT-2 embeddings encode too much signal
3. **Not enough user diversity**: Only using one objective (helpful or harmless)

**Solutions**:
- Reduce `latent_dim` from 512 → 128 to force compression
- Add label noise (flip 10-20% of preferences randomly)
- Increase user type complexity (see Advanced section)

## Dataset Structure

### User Types
- **Type 0 (Helpful)**: Prefers responses that are more helpful/informative
- **Type 1 (Harmless)**: Prefers responses that are more harmless/safe

### Data Subsets
- `helpful`: Only helpful preference samples
- `harmless`: Only harmless preference samples  
- `both`: Mix of both types (recommended for adaptation experiments)

### File Structure
```
data/relabeled_hh_rlhf/          # Original data (gzipped)
├── both/
│   ├── train.jsonl.gz
│   └── test.jsonl.gz
├── helpful/
└── harmless/

data_release/hh_rlhf/gpt2/       # Processed data with embeddings
├── both/
│   ├── train.jsonl              # ~60k samples
│   └── test.jsonl               # ~15k samples
├── helpful/
└── harmless/
```

## Configuration Options

### Training Parameters

Edit `run_streaming_vpl_hh.sh` to customize:

```bash
DATA_SUBSET="both"      # Use 'both' for adaptation experiments
SEQ_LENGTH=10           # Episode length (T)
LATENT_DIM=512          # Latent dimension (reduce to 128 if too easy)
BETA_END=0.1            # KL weight (increase to 0.5 for stronger regularization)
SEED=0                  # Random seed
```

### Hyperparameter Tuning

If adaptation is weak:
- **Increase BETA_END** (0.1 → 0.3): Stronger regularization forces reliance on sequential info
- **Reduce LATENT_DIM** (512 → 128): Forces information compression
- **Increase SEQ_LENGTH** (10 → 20): More opportunities to adapt

If training is unstable:
- **Reduce learning rate**: 1e-4 → 5e-5
- **Increase beta annealing**: beta_anneal_epochs 5 → 10
- **Add gradient clipping**: max_grad_norm 1.0

## Manual Usage

### Generate Embeddings (Manual)

```bash
python generate_hh_embeddings.py \
    --input_dir data/relabeled_hh_rlhf \
    --output_dir data_release/hh_rlhf/gpt2 \
    --data_subset both \
    --embed_dim 768 \
    --max_length 512
```

### Train (Manual)

```bash
python -m hidden_context.train_streaming_vpl \
    --data_path data_release/hh_rlhf/gpt2 \
    --data_subset both \
    --seq_length 10 \
    --latent_dim 512 \
    --beta_end 0.1 \
    --num_train_epochs 20 \
    --seed 0
```

### Evaluate (Manual)

```bash
python -m hidden_context.evaluate_streaming_vpl \
    --model_path experiments/streaming_vpl_hh/both_seq10_latent512_beta0.1_seed0/final_checkpoint/model.pt \
    --data_path data_release/hh_rlhf/gpt2 \
    --data_subset both \
    --seq_length 10 \
    --num_eval_episodes 200
```

## Diagnostics

### Check If Adaptation Is Working

Run the diagnostic script to analyze your trained model:

```bash
python diagnose_adaptation.py
```

This will:
- Check if z at t=0 is already predictive of user type
- Measure separation between user types over time
- Compute linear separability of latents

**Good signs**:
- t=0 latent separability < 80%
- t=9 latent separability > 90%
- Separation increases over time

**Bad signs**:
- t=0 latent separability > 95% → Task is still too easy
- No increase in separation → Model not learning

## Troubleshooting

### "FileNotFoundError: train.jsonl not found"
→ Run `./generate_hh_embeddings.sh` first to create the processed data

### "No pools have enough samples"
→ Check that embeddings were generated correctly. Verify files exist in `data_release/hh_rlhf/gpt2/both/`

### "CUDA out of memory"
→ Reduce batch size: `--per_device_train_batch_size 4` or use CPU (slower)

### Still seeing 100% accuracy at t=0
→ See "If You Still See 100% at t=0" section above

## Advanced: Creating More Complex User Types

To make the task more challenging, you can:

1. **Use 4+ user types**: Combine helpful/harmless with other attributes
2. **Add continuous preferences**: Use regression instead of binary classification
3. **Make types overlapping**: Users agree on some samples, disagree on others

See `ACCURACY_ISSUE_ANALYSIS.md` for detailed recommendations.

## Comparison: Pets vs HH-RLHF

| Aspect | Pets Dataset | HH-RLHF Dataset |
|--------|--------------|-----------------|
| User Types | Dog lover / Cat lover | Helpful / Harmless |
| Preference Signal | Explicit keywords | Abstract patterns |
| t=0 Accuracy | ~100% | ~60% (expected) |
| Adaptation | None (trivial) | Strong (realistic) |
| Use Case | Debugging | Actual experiments |

## Next Steps

After running experiments on HH-RLHF:

1. **Analyze results**: Check `experiments/streaming_vpl_hh/*/evaluation/results.json`
2. **Compare architectures**: Try different latent_dim, seq_length, beta values
3. **Ablation studies**: Remove GRU, remove KL annealing, etc.
4. **Visualize latents**: Use t-SNE to see user type clustering over time

## Questions?

See also:
- `ACCURACY_ISSUE_ANALYSIS.md` - Why Pets dataset gives 100% accuracy
- `STREAMING_VPL_README.md` - Original streaming VPL documentation
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details

