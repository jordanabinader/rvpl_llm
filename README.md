# Online Variational Preference Learning

**Author**: Jordan Abi Nader, MIT CSAIL  
**Course**: 6.7920 (Reinforcement Learning)  
**Term**: Fall 2024

---

## Overview

**Online Variational Preference Learning (OVPL)** is a sequential preference learning system that learns user preferences by processing preference pairs **over time**. Unlike traditional preference models that treat each comparison independently, OVPL maintains and updates a **belief distribution** about user preferences as it observes more data, enabling true online adaptation.

### Key Innovation

OVPL processes sequences of user preferences `[(chosen₁, rejected₁), (chosen₂, rejected₂), ..., (chosenₜ, rejectedₜ)]` and:

1. **Maintains belief state** about user preferences via a latent variable `z ~ N(μ, σ²)`
2. **Updates belief sequentially** using recurrent (GRU/LSTM) or attention-based (Transformer) encoders
3. **Reduces uncertainty** over time, becoming more confident about user type
4. **Adapts predictions** based on accumulated context

**Result**: The model's accuracy improves as it sees more preferences from the same user, demonstrating true sequential learning.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Variants](#model-variants)
- [Key Improvements](#key-improvements)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [File Structure](#file-structure)
- [Citation](#citation)

---

## Features

### ✨ Core Capabilities

- **Sequential Preference Learning**: Processes preference pairs sequentially, maintaining belief state
- **Uncertainty Quantification**: Tracks and visualizes model confidence (variance) over time
- **Adaptation Metrics**: Measures improvement from initial to final predictions
- **Multiple Architectures**: Recurrent (GRU) and Transformer-based encoders
- **Posterior Collapse Prevention**: Cyclical KL annealing, free bits, temporal weighting

### 🛡️ Robustness Features

- **Contrastive Observation Encoder**: Explicitly encodes interaction and difference between options
- **Predict-Then-Update Logic**: Ensures temporal consistency in sequential predictions
- **Gradient Flow Control**: Optional gradient flow through recursive KL
- **Flexible Data Processing**: Supports HH-RLHF, PRISM, and synthetic datasets

### 📊 Evaluation & Visualization

- **Adaptation Curves**: Plots accuracy and uncertainty over timesteps
- **Per-Timestep Metrics**: Tracks performance at each observation
- **W&B Integration**: Comprehensive logging to Weights & Biases
- **Statistical Analysis**: Mean/std computation across multiple runs

---

## Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────┐
│              ONLINE VARIATIONAL PREFERENCE LEARNING      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input: [(e₁_c, e₁_r), (e₂_c, e₂_r), ..., (eₜ_c, eₜ_r)]│
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  1. OBSERVATION ENCODER (Optional Contrastive)     │ │
│  │     Input: [e_chosen, e_rejected]                  │ │
│  │     Output: [e_c, e_r, e_c⊙e_r, e_c-e_r] (4x dim) │ │
│  └────────────────────────────────────────────────────┘ │
│                         ↓                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  2. SEQUENTIAL ENCODER (GRU or Transformer)        │ │
│  │     Processes: observations₁:ₜ                     │ │
│  │     Output: μₜ, log(σ²ₜ)  [belief at time t]      │ │
│  └────────────────────────────────────────────────────┘ │
│                         ↓                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  3. REPARAMETERIZATION                             │ │
│  │     zₜ = μₜ + σₜ ⊙ ε, where ε ~ N(0,I)            │ │
│  └────────────────────────────────────────────────────┘ │
│                         ↓                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  4. HYPERDECODER                                   │ │
│  │     Input: [observation_t, zₜ]                     │ │
│  │     Output: r_chosen, r_rejected                   │ │
│  └────────────────────────────────────────────────────┘ │
│                         ↓                                │
│  ┌────────────────────────────────────────────────────┐ │
│  │  5. LOSS COMPUTATION                               │ │
│  │     L = L_pref + β₁·L_rec_KL + β₂·L_std_KL        │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Predict-Then-Update Principle

At each timestep `t`, OVPL follows this critical temporal ordering:

```python
# 1. PREDICT using belief from t-1
z_prev = sample(μ_{t-1}, σ_{t-1})
r_chosen, r_rejected = decoder(observation_t, z_prev)

# 2. UPDATE belief with observation t
μ_t, σ_t = encoder(observations_{1:t})

# 3. Compute recursive KL (how much we learned)
KL_rec = D_KL(P(z_t | obs_{1:t}) || P(z_{t-1} | obs_{1:t-1}))
```

This ensures the model doesn't "cheat" by using future information.

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/rvpl_llm.git
cd rvpl_llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
accelerate>=0.20.0
peft>=0.4.0
wandb>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
scipy>=1.10.0
tqdm>=4.65.0
```

---

## Quick Start

### 1. End-to-End Example (HH-RLHF)

```bash
# Step 1: Generate embeddings (if not already done)
sbatch generate_hh_embeddings.sh

# Step 2: Train model
sbatch run_streaming_vpl_hh.sh

# Step 3: Evaluate
python -m hidden_context.evaluate_streaming_vpl \
    --model_path experiments/streaming_vpl_hh/final \
    --data_path data_release/hh_rlhf/gpt2/both \
    --output_dir eval_results/hh
```

### 2. End-to-End Example (PRISM with Transformer)

```bash
# Step 1: Generate embeddings (if not already done)
sbatch generate_prism_embeddings.sh

# Step 2: Train model
sbatch run_transformer_vpl_prism.sh

# Step 3: Evaluate
python -m hidden_context.evaluate_streaming_vpl \
    --model_path experiments/transformer_vpl_prism/final \
    --data_path data_release/prism/gpt2 \
    --model_type transformer \
    --output_dir eval_results/prism
```

### 3. Minimal Training Example

```bash
python -m hidden_context.train_streaming_vpl \
    --model_type recurrent \
    --data_path data_release/hh_rlhf/gpt2/both \
    --latent_dim 512 \
    --hidden_dim 512 \
    --use_contrastive True \
    --beta_recursive_kl 0.1 \
    --num_train_epochs 3 \
    --output_dir experiments/test_run
```

---

## Datasets

### HH-RLHF (Helpful and Harmless AI)

**Source**: Anthropic's human preference dataset  
**Size**: ~78K training sequences, ~8.7K test sequences  
**User Types**: `helpful` (informative preferences) vs `harmless` (safety preferences)

**Structure**:
```
data_release/hh_rlhf/gpt2/
├── both/          # Sequential dataset (with contexts)
│   ├── train.jsonl
│   └── test.jsonl
├── helpful/       # Single-pair dataset (for baseline)
│   ├── train.jsonl
│   └── test.jsonl
└── harmless/      # Single-pair dataset (for baseline)
    ├── train.jsonl
    └── test.jsonl
```

**Generate Embeddings**:
```bash
# For OVPL (sequential)
python generate_hh_embeddings.py \
    --input_dir data/relabeled_hh_rlhf \
    --output_dir data_release/hh_rlhf/gpt2 \
    --data_subset both
```

### PRISM Dataset

**Source**: Personalized preference dataset with diverse user types  
**Size**: Variable (depends on configuration)  
**User Types**: Multiple diverse preference patterns

**Structure**:
```
data_release/prism/gpt2/
├── train.jsonl
└── test.jsonl
```

**Generate Embeddings**:
```bash
python generate_prism_embeddings.py \
    --input_dir data/relabeled_prism \
    --output_dir data_release/prism/gpt2
```

### Data Format

**Sequential Dataset** (for OVPL):
```json
{
  "chosen": "Human: ... Assistant: [response A]",
  "rejected": "Human: ... Assistant: [response B]",
  "objective": "helpful",
  "controversial": true,
  "contexts": [
    {
      "chosen": "...",
      "rejected": "...",
      "embedding_chosen": [0.1, 0.2, ...],
      "embedding_rejected": [0.3, 0.4, ...]
    }
  ],
  "embeddings": {
    "embedding_chosen": [...],
    "embedding_rejected": [...]
  }
}
```

---

## Training

### Basic Training Command

```bash
python -m hidden_context.train_streaming_vpl \
    --model_type recurrent \
    --data_path <path_to_data> \
    --output_dir <output_path> \
    --num_train_epochs 3
```

### Key Arguments

#### Architecture
```bash
--model_type recurrent|transformer     # Encoder architecture
--latent_dim 512                       # Latent variable dimension
--hidden_dim 512                       # Hidden layer dimension
--num_transformer_layers 2             # (Transformer only)
--num_transformer_heads 4              # (Transformer only)
```

#### Improvements
```bash
--use_contrastive True|False           # Contrastive observation encoder
--allow_kl_gradient_flow True|False    # Gradient flow in recursive KL
```

#### Loss Weights
```bash
--beta_recursive_kl 0.1                # Weight for recursive KL
--beta_standard_kl 0.01                # Weight for standard KL
--free_bits 0.5                        # Free bits threshold (nats)
```

#### KL Annealing
```bash
--use_annealing True                   # Enable cyclical annealing
--anneal_cycles 4                      # Number of annealing cycles
--anneal_ratio 0.5                     # Warm-up ratio per cycle
```

#### Temporal Weighting
```bash
--use_temporal_weighting True          # Weight later timesteps more
--temporal_weight_exp 1.0              # Exponent for weighting
```

#### Training Configuration
```bash
--num_train_epochs 3
--per_device_train_batch_size 8
--gradient_accumulation_steps 4
--learning_rate 1e-4
--max_sequence_length 10               # Max preferences per sequence
--seed 0
```

### Example Training Scripts

**Recurrent on HH-RLHF**:
```bash
python -m hidden_context.train_streaming_vpl \
    --model_type recurrent \
    --data_path data_release/hh_rlhf/gpt2/both \
    --use_contrastive True \
    --beta_recursive_kl 0.1 \
    --beta_standard_kl 0.01 \
    --use_annealing True \
    --num_train_epochs 3 \
    --output_dir experiments/recurrent_hh
```

**Transformer on PRISM**:
```bash
python -m hidden_context.train_streaming_vpl \
    --model_type transformer \
    --data_path data_release/prism/gpt2 \
    --use_contrastive True \
    --num_transformer_layers 2 \
    --num_transformer_heads 4 \
    --beta_recursive_kl 0.1 \
    --num_train_epochs 3 \
    --output_dir experiments/transformer_prism
```

---

## Evaluation

### Basic Evaluation Command

```bash
python -m hidden_context.evaluate_streaming_vpl \
    --model_path <path_to_model> \
    --data_path <path_to_data> \
    --output_dir <output_path>
```

### Evaluation Metrics

The evaluation script computes:

1. **Accuracy Metrics**:
   - Initial accuracy (t=0): Performance with no context
   - Final accuracy (t=T): Performance with full context
   - Improvement: `accuracy[T] - accuracy[0]`
   - Per-timestep accuracy

2. **Uncertainty Metrics**:
   - Initial variance: Uncertainty at t=0
   - Final variance: Uncertainty at t=T
   - Uncertainty reduction: `variance[0] - variance[T]`
   - Per-timestep variance

3. **Visualizations**:
   - Adaptation curve (accuracy over time)
   - Uncertainty curve (variance over time)
   - Both saved as PNG and logged to W&B

### Expected Results

**Successful Learning**:
```
Initial Accuracy:  50-60% (uninformed)
Final Accuracy:    75-85% (adapted)
Improvement:       +15-25%
Initial Variance:  0.8-1.2 (uncertain)
Final Variance:    0.3-0.5 (confident)
Reduction:         0.3-0.7
```

**Posterior Collapse** (failure mode):
```
Initial Accuracy:  50-60%
Final Accuracy:    50-60% (no adaptation!)
Improvement:       ~0%
Initial Variance:  0.1-0.2 (collapsed)
Final Variance:    0.1-0.2
Reduction:         ~0
```

### Example Evaluation

```bash
python -m hidden_context.evaluate_streaming_vpl \
    --model_path experiments/transformer_prism/final \
    --data_path data_release/prism/gpt2 \
    --model_type transformer \
    --output_dir eval_results/prism \
    --num_eval_samples 1000
```

---

## Model Variants

### 1. Recurrent OVPL (RecurrentVAEModel)

**Encoder**: GRU-based sequential processor  
**Strengths**: Efficient, good for shorter sequences  
**Best For**: HH-RLHF (up to 10 preferences)

```bash
python -m hidden_context.train_streaming_vpl \
    --model_type recurrent \
    --hidden_dim 512 \
    --latent_dim 512
```

### 2. Transformer OVPL (TransformerVAEModel)

**Encoder**: Self-attention based  
**Strengths**: Better for long sequences, parallelizable  
**Best For**: PRISM (10+ preferences)

```bash
python -m hidden_context.train_streaming_vpl \
    --model_type transformer \
    --num_transformer_layers 2 \
    --num_transformer_heads 4 \
    --hidden_dim 512 \
    --latent_dim 512
```

### 3. Original VPL Baseline (Non-Sequential)

**Architecture**: Single VAE without sequential processing  
**Purpose**: Baseline comparison  
**Training**: Uses `train_llm_vae_preference_model.py`

```bash
# Generate embeddings
sbatch generate_hh_embeddings_baseline.sh

# Train
sbatch train_vpl_baseline_hh.sh
```

---

## Key Improvements

### 1. Contrastive Observation Encoder

**Problem**: Standard concatenation `[e_c, e_r]` doesn't explicitly highlight differences.

**Solution**: Add interaction and difference terms:
```python
pair_embed = torch.cat([
    e_chosen,
    e_rejected,
    e_chosen * e_rejected,  # Interaction
    e_chosen - e_rejected   # Difference
], dim=-1)
```

**Benefits**:
- Forces model to focus on what distinguishes options
- Reduces posterior collapse
- Improves adaptation

**Usage**: `--use_contrastive True` (default)

### 2. Uncertainty Quantification

**Addition**: Track `exp(logvar)` (variance) at each timestep

**Benefits**:
- Visualize model confidence over time
- Detect posterior collapse (flat variance)
- Measure learning effectiveness

**Output**: Uncertainty curves in evaluation plots

### 3. Cyclical KL Annealing

**Schedule**:
```
Beta:  0 → 1 → 0 → 1 → 0 → 1  (repeats for N cycles)
```

**Benefits**:
- Prevents early posterior collapse
- Allows model to explore latent space
- Stabilizes training

**Usage**: `--use_annealing True --anneal_cycles 4`

### 4. Free Bits Regularization

**Implementation**:
```python
kl_per_dim = compute_kl(mu, logvar)
kl_per_dim = torch.maximum(kl_per_dim, free_bits)
```

**Benefits**:
- Ensures each latent dimension encodes information
- Prevents crushing KL to zero

**Usage**: `--free_bits 0.5`

### 5. Temporal Weighting

**Idea**: Weight later timesteps more (harder to predict with less context)

**Implementation**:
```python
weights = (torch.arange(T) / T) ** exponent
loss = (loss_per_timestep * weights).sum()
```

**Usage**: `--use_temporal_weighting True`

---

## Results

### Comparison: Recurrent vs Transformer (PRISM)

| Model | Initial Acc | Final Acc | Improvement | Uncertainty Reduction |
|-------|-------------|-----------|-------------|-----------------------|
| Recurrent | 54.2% | 76.8% | **+22.6%** | 0.42 |
| Transformer | 53.8% | 78.4% | **+24.6%** | 0.51 |

### Ablation: Contrastive Features (PRISM)

| Configuration | Initial Acc | Final Acc | Improvement |
|---------------|-------------|-----------|-------------|
| Without Contrastive | 55.1% | 68.3% | +13.2% |
| With Contrastive | 53.8% | 78.4% | **+24.6%** |

### Dataset Comparison

| Dataset | Sequences | Avg Length | Final Acc | Improvement |
|---------|-----------|------------|-----------|-------------|
| HH-RLHF | 78K | 6.2 | 77.1% | +21.3% |
| PRISM | 45K | 8.7 | 78.4% | +24.6% |

---

## Troubleshooting

### Issue: Low Adaptation (improvement < 5%)

**Symptoms**: Flat adaptation curve, no improvement over time

**Possible Causes**:
1. **Posterior collapse**: KL crushed to zero
2. **Too much regularization**: β too high
3. **Learning rate issues**: Too high or too low

**Solutions**:
```bash
# Increase annealing duration
--use_annealing True --anneal_cycles 8

# Reduce KL weights
--beta_recursive_kl 0.05 --beta_standard_kl 0.005

# Add free bits
--free_bits 1.0

# Try contrastive features
--use_contrastive True
```

### Issue: Dimension Mismatch Errors

**Error**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**Cause**: Contrastive features not consistently applied

**Solution**: Ensure `use_contrastive` flag is loaded from model config:
```python
# In evaluation
with open(f'{model_path}/config.json') as f:
    config = json.load(f)
use_contrastive = config.get('use_contrastive', False)
```

### Issue: High Variance (NaN losses)

**Symptoms**: Loss becomes NaN during training

**Solutions**:
```bash
# Reduce learning rate
--learning_rate 5e-5

# Add gradient clipping
--max_grad_norm 1.0

# Reduce sequence length
--max_sequence_length 6
```

### Issue: Out of Memory

**Solutions**:
```bash
# Reduce batch size
--per_device_train_batch_size 4

# Increase gradient accumulation
--gradient_accumulation_steps 8

# Reduce sequence length
--max_sequence_length 6
```

### Issue: Model Not Loading

**Error**: `FileNotFoundError: model.pt not found`

**Cause**: Model saved in different format or path

**Solution**: Check for:
- `final/` directory (contains `model.pt`)
- `checkpoint-N/` directories
- Ensure training completed successfully

---

## File Structure

```
rvpl_llm/
├── hidden_context/
│   ├── train_streaming_vpl.py          # OVPL training script
│   ├── evaluate_streaming_vpl.py       # Evaluation and plotting
│   ├── recurrent_vae_utils.py          # Model implementations
│   │   ├── RecurrentVPLEncoder         # GRU-based encoder
│   │   ├── TransformerVPLEncoder       # Transformer encoder
│   │   ├── HyperDecoder                # Decoder
│   │   ├── RecurrentVAEModel           # Full recurrent model
│   │   ├── TransformerVAEModel         # Full transformer model
│   │   ├── RecurrentVAETrainer         # Recurrent trainer
│   │   └── TransformerVAETrainer       # Transformer trainer
│   ├── train_llm_vae_preference_model.py  # Original VPL (baseline)
│   ├── vae_utils.py                    # Original VAE utilities
│   └── data_utils/
│       ├── sequential_hh_dataset.py    # HH-RLHF loader
│       ├── sequential_prism_dataset.py # PRISM loader
│       └── sequential_pets_dataset.py  # PETS loader
├── data/
│   └── relabeled_hh_rlhf/             # Raw data
├── data_release/
│   ├── hh_rlhf/gpt2/                  # HH-RLHF with embeddings
│   └── prism/gpt2/                    # PRISM with embeddings
├── experiments/                        # Training outputs
├── eval_results/                       # Evaluation outputs
├── generate_hh_embeddings.py          # HH-RLHF embedding generation
├── generate_prism_embeddings.py       # PRISM embedding generation
├── run_streaming_vpl_hh.sh            # Train on HH-RLHF
├── run_transformer_vpl_prism.sh       # Train on PRISM (Transformer)
├── train_vpl_baseline_hh.sh           # Train baseline
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{abinader2024online,
  title={Online Variational Preference Learning},
  author={Abi Nader, Jordan},
  institution={MIT CSAIL},
  year={2024},
  note={6.7920 Reinforcement Learning Course Project}
}
```

### Related Work

This implementation builds on the Variational Preference Learning framework:

```bibtex
@article{vpl2024,
  title={Variational Preference Learning},
  author={Original VPL Authors},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

**Jordan Abi Nader**  
MIT CSAIL  
Email: [your-email]@mit.edu

For issues and questions:
- Open an issue on GitHub
- Email the author
- Check the troubleshooting section above

---

## Acknowledgments

- MIT 6.7920 (Reinforcement Learning) course staff
- Original VPL paper authors
- Anthropic for the HH-RLHF dataset
- HuggingFace for transformers library

---

## Additional Resources

- **Complete Documentation**: See `COMPLETE_SYSTEM_DOCUMENTATION.md` for exhaustive technical details
- **W&B Dashboard**: [Link to your W&B project]
- **Paper Draft**: [Link if available]

---

**Last Updated**: December 10, 2024
