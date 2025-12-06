# Complete System Documentation: Streaming VPL

**Date**: December 6, 2024  
**Project**: Recurrent Variational Preference Learning (RVPL) with LLM Integration

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Improvements Implemented](#improvements-implemented)
4. [Model Components](#model-components)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation Pipeline](#evaluation-pipeline)
7. [Datasets](#datasets)
8. [Scripts and Workflows](#scripts-and-workflows)
9. [Key Concepts](#key-concepts)
10. [Debugging and Fixes](#debugging-and-fixes)
11. [Baseline Comparison](#baseline-comparison)
12. [Experimental Design](#experimental-design)

---

## 1. Overview

### 1.1 What is Streaming VPL?

**Streaming VPL (Variational Preference Learning)** is a sequential preference learning system that:
- Processes preference pairs **sequentially over time**
- Maintains a **belief distribution** about user preferences (encoded as latent variable `z`)
- **Adapts** its understanding as it observes more preferences
- Predicts whether users will prefer option A or B based on accumulated context

### 1.2 Key Innovation

Unlike traditional preference models that process single pairs independently, Streaming VPL:
1. **Maintains state** across observations via a recurrent encoder (GRU/LSTM or Transformer)
2. **Reduces uncertainty** over time as more preferences are observed
3. **Learns user-specific preference patterns** encoded in latent space
4. Uses **predict-then-update** logic to ensure temporal consistency

### 1.3 Problem Being Solved

**Posterior Collapse**: In standard VAEs, the decoder can ignore the latent variable `z` if the input embeddings are too informative. Our improvements ensure:
- The decoder **must use** `z` to make predictions
- The model **confidently** learns user preferences
- The latent space **meaningfully represents** user types

---

## 2. Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMING VPL MODEL                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Sequence of (chosen, rejected) pairs                │
│         [e₁_c, e₁_r], [e₂_c, e₂_r], ..., [eₜ_c, eₜ_r]     │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         1. OBSERVATION ENCODER                         │ │
│  │  • Embeds each (chosen, rejected) pair                 │ │
│  │  • Optional: Adds contrastive features                 │ │
│  │    - Interaction: e_c * e_r                            │ │
│  │    - Difference: e_c - e_r                             │ │
│  │  Output: pair_embed [batch, 2*embed_dim or 4*embed_dim]│ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         2. SEQUENTIAL ENCODER (Recurrent/Transformer)  │ │
│  │  • GRU/LSTM or Transformer processes sequence          │ │
│  │  • Outputs at each timestep t:                         │ │
│  │    - μₜ (mean): [batch, latent_dim]                    │ │
│  │    - logσ²ₜ (log variance): [batch, latent_dim]        │ │
│  │  • Represents belief P(z|observations₁:ₜ)              │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         3. REPARAMETERIZATION                          │ │
│  │  • Sample: zₜ = μₜ + σₜ ⊙ ε, where ε ~ N(0,1)         │ │
│  │  • Enables gradient flow through sampling              │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         4. HYPERDECODER                                │ │
│  │  • Input: [pair_embed, zₜ]                             │ │
│  │  • Generates weights for reward head                   │ │
│  │  • Output: r_chosen, r_rejected                        │ │
│  │  • Bottleneck forces dependence on z                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         5. LOSS COMPUTATION                            │ │
│  │  • Preference Loss: -log(σ(r_c - r_r))                │ │
│  │  • Recursive KL: D_KL(P(z|obs₁:ₜ) || P(z|obs₁:ₜ₋₁))  │ │
│  │  • Standard KL: D_KL(P(z|obs₁:ₜ) || N(0,1))          │ │
│  │  • Total: L = L_pref + β₁·L_rec_kl + β₂·L_std_kl     │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Predict-Then-Update Logic

**Critical Temporal Ordering**: At each timestep `t`:

```python
# 1. PREDICT using belief from t-1
z_prev = sample(μ_{t-1}, σ²_{t-1})
r_chosen, r_rejected = decoder(pair_t, z_prev)

# 2. UPDATE belief to incorporate observation t
μ_t, logσ²_t = encoder(observations_{1:t})
z_t = sample(μ_t, σ²_t)

# 3. COMPUTE recursive KL (how much we learned from obs_t)
KL_rec = D_KL(P(z_t | obs_{1:t}) || P(z_{t-1} | obs_{1:t-1}))
```

**Why This Matters**:
- At timestep `t`, we predict using what we knew at `t-1`
- We then update our belief with the new observation
- Recursive KL penalizes large jumps in belief (smooth learning)

---

## 3. Improvements Implemented

### 3.1 Contrastive Observation Encoder

**Problem**: Encoder receives `[e_chosen, e_rejected]` but doesn't explicitly see **what distinguishes them**.

**Solution**: Add interaction and difference terms:

```python
if use_contrastive:
    pair_embed = torch.cat([
        e_chosen,                  # [batch, 768]
        e_rejected,                # [batch, 768]
        e_chosen * e_rejected,     # Interaction (element-wise product)
        e_chosen - e_rejected      # Difference (what makes them distinct)
    ], dim=-1)  # [batch, 3072 = 4*768]
else:
    pair_embed = torch.cat([e_chosen, e_rejected], dim=-1)  # [batch, 1536]
```

**Benefits**:
- Forces model to focus on **contrastive information**
- Helps decoder rely on `z` to interpret differences
- Reduces posterior collapse

**Implementation**:
- Optional flag: `--use_contrastive` (default: True)
- Applied in: `RecurrentVPLEncoder`, `TransformerVPLEncoder`, evaluation script
- Saved in model config for consistency

### 3.2 Optional KL Gradient Flow

**Problem**: In recursive KL, we detach previous belief:
```python
prev_mu = curr_mu.detach()
prev_logvar = curr_logvar.detach()
```

This prevents gradients from flowing backward through time, limiting the model's ability to learn smooth trajectories.

**Solution**: Make detachment optional:

```python
if not allow_kl_gradient_flow:
    prev_mu = curr_mu.detach()
    prev_logvar = curr_logvar.detach()
# else: gradients flow through time
```

**Trade-offs**:
- ✅ **With gradient flow**: Smoother belief trajectories, better long-term planning
- ❌ **Risk**: Potential gradient explosion, harder optimization
- ✅ **Without gradient flow**: Stable training (default)

**Implementation**:
- Optional flag: `--allow_kl_gradient_flow` (default: False)
- Applied in: `RecurrentVAETrainer`, `TransformerVAETrainer`

### 3.3 Uncertainty Tracking

**Problem**: No visibility into model confidence over time.

**Solution**: Track and visualize variance (uncertainty) at each timestep:

```python
variance_t = torch.exp(logvar_t)  # Convert log variance to variance
```

**Metrics Computed**:
- **Initial Variance**: Average variance at timestep 0
- **Final Variance**: Average variance at final timestep
- **Uncertainty Reduction**: `initial_var - final_var`
- **Per-Timestep Variance**: Plotted in adaptation curves

**Expected Behavior**:
- Variance should **decrease** over time (model becomes more confident)
- Successful adaptation = large uncertainty reduction

**Implementation**:
- Added to `evaluate_streaming_vpl.py`
- Logged to W&B: `initial_variance`, `final_variance`, `uncertainty_reduction`
- Plotted in second panel of adaptation curve

### 3.4 Enhanced Decoder Bottleneck

**Context**: The `HyperDecoder` generates reward head weights from `[pair_embed, z]`.

**Risk**: If embeddings are too powerful, decoder might ignore `z`.

**Existing Mitigation**:
- Contrastive encoder (forces focus on differences)
- HyperDecoder architecture (weight generation conditioned on `z`)
- Free bits threshold (prevents crushing KL to zero)

**Future Work**:
- Could add explicit bottleneck layer
- Could add information-theoretic constraints

---

## 4. Model Components

### 4.1 RecurrentVPLEncoder

**Architecture**:
```python
class RecurrentVPLEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, use_contrastive=True):
        # Pair encoder: projects observations to hidden space
        self.pair_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Recurrent layer: processes sequence
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Output projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
```

**Forward Pass**:
```python
def forward(self, observations, hidden=None):
    # observations: [batch, seq_len, 2*embed_dim] or [batch, seq_len, 4*embed_dim]
    
    # Optional: add contrastive features
    if self.use_contrastive:
        observations = add_contrastive_features(observations)
    
    # Project each observation
    pair_encoded = self.pair_encoder(observations)  # [batch, seq_len, hidden]
    
    # Process sequence
    rnn_out, hidden = self.rnn(pair_encoded, hidden)  # [batch, seq_len, hidden]
    
    # Project to latent distributions
    mu = self.fc_mu(rnn_out)          # [batch, seq_len, latent_dim]
    logvar = self.fc_logvar(rnn_out)  # [batch, seq_len, latent_dim]
    
    return mu, logvar, hidden
```

### 4.2 TransformerVPLEncoder

**Architecture**:
```python
class TransformerVPLEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_heads=4, 
                 num_layers=2, use_contrastive=True):
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
```

**Key Difference from Recurrent**:
- Uses **attention** instead of recurrence
- Can attend to all previous observations simultaneously
- No hidden state (all context in attention)
- Better for long sequences, parallelizable

### 4.3 HyperDecoder

**Purpose**: Generate reward predictions conditioned on latent variable `z`.

**Architecture**:
```python
class HyperDecoder(nn.Module):
    def __init__(self, embed_dim, latent_dim, hidden_dim):
        # Hypernetwork: generates weights from z
        self.hyper_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final reward head
        self.reward_head = nn.Linear(embed_dim + hidden_dim, 1)
    
    def forward(self, pair_embed, z):
        # pair_embed: [batch, embed_dim]
        # z: [batch, latent_dim]
        
        # Generate context from z
        context = self.hyper_net(z)  # [batch, hidden_dim]
        
        # Combine embedding and context
        combined = torch.cat([pair_embed, context], dim=-1)
        
        # Predict reward
        reward = self.reward_head(combined)  # [batch, 1]
        
        return reward
```

**Why It Works**:
- `context` depends on `z`, so predictions change based on user type
- Bottleneck forces decoder to rely on `z`
- Without `z`, decoder cannot distinguish user preferences

### 4.4 Full Models

#### RecurrentVAEModel
```python
class RecurrentVAEModel(nn.Module):
    def __init__(self, encoder_embed_dim, decoder_embed_dim, hidden_dim, 
                 latent_dim, use_contrastive=True):
        self.encoder = RecurrentVPLEncoder(
            input_dim=encoder_embed_dim * (4 if use_contrastive else 2),
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            use_contrastive=use_contrastive
        )
        self.decoder = HyperDecoder(decoder_embed_dim, latent_dim, hidden_dim)
```

#### TransformerVAEModel
```python
class TransformerVAEModel(nn.Module):
    def __init__(self, encoder_embed_dim, decoder_embed_dim, hidden_dim, 
                 latent_dim, num_heads=4, num_layers=2, use_contrastive=True):
        self.encoder = TransformerVPLEncoder(
            input_dim=encoder_embed_dim * (4 if use_contrastive else 2),
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_contrastive=use_contrastive
        )
        self.decoder = HyperDecoder(decoder_embed_dim, latent_dim, hidden_dim)
```

---

## 5. Training Pipeline

### 5.1 Training Script: `train_streaming_vpl.py`

**Key Arguments**:
```bash
# Model architecture
--model_type recurrent|transformer  # Choose encoder type
--latent_dim 512                    # Size of z
--hidden_dim 512                    # Hidden layer size
--num_transformer_layers 2          # If using transformer
--num_transformer_heads 4           # If using transformer

# Improvements
--use_contrastive True|False        # Enable contrastive features
--allow_kl_gradient_flow True|False # Enable gradient flow in recursive KL

# Loss weights
--beta_recursive_kl 0.1             # Weight for recursive KL
--beta_standard_kl 0.01             # Weight for standard KL
--free_bits 0.5                     # Free bits threshold

# Annealing
--use_annealing True                # Enable cyclical KL annealing
--anneal_cycles 4                   # Number of cycles
--anneal_ratio 0.5                  # Warm-up ratio per cycle

# Temporal weighting
--use_temporal_weighting True       # Weight later timesteps more
--temporal_weight_exp 1.0           # Exponent for weighting

# Data
--data_path data_release/prism/gpt2
--max_sequence_length 10            # Max preferences per sequence

# Training
--num_train_epochs 3
--per_device_train_batch_size 8
--learning_rate 1e-4
--gradient_accumulation_steps 4
```

### 5.2 Loss Computation

**Total Loss**:
```python
total_loss = (
    preference_loss +
    beta_recursive_kl * recursive_kl_loss +
    beta_standard_kl * standard_kl_loss
)
```

**1. Preference Loss** (Bradley-Terry):
```python
def preference_loss(r_chosen, r_rejected):
    # r_chosen, r_rejected: [batch, seq_len]
    logits = r_chosen - r_rejected
    loss = -F.logsigmoid(logits)  # -log(σ(r_c - r_r))
    return loss.mean()
```

**2. Recursive KL Loss**:
```python
def recursive_kl_loss(mu, logvar, prev_mu, prev_logvar):
    # Measures how much belief changed from t-1 to t
    # KL(P(z_t | obs_{1:t}) || P(z_{t-1} | obs_{1:t-1}))
    
    var = torch.exp(logvar)
    prev_var = torch.exp(prev_logvar)
    
    kl = 0.5 * (
        torch.log(prev_var / var) +
        (var + (mu - prev_mu)**2) / prev_var - 1
    )
    
    # Free bits: don't penalize KL below threshold
    kl = torch.maximum(kl, torch.tensor(free_bits))
    
    return kl.sum(dim=-1).mean()
```

**3. Standard KL Loss**:
```python
def standard_kl_loss(mu, logvar):
    # Regularize towards N(0, 1)
    # KL(P(z | obs) || N(0, 1))
    
    kl = -0.5 * (1 + logvar - mu**2 - torch.exp(logvar))
    kl = torch.maximum(kl, torch.tensor(free_bits))
    
    return kl.sum(dim=-1).mean()
```

### 5.3 Cyclical KL Annealing

**Purpose**: Prevent posterior collapse by gradually introducing KL penalty.

**Schedule**:
```python
class CyclicalKLAnnealer:
    def __init__(self, total_steps, num_cycles=4, ratio=0.5):
        self.cycle_length = total_steps // num_cycles
        self.warm_up_steps = int(self.cycle_length * ratio)
    
    def __call__(self, step):
        cycle_step = step % self.cycle_length
        
        if cycle_step < self.warm_up_steps:
            # Linear warm-up from 0 to 1
            beta = cycle_step / self.warm_up_steps
        else:
            # Hold at 1
            beta = 1.0
        
        return beta
```

**Effect**:
```
Beta over training:
1.0 |    ┌────┐    ┌────┐    ┌────┐    ┌────┐
    |   /      \   /      \   /      \   /      \
0.5 |  /        \ /        \ /        \ /        \
    | /          X          X          X          \
0.0 |/____________________________________________\
    0    Cycle1    Cycle2    Cycle3    Cycle4    End
```

### 5.4 Temporal Weighting

**Purpose**: Prioritize learning from later timesteps (more context available).

**Implementation**:
```python
def temporal_weighting(seq_len, exponent=1.0):
    # weights: [0, 1, 2, ..., seq_len-1]
    weights = torch.arange(seq_len, dtype=torch.float32)
    
    # Normalize and apply exponent
    weights = (weights / seq_len) ** exponent
    weights = weights / weights.sum()  # Normalize to sum to 1
    
    return weights

# Apply to loss
loss_per_timestep = preference_loss(r_chosen, r_rejected)  # [batch, seq_len]
weights = temporal_weighting(seq_len)  # [seq_len]
weighted_loss = (loss_per_timestep * weights).sum()
```

### 5.5 Trainer Classes

**RecurrentVAETrainer**:
```python
class RecurrentVAETrainer:
    def compute_loss(self, model, inputs):
        # 1. Extract inputs
        e_chosen_seq = inputs['embedding_chosen']      # [batch, seq, dim]
        e_rejected_seq = inputs['embedding_rejected']  # [batch, seq, dim]
        
        # 2. Initialize tracking
        all_mu, all_logvar = [], []
        all_r_chosen, all_r_rejected = [], []
        hidden = None
        
        # 3. Sequential processing (predict-then-update)
        for t in range(seq_len):
            e_chosen_t = e_chosen_seq[:, t, :]
            e_rejected_t = e_rejected_seq[:, t, :]
            
            # PREDICT using previous belief
            if t > 0:
                z_prev = reparameterize(all_mu[-1], all_logvar[-1])
                r_chosen = model.decoder(e_chosen_t, z_prev)
                r_rejected = model.decoder(e_rejected_t, z_prev)
            else:
                # t=0: use prior N(0, 1)
                z_prior = torch.randn(batch_size, latent_dim)
                r_chosen = model.decoder(e_chosen_t, z_prior)
                r_rejected = model.decoder(e_rejected_t, z_prior)
            
            all_r_chosen.append(r_chosen)
            all_r_rejected.append(r_rejected)
            
            # UPDATE belief
            pair = construct_pair(e_chosen_t, e_rejected_t, model.use_contrastive)
            mu_t, logvar_t, hidden = model.encoder(pair, hidden)
            
            all_mu.append(mu_t)
            all_logvar.append(logvar_t)
        
        # 4. Compute losses
        pref_loss = preference_loss(all_r_chosen, all_r_rejected)
        rec_kl = recursive_kl_loss(all_mu, all_logvar)
        std_kl = standard_kl_loss(all_mu, all_logvar)
        
        # 5. Apply annealing
        beta = self.kl_annealer(self.state.global_step)
        
        total_loss = pref_loss + beta * (rec_kl + std_kl)
        
        return total_loss
```

**TransformerVAETrainer**: Similar but processes entire sequence at once.

---

## 6. Evaluation Pipeline

### 6.1 Evaluation Script: `evaluate_streaming_vpl.py`

**Purpose**: Measure adaptation over time on test sequences.

**Key Metrics**:
1. **Accuracy at each timestep**: What % of preferences are correctly predicted?
2. **Improvement**: `accuracy[final] - accuracy[initial]`
3. **Uncertainty reduction**: `variance[initial] - variance[final]`
4. **Adaptation curves**: Plots showing accuracy and variance over time

### 6.2 Evaluation Function

```python
def evaluate_adaptation(model, test_dataset, device):
    results = {
        'timestep_accuracies': [],  # Per-timestep accuracy
        'timestep_variances': [],   # Per-timestep variance
        'improvement': 0.0,
        'uncertainty_reduction': 0.0
    }
    
    for batch in test_dataset:
        e_chosen_seq = batch['embedding_chosen']
        e_rejected_seq = batch['embedding_rejected']
        
        seq_len = e_chosen_seq.shape[1]
        
        # Track per-sequence accuracy and variance
        batch_accuracies = []
        batch_variances = []
        
        # Initialize belief (for recurrent models)
        hidden = None
        
        for t in range(seq_len):
            e_chosen_t = e_chosen_seq[:, t, :]
            e_rejected_t = e_rejected_seq[:, t, :]
            
            # UPDATE belief with observations up to t
            if model_type == "recurrent":
                obs_up_to_t = construct_sequence(e_chosen_seq[:, :t+1], 
                                                   e_rejected_seq[:, :t+1])
                mu_t, logvar_t, hidden = model.encoder(obs_up_to_t, hidden)
            else:  # transformer
                obs_up_to_t = construct_sequence(e_chosen_seq[:, :t+1], 
                                                   e_rejected_seq[:, :t+1])
                mu_t, logvar_t = model.encoder(obs_up_to_t)
                mu_t = mu_t[:, -1, :]  # Use last position
                logvar_t = logvar_t[:, -1, :]
            
            # PREDICT using current belief
            z_t = reparameterize(mu_t, logvar_t)
            r_chosen = model.decoder(e_chosen_t, z_t)
            r_rejected = model.decoder(e_rejected_t, z_t)
            
            # Compute accuracy
            correct = (r_chosen > r_rejected).float()
            batch_accuracies.append(correct.mean().item())
            
            # Track variance (uncertainty)
            variance = torch.exp(logvar_t).mean().item()
            batch_variances.append(variance)
        
        results['timestep_accuracies'].append(batch_accuracies)
        results['timestep_variances'].append(batch_variances)
    
    # Aggregate across batches
    mean_accuracies = np.mean(results['timestep_accuracies'], axis=0)
    mean_variances = np.mean(results['timestep_variances'], axis=0)
    
    results['improvement'] = mean_accuracies[-1] - mean_accuracies[0]
    results['uncertainty_reduction'] = mean_variances[0] - mean_variances[-1]
    
    return results, mean_accuracies, mean_variances
```

### 6.3 Adaptation Curve Visualization

**Plot 1: Accuracy Over Time**
```python
plt.subplot(2, 1, 1)
plt.plot(range(seq_len), mean_accuracies, marker='o')
plt.axhline(y=0.5, color='r', linestyle='--', label='Random')
plt.xlabel('Timestep')
plt.ylabel('Accuracy')
plt.title(f'Adaptation Curve (Improvement: {improvement:.3f})')
plt.legend()
plt.grid(True)
```

**Plot 2: Uncertainty Over Time**
```python
plt.subplot(2, 1, 2)
plt.plot(range(seq_len), mean_variances, marker='s', color='orange')
plt.xlabel('Timestep')
plt.ylabel('Variance (Uncertainty)')
plt.title(f'Uncertainty Reduction: {reduction:.3f}')
plt.grid(True)
```

**Expected Patterns**:
- ✅ **Good**: Accuracy increases, variance decreases
- ❌ **Posterior collapse**: Flat accuracy, flat variance
- ❌ **Ignoring context**: Accuracy doesn't improve over time

### 6.4 Weights & Biases Logging

```python
wandb.log({
    'eval/initial_accuracy': mean_accuracies[0],
    'eval/final_accuracy': mean_accuracies[-1],
    'eval/improvement': improvement,
    'eval/initial_variance': mean_variances[0],
    'eval/final_variance': mean_variances[-1],
    'eval/uncertainty_reduction': reduction,
    'eval/adaptation_curve': wandb.Image(plt),
    
    # Per-timestep metrics
    **{f'eval/accuracy_t{t}': acc for t, acc in enumerate(mean_accuracies)},
    **{f'eval/variance_t{t}': var for t, var in enumerate(mean_variances)}
})
```

---

## 7. Datasets

### 7.1 HH-RLHF (Human Preferences for Helpful and Harmless AI)

**Source**: Anthropic's human preference dataset

**Structure**:
```
data/relabeled_hh_rlhf/
├── helpful/
│   ├── train.jsonl.gz  (39,237 sequences)
│   └── test.jsonl.gz   (4,360 sequences)
└── harmless/
    ├── train.jsonl.gz  (39,238 sequences)
    └── test.jsonl.gz   (4,361 sequences)

data_release/hh_rlhf/gpt2/
├── both/  (Used by Streaming VPL)
│   ├── train.jsonl  (with embeddings and contexts)
│   └── test.jsonl
├── helpful/  (Used by Original VPL baseline)
│   ├── train.jsonl  (with embeddings, no contexts)
│   └── test.jsonl
└── harmless/  (Used by Original VPL baseline)
    ├── train.jsonl  (with embeddings, no contexts)
    └── test.jsonl
```

**Data Format** (Streaming VPL):
```json
{
  "chosen": "Human: ... Assistant: [helpful response]",
  "rejected": "Human: ... Assistant: [less helpful response]",
  "objective": "helpful" or "harmless",
  "controversial": true/false,
  "contexts": [
    {
      "chosen": "...",
      "rejected": "...",
      "embedding_chosen": [0.1, 0.2, ...],  // 768-dim
      "embedding_rejected": [0.3, 0.4, ...]
    },
    // ... more context pairs
  ],
  "embeddings": {
    "embedding_chosen": [...],   // Current pair embedding
    "embedding_rejected": [...]
  }
}
```

**User Types**:
- `helpful`: Users who prefer helpful, informative responses
- `harmless`: Users who prefer safe, cautious responses
- Model should learn to distinguish these preferences in latent space

### 7.2 PRISM Dataset

**Source**: Personalized preference dataset with diverse user types

**Structure**:
```
data_release/prism/gpt2/
├── train.jsonl  (with embeddings and contexts)
└── test.jsonl
```

**Data Format**: Similar to HH-RLHF but with different user types

**Characteristics**:
- More diverse user preferences
- Longer sequences (up to 10+ preferences)
- More challenging for adaptation

### 7.3 Dataset Classes

**SequentialHHDataset**:
```python
class SequentialHHDataset(Dataset):
    def __init__(self, data_path, split='train', max_sequence_length=10):
        self.data = self.load_data(data_path, split)
        self.max_seq_len = max_sequence_length
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract contexts (previous preferences)
        contexts = item['contexts'][:self.max_seq_len-1]
        
        # Add current preference
        full_sequence = contexts + [item]
        
        # Convert to tensors
        e_chosen = torch.stack([torch.tensor(x['embeddings']['embedding_chosen']) 
                                 for x in full_sequence])
        e_rejected = torch.stack([torch.tensor(x['embeddings']['embedding_rejected']) 
                                   for x in full_sequence])
        
        return {
            'embedding_chosen': e_chosen,      # [seq_len, 768]
            'embedding_rejected': e_rejected,  # [seq_len, 768]
            'objective': item['objective'],    # 'helpful' or 'harmless'
            'seq_len': len(full_sequence)
        }
```

**SequentialPRISMDataset**: Similar structure, different user types

### 7.4 Data Collators

**Purpose**: Batch sequences of variable length

```python
def sequential_collate_fn(batch):
    # Find max sequence length in batch
    max_len = max(x['seq_len'] for x in batch)
    
    # Pad all sequences to max_len
    batch_chosen = []
    batch_rejected = []
    
    for item in batch:
        seq_len = item['seq_len']
        pad_len = max_len - seq_len
        
        # Pad with zeros
        e_chosen = F.pad(item['embedding_chosen'], (0, 0, 0, pad_len))
        e_rejected = F.pad(item['embedding_rejected'], (0, 0, 0, pad_len))
        
        batch_chosen.append(e_chosen)
        batch_rejected.append(e_rejected)
    
    return {
        'embedding_chosen': torch.stack(batch_chosen),    # [batch, max_len, 768]
        'embedding_rejected': torch.stack(batch_rejected),
        'seq_lengths': torch.tensor([x['seq_len'] for x in batch])
    }
```

---

## 8. Scripts and Workflows

### 8.1 Training Scripts

#### HH-RLHF Streaming VPL
```bash
# File: run_streaming_vpl_hh.sh
sbatch run_streaming_vpl_hh.sh

# Trains on: data_release/hh_rlhf/gpt2/both/
# Architecture: Recurrent (GRU)
# Features: Contrastive encoder, annealing
# Output: experiments/streaming_vpl_hh/
```

#### PRISM Streaming VPL (Transformer)
```bash
# File: run_transformer_vpl_prism.sh
sbatch run_transformer_vpl_prism.sh

# Trains on: data_release/prism/gpt2/
# Architecture: Transformer (2 layers, 4 heads)
# Features: Contrastive encoder, annealing
# Output: experiments/transformer_vpl_prism/
```

#### Original VPL Baseline (HH-RLHF)
```bash
# Step 1: Generate embeddings
sbatch generate_hh_embeddings_baseline.sh
# Output: data_release/hh_rlhf/gpt2/helpful/, .../harmless/

# Step 2: Train
sbatch train_vpl_baseline_hh.sh
# Trains on: data_release/hh_rlhf/gpt2/ (both helpful and harmless)
# Architecture: Original VAE (non-sequential)
# Output: experiments/vpl_baseline_hh/
```

### 8.2 Evaluation Scripts

```bash
# File: eval_prism.sh
python -m hidden_context.evaluate_streaming_vpl \
    --model_path experiments/transformer_vpl_prism/checkpoint-1000 \
    --data_path data_release/prism/gpt2 \
    --output_dir eval_results/prism \
    --model_type transformer \
    --wandb_project streaming-vpl-eval
```

### 8.3 Embedding Generation Scripts

#### HH-RLHF Embeddings
```bash
# For Streaming VPL (with contexts)
sbatch generate_hh_embeddings.sh

# For Original VPL (separate helpful/harmless)
sbatch generate_hh_embeddings_baseline.sh
```

**What it does**:
1. Loads relabeled HH-RLHF data
2. Loads GPT-2 model
3. Generates embeddings for chosen and rejected texts
4. Saves embeddings alongside raw text

#### PRISM Embeddings
```bash
sbatch generate_prism_embeddings.sh
```

### 8.4 Complete Workflow Example

**Experiment: Compare Contrastive Features on PRISM**

```bash
# 1. Generate embeddings (if not done)
sbatch generate_prism_embeddings.sh
# Wait for completion (~2-4 hours)

# 2. Train WITHOUT contrastive features
python -m hidden_context.train_streaming_vpl \
    --model_type transformer \
    --data_path data_release/prism/gpt2 \
    --use_contrastive False \
    --output_dir experiments/prism_no_contrastive \
    --num_train_epochs 3

# 3. Train WITH contrastive features
python -m hidden_context.train_streaming_vpl \
    --model_type transformer \
    --data_path data_release/prism/gpt2 \
    --use_contrastive True \
    --output_dir experiments/prism_with_contrastive \
    --num_train_epochs 3

# 4. Evaluate both
python -m hidden_context.evaluate_streaming_vpl \
    --model_path experiments/prism_no_contrastive/final \
    --output_dir eval_results/no_contrastive

python -m hidden_context.evaluate_streaming_vpl \
    --model_path experiments/prism_with_contrastive/final \
    --output_dir eval_results/with_contrastive

# 5. Compare on W&B
# Both runs logged to same project, compare:
# - eval/improvement
# - eval/uncertainty_reduction
# - Adaptation curves
```

---

## 9. Key Concepts

### 9.1 Posterior Collapse

**Definition**: In VAEs, the decoder learns to ignore the latent variable `z`, rendering it useless.

**Symptoms**:
- KL divergence drops to near zero
- All users get similar `z` regardless of preferences
- Model doesn't adapt over time
- Latent space is uninformative

**Causes**:
1. **Powerful decoder**: If embeddings are too informative, decoder doesn't need `z`
2. **Too much KL penalty**: Model crushes `z` to prior to minimize loss
3. **Poor initialization**: Model gets stuck in local minimum

**Solutions** (all implemented):
- ✅ Contrastive encoder (reduce embedding informativeness)
- ✅ Free bits (don't penalize small KL)
- ✅ Cyclical annealing (gradually introduce KL penalty)
- ✅ HyperDecoder (force dependence on `z`)
- ✅ Temporal weighting (focus on harder examples)

### 9.2 Reparameterization Trick

**Problem**: Can't backpropagate through random sampling.

**Naive Approach**:
```python
z = torch.randn(batch_size, latent_dim) * sigma + mu  # No gradients!
```

**Solution**:
```python
# 1. Sample noise from standard normal
epsilon = torch.randn_like(mu)  # ε ~ N(0, 1)

# 2. Transform noise using learned parameters
z = mu + sigma * epsilon  # z ~ N(μ, σ²)
```

**Why it works**: Gradients flow through `mu` and `sigma`, not through `epsilon`.

### 9.3 Free Bits

**Purpose**: Prevent KL from being crushed to zero.

**Implementation**:
```python
kl_per_dim = compute_kl(mu, logvar)  # [batch, latent_dim]

# Don't penalize KL below threshold
kl_per_dim = torch.maximum(kl_per_dim, free_bits)

total_kl = kl_per_dim.sum(dim=-1)  # [batch]
```

**Effect**: Each latent dimension must have at least `free_bits` nats of information.

**Typical values**: 0.5 - 1.0 nats per dimension

### 9.4 Recursive KL vs Standard KL

**Recursive KL**: `D_KL(P(z|obs_{1:t}) || P(z|obs_{1:t-1}))`
- Measures: How much did belief change from last timestep?
- Encourages: Smooth, gradual updates
- Prevents: Large jumps in belief

**Standard KL**: `D_KL(P(z|obs) || N(0,1))`
- Measures: Distance from prior
- Encourages: Regularization, prevents overfitting
- Prevents: Arbitrarily large latent values

**Balance**: Use both with different weights:
- `beta_recursive_kl = 0.1` (main regularizer for sequential learning)
- `beta_standard_kl = 0.01` (weak regularizer for stability)

### 9.5 Predict-Then-Update

**Key Principle**: Prediction at time `t` uses belief from time `t-1`.

**Correct** (what we implement):
```python
for t in range(seq_len):
    # PREDICT using old belief
    if t > 0:
        z_pred = sample(mu[t-1], logvar[t-1])
    else:
        z_pred = sample(0, 1)  # Prior
    
    r_chosen = decoder(obs[t], z_pred)
    r_rejected = decoder(obs[t], z_pred)
    
    # UPDATE belief
    mu[t], logvar[t] = encoder(obs[:t+1])
```

**Wrong** (common mistake):
```python
for t in range(seq_len):
    # UPDATE first (uses future information!)
    mu[t], logvar[t] = encoder(obs[:t+1])
    
    # PREDICT using updated belief (cheating!)
    z = sample(mu[t], logvar[t])
    r_chosen = decoder(obs[t], z)
```

**Why it matters**: Ensures fair evaluation and proper temporal credit assignment.

---

## 10. Debugging and Fixes

### 10.1 Dimension Mismatch Errors

**Error**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x1536 and 3072x256)`

**Cause**: Contrastive features not consistently applied.

**Locations Fixed**:
1. `RecurrentVPLEncoder.forward()`: Added conditional concatenation
2. `TransformerVPLEncoder.forward()`: Added conditional concatenation
3. `TransformerVAETrainer.compute_loss()`: Added conditional pair construction
4. `TransformerVAEModel.forward()`: Added conditional pair construction
5. `evaluate_streaming_vpl.py`: Added conditional pair construction

**Solution Pattern**:
```python
if model.encoder.use_contrastive:
    pair = torch.cat([e_c, e_r, e_c * e_r, e_c - e_r], dim=-1)
else:
    pair = torch.cat([e_c, e_r], dim=-1)
```

### 10.2 Missing Config Field Errors

**Error**: `AttributeError: 'Namespace' object has no attribute 'use_contrastive'`

**Cause**: New flag not saved/loaded in model config.

**Fix**:
```python
# Save config
config = {
    'model_type': args.model_type,
    'latent_dim': args.latent_dim,
    'use_contrastive': args.use_contrastive,  # NEW
    'allow_kl_gradient_flow': args.allow_kl_gradient_flow,  # NEW
    # ... other params
}
with open(os.path.join(output_dir, 'config.json'), 'w') as f:
    json.dump(config, f)

# Load config
with open(os.path.join(model_path, 'config.json')) as f:
    config = json.load(f)
args = argparse.Namespace(**config)
```

### 10.3 Empty Contexts Error

**Error**: `KeyError: 'contexts'`

**Cause**: HH-RLHF is non-sequential, doesn't have `contexts` field.

**Fix**: Add empty contexts during dataset loading:
```python
load_dataset(data_path, split=split).map(
    lambda data: {
        "data_subset": "helpful",
        "contexts": []  # Empty for non-sequential data
    }
)
```

### 10.4 HuggingFace Auth Error

**Error**: `LocalTokenNotFoundError: Token is required`

**Cause**: Script trying to authenticate for public model (GPT-2).

**Fix**:
```python
# Before
tokenizer = AutoTokenizer.from_pretrained(name, use_auth_token=True)

# After
tokenizer = AutoTokenizer.from_pretrained(name, use_auth_token=False)
```

### 10.5 Variance Plotting Dimension Error

**Error**: `ValueError: x and y must have same first dimension, but have shapes (6,) and (12,)`

**Cause**: Variance data collected/processed twice.

**Fix**: Removed duplicate data collection code in `evaluate_streaming_vpl.py`.

---

## 11. Baseline Comparison

### 11.1 Models to Compare

| Model | Architecture | Dataset | Sequential | Features |
|-------|-------------|---------|-----------|----------|
| **Original VPL** | Single VAE | HH-RLHF | ❌ No | Standard embeddings |
| **Streaming VPL (Recurrent)** | GRU + VAE | HH-RLHF/PRISM | ✅ Yes | Contrastive encoder |
| **Streaming VPL (Transformer)** | Transformer + VAE | HH-RLHF/PRISM | ✅ Yes | Contrastive encoder |
| **Streaming VPL (No Contrastive)** | Transformer + VAE | PRISM | ✅ Yes | Standard embeddings |

### 11.2 Evaluation Metrics

**Original VPL** (from training logs):
- Final test accuracy
- Final KL divergence
- Latent space t-SNE plots

**Streaming VPL** (from evaluation script):
- Adaptation curves (accuracy over time)
- Improvement: `acc[T] - acc[0]`
- Uncertainty reduction: `var[0] - var[T]`
- Initial/final accuracy and variance

### 11.3 Comparison Strategy

**Question 1: Does streaming help?**
- Compare: Original VPL vs Streaming VPL (Recurrent) on HH-RLHF
- Metric: Final accuracy (comparable for both)
- Expected: Streaming should match or exceed original

**Question 2: Do contrastive features help?**
- Compare: Streaming (with contrastive) vs Streaming (without) on PRISM
- Metric: Improvement, uncertainty reduction
- Expected: Contrastive should show better adaptation

**Question 3: Recurrent vs Transformer?**
- Compare: Streaming (GRU) vs Streaming (Transformer) on PRISM
- Metric: Improvement, final accuracy, training time
- Expected: Transformer might handle longer sequences better

**Question 4: Does model adapt?**
- Metric: Improvement > 0, uncertainty reduction > 0
- Expected: Both should be positive if learning works
- Red flag: Flat adaptation curve (posterior collapse)

### 11.4 Expected Results

**Successful Learning**:
```
Initial accuracy:  50-60%  (near random)
Final accuracy:    75-85%  (strong adaptation)
Improvement:       +15-25%
Initial variance:  0.8-1.2
Final variance:    0.3-0.5
Reduction:         0.3-0.7
```

**Posterior Collapse**:
```
Initial accuracy:  50-60%
Final accuracy:    50-60%  (no adaptation)
Improvement:       ~0%
Initial variance:  0.1-0.2  (crushed)
Final variance:    0.1-0.2
Reduction:         ~0
```

**Partial Learning**:
```
Initial accuracy:  50-60%
Final accuracy:    65-70%  (some adaptation)
Improvement:       +5-10%
Variance:          Moderate reduction
```

---

## 12. Experimental Design

### 12.1 Ablation Studies

**Contrastive Encoder Ablation**:
```bash
# Baseline (no contrastive)
python train_streaming_vpl.py --use_contrastive False

# With contrastive
python train_streaming_vpl.py --use_contrastive True
```

**KL Gradient Flow Ablation**:
```bash
# No gradient flow (default, stable)
python train_streaming_vpl.py --allow_kl_gradient_flow False

# With gradient flow (experimental)
python train_streaming_vpl.py --allow_kl_gradient_flow True
```

**Architecture Ablation**:
```bash
# Recurrent (GRU)
python train_streaming_vpl.py --model_type recurrent

# Transformer
python train_streaming_vpl.py --model_type transformer
```

### 12.2 Hyperparameter Sweeps

**Beta Sweep** (KL weight):
```bash
for beta in 0.01 0.05 0.1 0.5 1.0; do
    python train_streaming_vpl.py \
        --beta_recursive_kl $beta \
        --output_dir experiments/beta_$beta
done
```

**Free Bits Sweep**:
```bash
for fb in 0.0 0.25 0.5 1.0 2.0; do
    python train_streaming_vpl.py \
        --free_bits $fb \
        --output_dir experiments/freebits_$fb
done
```

**Latent Dimension Sweep**:
```bash
for dim in 128 256 512 1024; do
    python train_streaming_vpl.py \
        --latent_dim $dim \
        --output_dir experiments/latent_$dim
done
```

### 12.3 Metrics to Track

**W&B Logging** (automatic):
```python
wandb.log({
    # Training metrics
    'train/loss': total_loss,
    'train/preference_loss': pref_loss,
    'train/recursive_kl': rec_kl,
    'train/standard_kl': std_kl,
    'train/beta': beta,
    'train/learning_rate': lr,
    
    # Validation metrics (during training)
    'val/accuracy': val_acc,
    'val/kld': val_kl,
    
    # Evaluation metrics (separate script)
    'eval/initial_accuracy': init_acc,
    'eval/final_accuracy': final_acc,
    'eval/improvement': improvement,
    'eval/initial_variance': init_var,
    'eval/final_variance': final_var,
    'eval/uncertainty_reduction': reduction,
    'eval/adaptation_curve': wandb.Image(fig),
    
    # Per-timestep
    **{f'eval/accuracy_t{t}': acc for t, acc in enumerate(accs)},
    **{f'eval/variance_t{t}': var for t, var in enumerate(vars)}
})
```

### 12.4 Statistical Analysis

**Comparing Models**:
```python
# Run each model 5 times with different seeds
seeds = [0, 1, 2, 3, 4]

results = []
for seed in seeds:
    model = train(seed=seed)
    metrics = evaluate(model)
    results.append(metrics)

# Compute mean and std
mean_improvement = np.mean([r['improvement'] for r in results])
std_improvement = np.std([r['improvement'] for r in results])

print(f"Improvement: {mean_improvement:.3f} ± {std_improvement:.3f}")

# Statistical significance test
from scipy.stats import ttest_ind
model_a_improvements = [...]
model_b_improvements = [...]
t_stat, p_value = ttest_ind(model_a_improvements, model_b_improvements)
print(f"p-value: {p_value:.4f}")
```

---

## Appendix A: File Structure

```
rvpl_llm/
├── hidden_context/
│   ├── train_streaming_vpl.py           # Main training script (streaming)
│   ├── train_llm_vae_preference_model.py # Original VPL training
│   ├── evaluate_streaming_vpl.py        # Evaluation script
│   ├── recurrent_vae_utils.py          # Model implementations
│   ├── vae_utils.py                    # Original VAE utilities
│   ├── train_llm_preference_model.py   # Dataset utilities
│   └── data_utils/
│       ├── sequential_hh_dataset.py     # HH-RLHF sequential loader
│       ├── sequential_prism_dataset.py  # PRISM sequential loader
│       └── sequential_pets_dataset.py   # PETS sequential loader
├── data/
│   └── relabeled_hh_rlhf/              # Raw HH-RLHF data
│       ├── both/
│       ├── helpful/
│       └── harmless/
├── data_release/
│   ├── hh_rlhf/gpt2/                   # HH-RLHF with embeddings
│   │   ├── both/                        # For streaming VPL
│   │   ├── helpful/                     # For original VPL
│   │   └── harmless/                    # For original VPL
│   └── prism/gpt2/                     # PRISM with embeddings
├── experiments/                         # Training outputs
│   ├── streaming_vpl_hh/
│   ├── transformer_vpl_prism/
│   └── vpl_baseline_hh/
├── eval_results/                        # Evaluation outputs
├── logs/                                # SLURM logs
├── generate_hh_embeddings.py           # Embedding generation
├── generate_hh_embeddings.sh           # Streaming VPL embeddings
├── generate_prism_embeddings.py        # PRISM embedding generation
├── generate_prism_embeddings.sh        # PRISM embedding script
├── run_streaming_vpl_hh.sh             # Train streaming on HH
├── run_transformer_vpl_prism.sh        # Train transformer on PRISM
├── train_vpl_baseline_hh.sh            # Train original VPL
├── eval_prism.sh                       # Evaluate PRISM models
└── requirements.txt                     # Python dependencies
```

---

## Appendix B: Key Equations

### B.1 VAE Loss
$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \, \mathcal{L}_{\text{KL}}$$

### B.2 Preference Loss (Bradley-Terry)
$$\mathcal{L}_{\text{pref}} = -\log \sigma(r_{\text{chosen}} - r_{\text{rejected}})$$

where $\sigma$ is the sigmoid function.

### B.3 Recursive KL Divergence
$$\mathcal{L}_{\text{rec-KL}} = D_{\text{KL}}(q(z_t | x_{1:t}) \,||\, q(z_{t-1} | x_{1:t-1}))$$

$$= \frac{1}{2} \sum_{i=1}^{d} \left[ \log \frac{\sigma_{t-1,i}^2}{\sigma_{t,i}^2} + \frac{\sigma_{t,i}^2 + (\mu_{t,i} - \mu_{t-1,i})^2}{\sigma_{t-1,i}^2} - 1 \right]$$

### B.4 Standard KL Divergence
$$\mathcal{L}_{\text{std-KL}} = D_{\text{KL}}(q(z | x) \,||\, p(z))$$

$$= -\frac{1}{2} \sum_{i=1}^{d} \left[ 1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2 \right]$$

### B.5 Free Bits
$$\mathcal{L}_{\text{KL}}^{\text{fb}} = \max(\mathcal{L}_{\text{KL}}, \lambda_{\text{fb}})$$

where $\lambda_{\text{fb}}$ is the free bits threshold (e.g., 0.5 nats).

### B.6 Reparameterization
$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### B.7 Temporal Weighting
$$w_t = \frac{(t / T)^{\alpha}}{\sum_{s=0}^{T} (s / T)^{\alpha}}$$

where $\alpha$ is the weighting exponent and $T$ is sequence length.

---

## Appendix C: Command Reference

### Training Commands

```bash
# Streaming VPL (Recurrent) on HH-RLHF
python -m hidden_context.train_streaming_vpl \
    --model_type recurrent \
    --data_path data_release/hh_rlhf/gpt2/both \
    --use_contrastive True \
    --beta_recursive_kl 0.1 \
    --beta_standard_kl 0.01 \
    --free_bits 0.5 \
    --use_annealing True \
    --num_train_epochs 3 \
    --output_dir experiments/streaming_hh

# Streaming VPL (Transformer) on PRISM
python -m hidden_context.train_streaming_vpl \
    --model_type transformer \
    --data_path data_release/prism/gpt2 \
    --use_contrastive True \
    --num_transformer_layers 2 \
    --num_transformer_heads 4 \
    --num_train_epochs 3 \
    --output_dir experiments/streaming_prism

# Original VPL on HH-RLHF
python -m hidden_context.train_llm_vae_preference_model \
    --model_name gpt2 \
    --data_path data_release/hh_rlhf/gpt2 \
    --data_subset both \
    --num_train_epochs 2 \
    --output_dir experiments/vpl_baseline
```

### Evaluation Commands

```bash
# Evaluate streaming model
python -m hidden_context.evaluate_streaming_vpl \
    --model_path experiments/streaming_prism/final \
    --data_path data_release/prism/gpt2 \
    --output_dir eval_results/streaming_prism \
    --model_type transformer

# Generate adaptation curves
python -m hidden_context.evaluate_streaming_vpl \
    --model_path experiments/streaming_hh/checkpoint-1000 \
    --data_path data_release/hh_rlhf/gpt2/both \
    --output_dir eval_results/checkpoint_1000 \
    --model_type recurrent
```

### Embedding Generation Commands

```bash
# HH-RLHF (streaming)
python generate_hh_embeddings.py \
    --input_dir data/relabeled_hh_rlhf \
    --output_dir data_release/hh_rlhf/gpt2 \
    --data_subset both

# PRISM
python generate_prism_embeddings.py \
    --input_dir data/relabeled_prism \
    --output_dir data_release/prism/gpt2
```

---

## Summary

This document provides a complete reference for the Streaming VPL system, including:

✅ **Architecture**: Detailed explanation of encoder, decoder, and training loop  
✅ **Improvements**: Contrastive encoder, KL gradient flow, uncertainty tracking  
✅ **Implementation**: Code snippets for all major components  
✅ **Training**: Complete pipeline with loss functions and annealing  
✅ **Evaluation**: Adaptation curves and uncertainty metrics  
✅ **Datasets**: HH-RLHF and PRISM structure and loaders  
✅ **Scripts**: All training, evaluation, and utility scripts  
✅ **Debugging**: Common errors and fixes  
✅ **Experiments**: Ablation studies and comparison strategies  

This system implements state-of-the-art sequential preference learning with careful attention to preventing posterior collapse and ensuring meaningful adaptation over time.

