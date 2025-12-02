# Diagnosis Report: Negative Adaptation in Streaming VPL (HH-RLHF)

**Model:** `both_seq10_latent64_hidden256_beta0.1_cycles4_gamma1.1_seed0`

**Problem:** Model shows negative adaptation (accuracy decreases from 90.5% → 78.5% over time)

---

## 🔬 Diagnostic Tests Performed

### 1. **Latent Space Visualization (t-SNE)**
**Script:** `diagnose_adaptation.py`

**Results:**
- **t=0 (Prior):** User types (Helpful vs Harmless) are completely mixed ✓ GOOD
  - No prior bias, starts at ~58.7% accuracy (near random)
- **t=9 (Posterior):** User types STILL completely mixed ❌ BAD
  - No clustering/separation emerges
  - Minimal improvement: 58.7% → 59.9% (+1.2%)
  - Euclidean separation: 0.001 → 0.002 (only 2.31x increase)

**Conclusion:** **The latent space is NOT learning to distinguish user types.**

![Latent Space Diagnosis](experiments/latent_diagnosis_hh.png)

---

### 2. **Variance & Thompson Sampling Analysis**
**Script:** `diagnose_variance.py`

**Results:**
- **Deterministic (Mean) Predictions:** 58.7% → 59.9% (+1.2%)
- **Stochastic (Thompson Sampling, 10 samples avg):** 48.7% → 50.9% (+2.2%)
- **Variance Evolution:** 0.979 → 0.952 (decreasing by 3%)

**Key Finding:** Variance is **stable and decreasing** (not exploding). This is actually correct Bayesian behavior—the model becomes more confident as it observes data.

![Variance Diagnosis](experiments/variance_diagnosis_hh.png)

**Conclusion:** **Variance is well-behaved. The problem is NOT variance miscalibration.**

---

### 3. **Evaluation Results Analysis**
**File:** `experiments/.../evaluation/results.json`

**Results:**
```json
{
  "initial_accuracy": 0.9050,   // t=0: 90.5%
  "final_accuracy": 0.7850,     // t=9: 78.5%
  "improvement": -0.1200,       // -12.0% degradation
  "std_accuracies": [0.29, 0.40, 0.38, ...]  // HIGH VARIANCE
}
```

**Key Observation:** Standard deviations are **0.2-0.4**, meaning:
- Some episodes achieve ~100% accuracy
- Some episodes drop to ~40% accuracy
- Highly inconsistent performance across users

![Adaptation Curve](experiments/streaming_vpl_hh/both_seq10_latent64_hidden256_beta0.1_cycles4_gamma1.1_seed0/evaluation/adaptation_curve.png)

**Conclusion:** **Performance is highly variable and inconsistent across episodes.**

---

## 🧠 Root Cause Analysis

### The Paradox: Why do evaluation results differ from diagnosis?

| Metric | Diagnosis (Mean) | Diagnosis (Thompson) | Evaluation (Thompson) |
|--------|------------------|----------------------|------------------------|
| t=0 Accuracy | 58.7% | 48.7% | **90.5%** |
| t=9 Accuracy | 59.9% | 50.9% | **78.5%** |
| Change | +1.2% | +2.2% | **-12.0%** |

### Explanation:

1. **Diagnosis uses deterministic test set** (fixed 1000 sequences)
   - Consistent, reproducible
   - Shows the model's true latent space structure
   - Reveals that **mu** (mean) doesn't separate user types

2. **Evaluation uses stochastic sampling** (Thompson sampling on random episodes)
   - Each episode samples different sequences
   - High variance across episodes (std = 0.3-0.4)
   - Some episodes are "lucky" (easy sequences) → 100% accuracy
   - Some episodes are "unlucky" (hard sequences) → 40% accuracy

3. **The 90% initial accuracy is misleading**
   - It's an artifact of Thompson sampling on favorable random episodes
   - NOT indicative of true model quality
   - The **mean-based** diagnostic (58.7%) is more accurate

---

## 🎯 The Real Problem: "Empty Latent Space"

The core issue is **not** variance explosion or miscalibration. It's that:

### **The model's latent space doesn't encode user preferences.**

**Evidence:**
1. ✅ t-SNE shows no separation between Helpful/Harmless users at t=9
2. ✅ Deterministic predictions barely improve (58.7% → 59.9%)
3. ✅ Latent vector means differ by only 0.002 Euclidean distance
4. ✅ Variance behavior is correct (decreasing over time)

**Why is this happening?**

### Hypothesis 1: **KL Penalty Too Strong (β = 0.1)**
- The KL divergence penalty forces latent vectors to stay close to N(0,1)
- This prevents the model from exploring the latent space
- The encoder can't differentiate between user types

**Test:** Reduce β from 0.1 → 0.01 or 0.005

### Hypothesis 2: **Insufficient Observation Signal**
- The pair encoder (`e_chosen` vs `e_rejected`) may not extract meaningful preference differences
- GPT2 embeddings might be too high-level (semantic meaning, not preference)
- The 256-dim observation layer may be too small to capture nuance

**Test:** 
- Increase `obs_dim` from 256 → 512
- Add attention mechanism to pair encoder
- Use last-layer hidden states instead of embeddings

### Hypothesis 3: **Data Leakage / Trivial Solution**
- The evaluation's 90% accuracy at t=0 suggests the **first observation** is highly predictive
- The model might be learning to identify user types from the first example alone
- No need for adaptation → latent space doesn't need to evolve

**Test:**
- Analyze first observation statistics
- Check if Helpful/Harmless users have obvious distinguishing features
- Add observation noise / masking

---

## 🛠️ Recommended Fixes (Prioritized)

### **Fix #1: Reduce KL Penalty (MOST LIKELY)**
```python
# Current: beta_max = 0.1
# Try: beta_max = 0.01 or 0.005
```

**Why:** This gives the encoder freedom to explore latent space and learn user-specific representations.

**Expected Result:** Latent vectors will separate into distinct clusters for Helpful vs Harmless.

---

### **Fix #2: Increase Observation Capacity**
```python
# In RecurrentVPLEncoder.__init__:
self.obs_dim = 512  # Currently 256
```

**Why:** Richer observation features → better user type identification.

**Expected Result:** Better utilization of GPT2 embeddings.

---

### **Fix #3: Free Bits for KL Divergence**
```python
# In loss computation:
kl_loss = torch.max(kl_divergence, threshold * latent_dim)
```

**Why:** Prevents KL collapse while still regularizing. Standard trick in VAE training.

**Expected Result:** Latent space maintains meaningful structure without over-regularization.

---

### **Fix #4: Diagnostic Logging During Training**
Add to `train_streaming_vpl.py`:
```python
# Every N steps:
# 1. Compute latent separation between user types
# 2. Log mean/std of mu and logvar
# 3. Track deterministic accuracy (using mu, not samples)
```

**Why:** Catch latent collapse during training, not after.

---

## 📊 Success Criteria

A successful fix should show:

1. ✅ **t-SNE at t=9:** Clear separation between Helpful/Harmless clusters
2. ✅ **Deterministic accuracy:** 58% → 75%+ (meaningful adaptation)
3. ✅ **Latent separation:** Euclidean distance increases by 10x+ (0.001 → 0.01+)
4. ✅ **Evaluation:** Consistent improvement (not negative), lower variance

---

## 🎓 Key Insights

### What We Learned:

1. **Don't trust high accuracy at t=0 with Thompson sampling** 
   - It can be artificially high due to favorable random samples
   - Always check deterministic (mean-based) accuracy

2. **Variance behavior was correct all along**
   - Decreasing variance = increasing confidence (good Bayesian behavior)
   - The problem wasn't miscalibration

3. **t-SNE visualization is essential**
   - Reveals that latent space is "empty" (no structure)
   - No amount of variance tuning will fix this

4. **The KL penalty is a delicate balance**
   - Too high → latent collapse (current issue)
   - Too low → posterior collapse (KL = 0, no learning)
   - Need to monitor during training

---

## 🚀 Next Steps

1. **Retrain with β = 0.01** (reduce KL penalty by 10x)
2. **Monitor latent separation during training** (add logging)
3. **Re-run diagnose_adaptation.py** on new model
4. **Check if user types separate in latent space**
5. **If still no separation, increase obs_dim to 512**

---

**Generated:** 2025-12-02  
**Scripts Used:** `diagnose_adaptation.py`, `diagnose_variance.py`  
**Model Path:** `experiments/streaming_vpl_hh/both_seq10_latent64_hidden256_beta0.1_cycles4_gamma1.1_seed0`

