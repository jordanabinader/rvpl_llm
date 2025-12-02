# PRISM Alignment Dataset Integration

This document describes how to use the PRISM alignment dataset with the Streaming VPL system.

## Overview

The **PRISM (Participatory, Representative, and Individualised Subjective Models)** dataset is a diverse human feedback dataset for preference and value alignment in Large Language Models. It provides:

- **Real user conversations** with LLMs (8,011 conversation trees)
- **User identification** across multiple conversations (1,500 participants)
- **Cardinal preference ratings** on a 1-100 scale (68,371 rated utterances)
- **Multi-turn interactions** (2-22 turns per conversation)
- **Demographic and preference surveys** for each user

This makes PRISM ideal for testing personalized preference learning because:
1. Users are **naturally identified** across multiple conversations
2. Conversations have **sequential turn numbers** for temporal modeling
3. Ratings are **cardinal** (not just binary), providing richer signal
4. The dataset captures **real preference diversity** across different demographics

## Dataset Details

### Source
- **HuggingFace**: `HannahRoseKirk/prism-alignment`
- **Paper**: [The PRISM Alignment Project](https://arxiv.org/abs/2404.16019)
- **License**: CC-BY-4.0 (human text), CC-BY-NC-4.0 (model responses)

### Structure

Each conversation in PRISM contains:
- `conversation_id`: Unique identifier for the conversation
- `user_id`: Unique identifier for the user (persistent across conversations)
- `conversation_type`: One of "Unguided", "Values guided", or "Controversy guided"
- `conversation_turns`: Number of turns (2-22)
- `conversation_history`: List of user prompts and model responses
- `opening_prompt`: The user's first prompt
- Performance and feedback attributes

### Conversation Format

```python
{
  "conversation_id": "abc123",
  "user_id": "user_456",
  "conversation_turns": 5,
  "conversation_history": [
    {"role": "user", "content": "Tell me about...", ...},
    {"role": "model", "content": "...", "score": 85, "model_name": "gpt-4", ...},
    {"role": "model", "content": "...", "score": 72, "model_name": "claude", ...},
    # ... more turns
  ]
}
```

At each turn:
- **Turn 0**: 4 different models respond, user rates all 4
- **Turn 1+**: The highest-rated model from turn 0 generates A/B variants, user rates both

## Setup Instructions

### 1. Test Dataset Access

First, verify you can download and access the PRISM dataset:

```bash
python test_prism_setup.py
```

This will:
- Download a sample of PRISM from HuggingFace
- Analyze the dataset structure
- Compute statistics about users, conversations, and ratings
- Estimate the number of preference pairs available

**Expected output:**
- ~1,500 unique users
- ~8,000 conversations
- ~3-5 conversations per user on average
- Hundreds of thousands of potential preference pairs

### 2. Preprocess Data and Generate Embeddings

Run the preprocessing script to:
1. Download the full PRISM dataset
2. Extract preference pairs from conversations
3. Generate embeddings using GPT-2 or Llama
4. Split into train/test sets

```bash
# Using GPT-2 (default)
bash generate_prism_embeddings.sh

# Or with custom settings
bash generate_prism_embeddings.sh gpt2 1024 10.0
#                                 ^     ^    ^
#                                 |     |    min rating difference
#                                 |     embedding dimension
#                                 model type
```

**Parameters:**
- `model_type`: `gpt2` or `llama` (for embeddings)
- `embed_dim`: Embedding dimension (default: 1024)
- `min_rating_diff`: Minimum rating difference to consider a clear preference (default: 10.0)

**Output:**
- `data_release/prism/gpt2/train.jsonl`: Training data with embeddings
- `data_release/prism/gpt2/test.jsonl`: Test data with embeddings

**Processing time:** 
- ~1-2 hours with GPU for full dataset
- Processes ~8K conversations into ~50-100K preference pairs (depending on filtering)

### 3. Train Streaming VPL Model

Once preprocessing is complete, train the model:

```bash
# Default settings
bash run_streaming_vpl_prism.sh

# With custom hyperparameters
bash run_streaming_vpl_prism.sh 10 512 0.1 0
#                                ^  ^   ^   ^
#                                |  |   |   seed
#                                |  |   beta (KL weight)
#                                |  latent dimension
#                                sequence length
```

**Training Configuration:**
- Sequence length: 10 (interactions per episode)
- Latent dimension: 512 (user preference vector size)
- Beta: 0.1 (KL divergence weight)
- Batch size: 32
- Learning rate: 1e-4
- Epochs: 10

**Output:**
- Checkpoints saved to `experiments/streaming_vpl_prism/prism_seq{SEQ}_latent{DIM}_beta{BETA}_seed{SEED}/`
- Evaluation metrics logged to WandB (project: `streaming-vpl-prism`)

## What Makes PRISM Different?

### vs. HH-RLHF (Helpful/Harmless)

| Feature | HH-RLHF | PRISM |
|---------|---------|-------|
| User Identity | **Implicit** (inferred from objective) | **Explicit** (tracked user_id) |
| User Pool | 2 types (helpful/harmless) | 1,500 real users |
| Preferences | Binary (chosen/rejected) | Cardinal (1-100 rating) |
| Turns per Interaction | 1 | 2-22 |
| Realism | Synthetic splits | Real user conversations |

### vs. Pets Dataset

| Feature | Pets | PRISM |
|---------|------|-------|
| User Identity | Synthetic | Real |
| Preference Complexity | Simple (dog vs. cat) | Complex, multidimensional |
| Domain | Toy problem | Real LLM alignment |
| Scale | ~1K samples | ~68K rated interactions |

## Key Advantages for Personalization Research

1. **Natural User Tracking**: Each `user_id` represents a real person with consistent preferences across multiple conversations

2. **Sequential Context**: Turn numbers allow modeling how preferences emerge or clarify over time within a conversation

3. **Preference Diversity**: 1,500 users with diverse demographics, values, and stated preferences (captured in survey data)

4. **Cardinal Ratings**: 1-100 scale provides richer signal than binary preferences

5. **Multiple Conversation Types**: Users engage in unguided, values-guided, and controversy-guided conversations

6. **Hard Negatives**: Many pairs have similar ratings (e.g., 75 vs 72), requiring fine-grained discrimination

## Data Processing Details

### Preference Pair Extraction

From conversations, we extract preference pairs where:
1. Multiple model responses exist for the same user prompt
2. Ratings differ by at least `min_rating_diff` (default: 10 points)
3. The higher-rated response becomes "chosen", lower becomes "rejected"

### Train/Test Split

- **Strategy**: Random 80/20 split at the preference pair level
- **Preserves**: User diversity across both splits
- **Test Set Size**: ~20% of total pairs (10-20K pairs typically)

### Difficulty Scoring

Each preference pair is scored for difficulty based on:
- **Rating similarity** (40% weight): Closer ratings = harder
- **Length similarity** (30% weight): Similar length = harder
- **Embedding similarity** (30% weight): Semantic similarity = harder

This enables hard negative mining during training.

## Evaluation Approach

### Metrics

1. **Per-User Adaptation**: Track accuracy over time for each user
   - Initial accuracy (t=0): Without any user context
   - Final accuracy (t=10): After observing 10 interactions
   - Adaptation gain: Final - Initial

2. **Cross-User Generalization**: Evaluate on held-out users
   - Does the model learn user-specific patterns?
   - Or just dataset-wide patterns?

3. **Turn-by-Turn Analysis**: How does performance improve across turns?

### Comparison Baselines

- **No personalization**: Fixed preference model (t=0 performance)
- **User embeddings**: Static user representation
- **Context window**: Standard transformer over interaction history
- **Streaming VPL**: Recurrent VAE with belief updates (your system)

## Example Usage

### Loading Processed Data

```python
from hidden_context.data_utils.sequential_prism_dataset import (
    SequentialPRISMDataset,
    sequential_prism_collate_fn
)

# Create dataset
dataset = SequentialPRISMDataset(
    data_path="data_release/prism/gpt2",
    split="train",
    seq_length=10,
    epoch_size=1000,
    seed=0,
    use_turn_order=True  # Sample consecutive turns
)

# Sample an episode
episode = dataset[0]
print(f"User ID: {episode['user_id']}")
print(f"Embeddings shape: {len(episode['embeddings_chosen'])} x {len(episode['embeddings_chosen'][0])}")
print(f"Turn numbers: {episode['turn_numbers']}")
```

### Training with PRISM

```python
from torch.utils.data import DataLoader

# Create data loader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=sequential_prism_collate_fn,
    shuffle=True
)

# Training loop automatically handles user-specific episodes
for batch in train_loader:
    embeddings_chosen = batch['embeddings_chosen']  # [B, T, D]
    embeddings_rejected = batch['embeddings_rejected']  # [B, T, D]
    user_ids = batch['user_ids']  # List of B user_id strings
    # ... your training code
```

## Expected Results

Based on the PRISM characteristics, you should observe:

1. **Clear Adaptation**: Accuracy should improve from t=0 to t=10 as the model learns user preferences

2. **User Diversity**: Different users should show different adaptation patterns
   - Some users may have very consistent preferences (fast adaptation)
   - Others may have context-dependent preferences (slower adaptation)

3. **Hard Negative Challenge**: Close rating pairs (e.g., 75 vs 72) should be harder than extreme pairs (e.g., 90 vs 40)

4. **Turn Order Matters**: Preserving turn order should improve performance vs. random sampling

5. **Better than Synthetic**: Results should be more realistic than Pets dataset, showing true preference complexity

## Troubleshooting

### Dataset Download Issues

If HuggingFace download fails:
```bash
# Set HF token if needed
export HF_TOKEN=your_token_here

# Test connection
python -c "from datasets import load_dataset; load_dataset('HannahRoseKirk/prism-alignment', split='train[:10]')"
```

### Memory Issues

If embedding generation runs out of memory:
- Reduce batch size in `generate_prism_embeddings.py`
- Use gradient checkpointing
- Process in smaller chunks

### Too Few Preference Pairs

If filtering removes too many pairs:
- Reduce `min_rating_diff` (e.g., to 5.0)
- Check that conversations have multiple responses per turn
- Verify the conversation_history format

## References

- **PRISM Paper**: Kirk et al. (2024), "The PRISM Alignment Project: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models"
- **Dataset**: https://huggingface.co/datasets/HannahRoseKirk/prism-alignment
- **Project Website**: https://hannahkirk.github.io/prism-alignment/

## Citation

If you use PRISM in your research, please cite:

```bibtex
@misc{kirk2024PRISM,
  title={The PRISM Alignment Project: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models}, 
  author={Hannah Rose Kirk and Alexander Whitefield and Paul Röttger and Andrew Bean and Katerina Margatina and Juan Ciro and Rafael Mosquera and Max Bartolo and Adina Williams and He He and Bertie Vidgen and Scott A. Hale},
  year={2024},
  url={http://arxiv.org/abs/2404.16019},
  eprint={2404.16019},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

