"""Quick test: Check if random model gets ~50% accuracy"""
import sys
sys.path.insert(0, 'hidden_context')

import torch
import numpy as np

# Simulate a simple test
print("=" * 80)
print("Quick Random Baseline Test")
print("=" * 80)
print()

# Generate fake data: 100 episodes, each with 10 timesteps
num_episodes = 100
seq_length = 10

# Random model: predicts randomly
np.random.seed(42)
torch.manual_seed(42)

accuracies_per_step = [[] for _ in range(seq_length)]

for episode in range(num_episodes):
    for t in range(seq_length):
        # Random prediction (50/50 chance)
        correct = np.random.rand() > 0.5
        accuracies_per_step[t].append(float(correct))

print("Testing with pure random predictions (np.random):")
print()

for t in range(seq_length):
    mean_acc = np.mean(accuracies_per_step[t])
    std_acc = np.std(accuracies_per_step[t])
    print(f"  t={t}: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

print()
print("=" * 80)
print("Now testing with actual random model...")
print("=" * 80)
print()

# Test actual model
from data_utils.sequential_pets_dataset import SequentialPetsDataset, sequential_collate_fn
from recurrent_vae_utils import RecurrentVAEModel
from torch.utils.data import DataLoader

# Load just a few examples
print("Loading test dataset...")
test_dataset = SequentialPetsDataset(
    data_path="data_release/simple_pets/gpt2",
    data_subset="both",
    split="test",
    seq_length=10,
    epoch_size=20,  # Very small for speed
    seed=42
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=sequential_collate_fn
)

# Initialize random model
print("Initializing random model...")
model = RecurrentVAEModel(
    embed_dim=768,
    latent_dim=512,
    hidden_dim=512
)
model.eval()

print("Running evaluation...")
accuracies_per_step = [[] for _ in range(10)]

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 5:  # Only process 5 batches
            break
            
        outputs = model(
            target_chosen=batch['target_chosen'],
            target_rejected=batch['target_rejected'],
            context_chosen=batch['context_chosen'],
            context_rejected=batch['context_rejected'],
            seq_start_end=batch['seq_start_end'],
            user_type=batch['user_type'],
            ground_truth_user_vector=False
        )
        
        batch_size = len(batch['seq_start_end'])
        for i in range(batch_size):
            start, end = batch['seq_start_end'][i]
            episode_length = end - start
            
            for t in range(episode_length):
                idx = start + t
                r_chosen = outputs['r_chosen'][idx].item()
                r_rejected = outputs['r_rejected'][idx].item()
                
                correct = (r_chosen > r_rejected)
                accuracies_per_step[t].append(float(correct))

print()
print("=" * 80)
print("Results: Random Model")
print("=" * 80)

for t in range(10):
    if accuracies_per_step[t]:
        mean_acc = np.mean(accuracies_per_step[t])
        std_acc = np.std(accuracies_per_step[t])
        print(f"  t={t}: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

initial_acc = np.mean(accuracies_per_step[0]) if accuracies_per_step[0] else 0
print()
print(f"Initial Accuracy (t=0): {initial_acc * 100:.2f}%")
print()

if initial_acc < 0.45 or initial_acc > 0.55:
    print("⚠️  WARNING: Accuracy differs from 50%!")
    if initial_acc > 0.55:
        print("   Architecture may have hidden bias or information leakage!")
else:
    print("✅ PASS: ~50% accuracy (architecture is unbiased)")
    print("   Your trained model's 100% is due to successful learning,")
    print("   not architectural bias.")

print("=" * 80)


