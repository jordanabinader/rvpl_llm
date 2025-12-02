"""
Enhanced diagnostic script to analyze variance calibration and Thompson sampling.

This script compares:
1. Mean-based predictions (deterministic)
2. Thompson sampling predictions (stochastic)
3. Variance evolution over time

This helps diagnose why Thompson sampling causes negative adaptation.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

from hidden_context.data_utils.sequential_hh_dataset import SequentialHHDataset, sequential_hh_collate_fn
from torch.utils.data import DataLoader

# Import minimal architecture (same as diagnose_adaptation.py)
class RecurrentVPLEncoder(nn.Module):
    def __init__(self, embed_dim: int, latent_dim: int, hidden_dim: int):
        super(RecurrentVPLEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.obs_dim = 256
        
        self.pair_encoder = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.obs_dim),
            nn.LeakyReLU(0.2)
        )
        self.lstm = nn.LSTMCell(self.obs_dim, latent_dim)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, e_chosen, e_rejected, h, c):
        pair = torch.cat([e_chosen, e_rejected], dim=-1)
        obs = self.pair_encoder(pair)
        h_next, c_next = self.lstm(obs, (h, c))
        mu = self.fc_mu(h_next)
        logvar = self.fc_logvar(h_next)
        return mu, logvar, h_next, c_next


class HyperDecoder(nn.Module):
    def __init__(self, embed_dim: int, latent_dim: int):
        super(HyperDecoder, self).__init__()
        self.z_to_w = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, embed_dim)
        )
    
    def forward(self, e_chosen, e_rejected, z):
        w = self.z_to_w(z)
        r_chosen = torch.sum(e_chosen * w, dim=-1)
        r_rejected = torch.sum(e_rejected * w, dim=-1)
        return r_chosen, r_rejected


class RecurrentVAEModel(nn.Module):
    def __init__(self, encoder_embed_dim: int, decoder_embed_dim: int, hidden_dim: int, latent_dim: int):
        super(RecurrentVAEModel, self).__init__()
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = RecurrentVPLEncoder(encoder_embed_dim, latent_dim, hidden_dim)
        self.decoder = HyperDecoder(decoder_embed_dim, latent_dim)
    
    def load_model(self, path: str):
        self.load_state_dict(torch.load(path, map_location='cpu'))


def diagnose_variance(model_path, data_path, latent_dim=64, hidden_dim=256, seq_length=10, num_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from {model_path}...")
    
    model = RecurrentVAEModel(
        encoder_embed_dim=768,
        decoder_embed_dim=768,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    ).to(device)
    
    model.load_model(model_path)
    model.eval()
    print(f"Model loaded successfully (latent_dim={latent_dim}, hidden_dim={hidden_dim})")

    # Load Data
    print("Loading test data...")
    dataset = SequentialHHDataset(
        data_path=data_path,
        data_subset='both',
        seq_length=seq_length,
        split='test'
    )
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=sequential_hh_collate_fn, shuffle=False)

    # Storage
    mean_accuracies = [[] for _ in range(seq_length)]
    sampled_accuracies = [[] for _ in range(seq_length)]
    variances = [[] for _ in range(seq_length)]

    print(f"Running inference with {num_samples} Thompson samples per prediction...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            e_chosen = batch['embeddings_chosen'].to(device)
            e_rejected = batch['embeddings_rejected'].to(device)
            
            bs = e_chosen.size(0)
            h = torch.zeros(bs, model.latent_dim).to(device)
            c = torch.zeros(bs, model.latent_dim).to(device)
            
            for t in range(seq_length):
                # Update belief
                mu, logvar, h, c = model.encoder(
                    e_chosen[:, t],
                    e_rejected[:, t],
                    h, c
                )
                
                # 1. Mean-based prediction (deterministic)
                r_chosen_mean, r_rejected_mean = model.decoder(
                    e_chosen[:, t],
                    e_rejected[:, t],
                    mu
                )
                acc_mean = (r_chosen_mean > r_rejected_mean).float().cpu().numpy()
                mean_accuracies[t].append(acc_mean)
                
                # 2. Thompson sampling prediction (stochastic - multiple samples)
                std = torch.exp(0.5 * logvar)
                correct_samples = []
                
                for _ in range(num_samples):
                    eps = torch.randn_like(std)
                    z_sample = mu + std * eps
                    
                    r_chosen_sample, r_rejected_sample = model.decoder(
                        e_chosen[:, t],
                        e_rejected[:, t],
                        z_sample
                    )
                    correct = (r_chosen_sample > r_rejected_sample).float()
                    correct_samples.append(correct)
                
                # Average over samples
                acc_sampled = torch.stack(correct_samples).mean(dim=0).cpu().numpy()
                sampled_accuracies[t].append(acc_sampled)
                
                # 3. Track variance
                var = torch.exp(logvar).mean(dim=-1).cpu().numpy()  # Average over latent dims
                variances[t].append(var)

    # Aggregate results
    print("\nAggregating results...")
    mean_acc_per_t = [np.concatenate(mean_accuracies[t]).mean() for t in range(seq_length)]
    sampled_acc_per_t = [np.concatenate(sampled_accuracies[t]).mean() for t in range(seq_length)]
    var_per_t = [np.concatenate(variances[t]).mean() for t in range(seq_length)]

    # Print results
    print("\n" + "="*80)
    print("VARIANCE DIAGNOSIS RESULTS")
    print("="*80)
    print(f"\nDeterministic (Mean) Predictions:")
    print(f"  t=0: {mean_acc_per_t[0]:.2%}")
    print(f"  t={seq_length-1}: {mean_acc_per_t[-1]:.2%}")
    print(f"  Improvement: {(mean_acc_per_t[-1] - mean_acc_per_t[0]):.2%}")
    
    print(f"\nStochastic (Thompson Sampling) Predictions:")
    print(f"  t=0: {sampled_acc_per_t[0]:.2%}")
    print(f"  t={seq_length-1}: {sampled_acc_per_t[-1]:.2%}")
    print(f"  Improvement: {(sampled_acc_per_t[-1] - sampled_acc_per_t[0]):.2%}")
    
    print(f"\nVariance Statistics:")
    print(f"  t=0: {var_per_t[0]:.4f}")
    print(f"  t={seq_length-1}: {var_per_t[-1]:.4f}")
    print(f"  Ratio (t_final/t_0): {var_per_t[-1]/var_per_t[0]:.2f}x")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    timesteps = np.arange(seq_length)

    # Plot 1: Accuracy comparison
    axes[0].plot(timesteps, mean_acc_per_t, 'o-', label='Deterministic (Mean)', linewidth=2, markersize=8, color='#2E86AB')
    axes[0].plot(timesteps, sampled_acc_per_t, 's-', label=f'Stochastic (Thompson, {num_samples} samples)', linewidth=2, markersize=8, color='#A23B72')
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
    axes[0].set_xlabel('Timestep', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy Evolution: Deterministic vs Stochastic', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Plot 2: Variance evolution
    axes[1].plot(timesteps, var_per_t, 'o-', linewidth=2, markersize=8, color='#F18F01')
    axes[1].set_xlabel('Timestep', fontsize=12)
    axes[1].set_ylabel('Mean Variance (exp(logvar))', fontsize=12)
    axes[1].set_title('Posterior Variance Evolution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = "experiments/variance_diagnosis_hh.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Variance diagnosis plot saved to {output_path}")

    # Diagnosis
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if sampled_acc_per_t[0] > mean_acc_per_t[0] + 0.1:
        print("⚠️  HIGH INITIAL VARIANCE BENEFIT")
        print("   Thompson sampling gives much higher accuracy at t=0 than the mean.")
        print("   This suggests the model is 'getting lucky' with random samples.")
        
    if sampled_acc_per_t[-1] < sampled_acc_per_t[0]:
        print("\n⚠️  THOMPSON SAMPLING DEGRADATION")
        print("   Stochastic predictions get WORSE over time.")
        print("   This indicates variance is growing or becoming poorly calibrated.")
        
    if var_per_t[-1] > var_per_t[0] * 1.5:
        print("\n⚠️  VARIANCE EXPLOSION")
        print(f"   Variance increased by {var_per_t[-1]/var_per_t[0]:.1f}x")
        print("   The model is becoming MORE uncertain, not less!")
        print("\nRecommendations:")
        print("  1. Increase KL penalty (beta) to regularize variance")
        print("  2. Add variance clamping: logvar = torch.clamp(logvar, -5, 2)")
        print("  3. Use a tighter prior variance")
    elif var_per_t[-1] < var_per_t[0] * 0.5:
        print("\n⚠️  VARIANCE COLLAPSE")
        print(f"   Variance decreased by {var_per_t[0]/var_per_t[-1]:.1f}x")
        print("   The model is collapsing to a point estimate!")
        print("\nRecommendations:")
        print("  1. Decrease KL penalty to allow more variance")
        print("  2. Use free bits: KL = max(KL, threshold)")
    
    if abs(mean_acc_per_t[-1] - mean_acc_per_t[0]) < 0.05:
        print("\n⚠️  NO LEARNING IN MEAN SPACE")
        print("   The deterministic predictions show minimal adaptation.")
        print("   The latent mean (mu) is not learning user preferences.")
        print("\nRecommendations:")
        print("  1. Increase model capacity (hidden_dim or latent_dim)")
        print("  2. Check if observations contain sufficient signal")
        print("  3. Reduce KL penalty to allow more flexibility")
    
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose variance and Thompson sampling")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data_release/hh_rlhf/gpt2")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of Thompson samples to average over")
    args = parser.parse_args()
    
    diagnose_variance(
        model_path=args.model_path,
        data_path=args.data_path,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        seq_length=args.seq_length,
        num_samples=args.num_samples
    )

