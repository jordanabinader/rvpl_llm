"""
Diagnostic script to visualize latent space evolution and diagnose negative adaptation.

This script:
1. Runs the model on test data
2. Extracts latent vectors z at t=0 (prior) and t=T (posterior)
3. Uses t-SNE to project them to 2D
4. Visualizes separation of user types (Helpful vs Harmless)

Expected behavior:
- At t=0: User types should be mixed/overlapping (uninformative prior)
- At t=T: User types should separate into distinct clusters (learned posterior)

If the model shows negative adaptation (accuracy decreases), possible causes:
1. KL penalty too weak → encoder overfits to noise
2. Observation encoder extracts irrelevant features
3. Latent space is not learning meaningful structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
import argparse
import sys

# Avoid problematic transformers imports by importing specific modules
# We only need the model architecture, not the Trainer
from hidden_context.data_utils.sequential_hh_dataset import SequentialHHDataset, sequential_hh_collate_fn
from torch.utils.data import DataLoader

# Import architecture components directly
# We'll avoid importing the full recurrent_vae_utils to skip transformers imports
# Instead, we'll reconstruct the model class here


# Minimal model class definitions (copied from recurrent_vae_utils to avoid transformers import)
class RecurrentVPLEncoder(nn.Module):
    """Recurrent encoder for Streaming VPL."""
    
    def __init__(self, embed_dim: int, latent_dim: int, hidden_dim: int):
        super(RecurrentVPLEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.obs_dim = 256
        
        # Pair encoder
        self.pair_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.obs_dim),
            nn.LeakyReLU(0.2)
        )
        
        # LSTM cell
        self.lstm = nn.LSTMCell(self.obs_dim, latent_dim)
        
        # Projection to mu and logvar
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, e_chosen, e_rejected, h, c):
        """
        Args:
            e_chosen: [batch, embed_dim]
            e_rejected: [batch, embed_dim]
            h: [batch, latent_dim] - hidden state
            c: [batch, latent_dim] - cell state
        Returns:
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
            h_next: [batch, latent_dim]
            c_next: [batch, latent_dim]
        """
        # Compare chosen vs rejected
        pair = torch.cat([e_chosen, e_rejected], dim=-1)
        obs = self.pair_encoder(pair)
        
        # Update LSTM
        h_next, c_next = self.lstm(obs, (h, c))
        
        # Project to latent distribution
        mu = self.fc_mu(h_next)
        logvar = self.fc_logvar(h_next)
        
        return mu, logvar, h_next, c_next


class HyperDecoder(nn.Module):
    """Decoder that uses z to generate preference vector."""
    
    def __init__(self, embed_dim: int, latent_dim: int):
        super(HyperDecoder, self).__init__()
        
        # Maps latent z to a preference direction w in embedding space
        self.z_to_w = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, embed_dim)
        )
    
    def forward(self, e_chosen, e_rejected, z):
        """
        Args:
            e_chosen: [batch, embed_dim]
            e_rejected: [batch, embed_dim]
            z: [batch, latent_dim]
        Returns:
            r_chosen: [batch] - reward for chosen
            r_rejected: [batch] - reward for rejected
        """
        # Generate preference vector from z
        w = self.z_to_w(z)
        
        # Compute rewards via dot product
        r_chosen = torch.sum(e_chosen * w, dim=-1)
        r_rejected = torch.sum(e_rejected * w, dim=-1)
        
        return r_chosen, r_rejected


class RecurrentVAEModel(nn.Module):
    """Full Recurrent VAE model for Streaming VPL."""
    
    def __init__(
        self,
        encoder_embed_dim: int,
        decoder_embed_dim: int,
        hidden_dim: int,
        latent_dim: int
    ):
        super(RecurrentVAEModel, self).__init__()
        
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = RecurrentVPLEncoder(encoder_embed_dim, latent_dim, hidden_dim)
        self.decoder = HyperDecoder(decoder_embed_dim, latent_dim)
    
    def load_model(self, path: str):
        """Load model from disk."""
        self.load_state_dict(torch.load(path, map_location='cpu'))


def diagnose(model_path, data_path, latent_dim=512, hidden_dim=512, seq_length=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model from {model_path}...")
    
    # Load Model
    model = RecurrentVAEModel(
        encoder_embed_dim=768,  # GPT2 embedding size
        decoder_embed_dim=768,  # GPT2 embedding size
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
        data_subset='both',  # We need mixed users to see separation
        seq_length=seq_length,
        split='test'
    )
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=sequential_hh_collate_fn, shuffle=False)
    print(f"Loaded {len(dataset)} test sequences")

    # Storage
    latents_t0 = []
    latents_t_final = []
    user_types = []
    accuracy_t0 = []
    accuracy_t_final = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            e_chosen = batch['embeddings_chosen'].to(device)
            e_rejected = batch['embeddings_rejected'].to(device)
            u_type = batch['user_type'].cpu().numpy()
            
            bs = e_chosen.size(0)
            
            # Init LSTM States
            h = torch.zeros(bs, model.latent_dim).to(device)
            c = torch.zeros(bs, model.latent_dim).to(device)
            
            for t in range(seq_length):
                # Update belief using LSTM encoder
                mu, logvar, h, c = model.encoder(
                    e_chosen[:, t], 
                    e_rejected[:, t], 
                    h, c
                )
                
                # Use mean (not sampling) for diagnosis
                z = mu
                
                # Predict rewards
                r_chosen, r_rejected = model.decoder(
                    e_chosen[:, t],
                    e_rejected[:, t],
                    z
                )
                
                # Compute accuracy
                correct = (r_chosen > r_rejected).float().cpu().numpy()
                
                # Store t=0 (Prior/First Impression)
                if t == 0:
                    latents_t0.append(mu.cpu().numpy())
                    accuracy_t0.append(correct)
            
            # Store t=Final (Posterior)
            latents_t_final.append(mu.cpu().numpy())
            accuracy_t_final.append(correct)
            user_types.append(u_type)

    # Concatenate
    z_0 = np.concatenate(latents_t0, axis=0)
    z_T = np.concatenate(latents_t_final, axis=0)
    labels = np.concatenate(user_types, axis=0)
    acc_0 = np.concatenate(accuracy_t0, axis=0)
    acc_T = np.concatenate(accuracy_t_final, axis=0)
    
    print(f"\nCollected {z_0.shape[0]} trajectories.")
    print(f"Accuracy at t=0: {acc_0.mean():.2%}")
    print(f"Accuracy at t={seq_length-1}: {acc_T.mean():.2%}")
    print(f"Improvement: {(acc_T.mean() - acc_0.mean()):.2%}")

    # Latent Statistics
    print(f"\nLatent Statistics:")
    print(f"  t=0 mean norm: {np.linalg.norm(z_0, axis=1).mean():.3f} ± {np.linalg.norm(z_0, axis=1).std():.3f}")
    print(f"  t={seq_length-1} mean norm: {np.linalg.norm(z_T, axis=1).mean():.3f} ± {np.linalg.norm(z_T, axis=1).std():.3f}")

    # User Type Separation Analysis
    helpful_idx = (labels == 0)
    harmless_idx = (labels == 1)
    
    print(f"\nUser type distribution: Helpful={helpful_idx.sum()}, Harmless={harmless_idx.sum()}")
    
    if helpful_idx.sum() > 0 and harmless_idx.sum() > 0:
        z_0_helpful_mean = z_0[helpful_idx].mean(axis=0)
        z_0_harmless_mean = z_0[harmless_idx].mean(axis=0)
        z_T_helpful_mean = z_T[helpful_idx].mean(axis=0)
        z_T_harmless_mean = z_T[harmless_idx].mean(axis=0)
        
        separation_0 = np.linalg.norm(z_0_helpful_mean - z_0_harmless_mean)
        separation_T = np.linalg.norm(z_T_helpful_mean - z_T_harmless_mean)
        
        print(f"\nUser Type Separation (Euclidean distance between centroids):")
        print(f"  t=0: {separation_0:.3f}")
        print(f"  t={seq_length-1}: {separation_T:.3f}")
        print(f"  Ratio (t_final/t_0): {separation_T/separation_0:.2f}x")

    # t-SNE Visualization
    print("\nComputing t-SNE (this might take a moment)...")
    
    # Combine to ensure same projection space
    combined_z = np.concatenate([z_0, z_T], axis=0)
    perplexity = min(30, len(z_0) // 3)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    z_embedded = tsne.fit_transform(combined_z)
    
    z_0_emb = z_embedded[:len(z_0)]
    z_T_emb = z_embedded[len(z_0):]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color map: 0=Helpful, 1=Harmless
    colors = ['#2E86AB', '#A23B72']  # Blue for Helpful, Magenta for Harmless
    labels_map = ['Helpful', 'Harmless']

    # Plot t=0
    for i in range(2):
        idx = (labels == i)
        if idx.sum() > 0:
            axes[0].scatter(z_0_emb[idx, 0], z_0_emb[idx, 1], 
                           c=colors[i], label=labels_map[i], alpha=0.6, s=20, edgecolors='white', linewidths=0.5)
    axes[0].set_title(f"Latent Space at t=0 (Prior)\nExpected: Mixed/Overlapping\nAccuracy: {acc_0.mean():.1%}", 
                      fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')

    # Plot t=T
    for i in range(2):
        idx = (labels == i)
        if idx.sum() > 0:
            axes[1].scatter(z_T_emb[idx, 0], z_T_emb[idx, 1], 
                           c=colors[i], label=labels_map[i], alpha=0.6, s=20, edgecolors='white', linewidths=0.5)
    axes[1].set_title(f"Latent Space at t={seq_length-1} (Posterior)\nExpected: Well-Separated Clusters\nAccuracy: {acc_T.mean():.1%}", 
                      fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')

    plt.tight_layout()
    output_path = "experiments/latent_diagnosis_hh.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Diagnosis plot saved to {output_path}")
    
    # Diagnosis Summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    
    if acc_T.mean() < acc_0.mean():
        print("⚠️  NEGATIVE ADAPTATION DETECTED")
        print(f"   Accuracy decreased by {(acc_0.mean() - acc_T.mean())*100:.1f}%")
        print("\nPossible Causes:")
        print("  1. KL penalty too weak → latent space not regularized")
        print("  2. Observation encoder overfitting to noise")
        print("  3. Hidden states diverging instead of converging")
        print("\nRecommendations:")
        print("  - Increase beta (KL weight) to regularize latent space")
        print("  - Add dropout to observation encoder")
        print("  - Visualize whether clusters are separating or merging")
    elif acc_T.mean() - acc_0.mean() < 0.05:
        print("⚠️  MINIMAL ADAPTATION")
        print(f"   Accuracy improved by only {(acc_T.mean() - acc_0.mean())*100:.1f}%")
        print("\nCheck the t-SNE plot:")
        print("  - If clusters are already separated at t=0: Problem is too easy")
        print("  - If clusters don't separate at t=T: Model isn't learning")
    else:
        print("✅ POSITIVE ADAPTATION")
        print(f"   Accuracy improved by {(acc_T.mean() - acc_0.mean())*100:.1f}%")
        print("   Model is successfully learning user preferences over time")
    
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose latent space adaptation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint (.pt file)")
    parser.add_argument("--data_path", type=str, default="data_release/hh_rlhf/gpt2",
                       help="Path to HH-RLHF dataset")
    parser.add_argument("--latent_dim", type=int, default=512,
                       help="Latent dimension (must match trained model)")
    parser.add_argument("--hidden_dim", type=int, default=512,
                       help="Hidden dimension (must match trained model)")
    parser.add_argument("--seq_length", type=int, default=10,
                       help="Sequence length")
    args = parser.parse_args()
    
    diagnose(
        model_path=args.model_path,
        data_path=args.data_path,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        seq_length=args.seq_length
    )
