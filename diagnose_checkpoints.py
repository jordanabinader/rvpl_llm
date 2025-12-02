"""
Checkpoint diagnostic script for visualizing latent space evolution during training.

This script:
1. Takes an experiment directory as input
2. Finds all checkpoint-* subdirectories and final_checkpoint
3. Runs t-SNE visualization on each
4. Generates a multi-panel comparison showing evolution
5. Optionally creates an animated GIF
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
import argparse
import glob
from pathlib import Path
import re

from hidden_context.data_utils.sequential_hh_dataset import SequentialHHDataset, sequential_hh_collate_fn
from torch.utils.data import DataLoader


# Minimal model architecture (same as diagnose_adaptation.py)
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


def extract_checkpoint_number(checkpoint_path):
    """Extract checkpoint step number from path."""
    basename = os.path.basename(checkpoint_path)
    if basename == 'final_checkpoint':
        return float('inf')
    match = re.search(r'checkpoint-(\d+)', basename)
    if match:
        return int(match.group(1))
    return 0


def diagnose_checkpoint(model_path, data_path, latent_dim, hidden_dim, seq_length, max_samples=500):
    """Run diagnosis on a single checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = RecurrentVAEModel(
        encoder_embed_dim=768,
        decoder_embed_dim=768,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    ).to(device)
    
    model.load_model(model_path)
    model.eval()
    
    # Load data (limited samples for speed)
    dataset = SequentialHHDataset(
        data_path=data_path,
        data_subset='both',
        seq_length=seq_length,
        split='test'
    )
    
    # Limit to max_samples for faster processing
    if len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=sequential_hh_collate_fn, shuffle=False)
    
    # Storage
    latents_t0 = []
    latents_t_final = []
    user_types = []
    accuracy_t0 = []
    accuracy_t_final = []
    
    with torch.no_grad():
        for batch in dataloader:
            e_chosen = batch['embeddings_chosen'].to(device)
            e_rejected = batch['embeddings_rejected'].to(device)
            u_type = batch['user_type'].cpu().numpy()
            
            bs = e_chosen.size(0)
            h = torch.zeros(bs, model.latent_dim).to(device)
            c = torch.zeros(bs, model.latent_dim).to(device)
            
            for t in range(seq_length):
                mu, logvar, h, c = model.encoder(
                    e_chosen[:, t],
                    e_rejected[:, t],
                    h, c
                )
                
                z = mu
                r_chosen, r_rejected = model.decoder(
                    e_chosen[:, t],
                    e_rejected[:, t],
                    z
                )
                
                correct = (r_chosen > r_rejected).float().cpu().numpy()
                
                if t == 0:
                    latents_t0.append(mu.cpu().numpy())
                    accuracy_t0.append(correct)
            
            latents_t_final.append(mu.cpu().numpy())
            accuracy_t_final.append(correct)
            user_types.append(u_type)
    
    # Aggregate
    z_0 = np.concatenate(latents_t0, axis=0)
    z_T = np.concatenate(latents_t_final, axis=0)
    labels = np.concatenate(user_types, axis=0)
    acc_0 = np.concatenate(accuracy_t0, axis=0)
    acc_T = np.concatenate(accuracy_t_final, axis=0)
    
    # Calculate metrics
    helpful_idx = (labels == 0)
    harmless_idx = (labels == 1)
    
    separation_0 = 0.0
    separation_T = 0.0
    if helpful_idx.sum() > 0 and harmless_idx.sum() > 0:
        z_0_helpful_mean = z_0[helpful_idx].mean(axis=0)
        z_0_harmless_mean = z_0[harmless_idx].mean(axis=0)
        z_T_helpful_mean = z_T[helpful_idx].mean(axis=0)
        z_T_harmless_mean = z_T[harmless_idx].mean(axis=0)
        
        separation_0 = np.linalg.norm(z_0_helpful_mean - z_0_harmless_mean)
        separation_T = np.linalg.norm(z_T_helpful_mean - z_T_harmless_mean)
    
    return {
        'z_0': z_0,
        'z_T': z_T,
        'labels': labels,
        'acc_0': acc_0.mean(),
        'acc_T': acc_T.mean(),
        'separation_0': separation_0,
        'separation_T': separation_T
    }


def main(exp_dir, data_path, latent_dim, hidden_dim, seq_length):
    """Process all checkpoints in an experiment directory."""
    print(f"Processing experiment: {exp_dir}")
    
    # Find all checkpoints
    checkpoint_dirs = glob.glob(os.path.join(exp_dir, 'checkpoint-*'))
    final_ckpt = os.path.join(exp_dir, 'final_checkpoint')
    if os.path.exists(final_ckpt):
        checkpoint_dirs.append(final_ckpt)
    
    # Sort by checkpoint number
    checkpoint_dirs = sorted(checkpoint_dirs, key=extract_checkpoint_number)
    
    if not checkpoint_dirs:
        print("ERROR: No checkpoints found in experiment directory")
        return
    
    print(f"Found {len(checkpoint_dirs)} checkpoints")
    
    # Process each checkpoint
    results = []
    for ckpt_dir in tqdm(checkpoint_dirs, desc="Processing checkpoints"):
        model_path = os.path.join(ckpt_dir, 'model.pt')
        if not os.path.exists(model_path):
            print(f"WARNING: No model.pt found in {ckpt_dir}")
            continue
        
        ckpt_name = os.path.basename(ckpt_dir)
        print(f"\nProcessing {ckpt_name}...")
        
        result = diagnose_checkpoint(model_path, data_path, latent_dim, hidden_dim, seq_length)
        result['name'] = ckpt_name
        result['step'] = extract_checkpoint_number(ckpt_dir)
        results.append(result)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Multi-panel t-SNE progression
    n_checkpoints = len(results)
    n_cols = min(4, n_checkpoints)
    n_rows = (n_checkpoints + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_checkpoints == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Combine all latents for consistent t-SNE projection
    all_z_T = np.concatenate([r['z_T'] for r in results], axis=0)
    all_labels = np.concatenate([r['labels'] for r in results], axis=0)
    
    print("Computing t-SNE for all checkpoints...")
    perplexity = min(30, all_z_T.shape[0] // 3)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    all_z_embedded = tsne.fit_transform(all_z_T)
    
    # Split back into checkpoints
    offset = 0
    colors = ['#2E86AB', '#A23B72']
    labels_map = ['Helpful', 'Harmless']
    
    for idx, result in enumerate(results):
        n_samples = len(result['labels'])
        z_emb = all_z_embedded[offset:offset+n_samples]
        labels = result['labels']
        offset += n_samples
        
        ax = axes[idx]
        for i in range(2):
            mask = (labels == i)
            if mask.sum() > 0:
                ax.scatter(z_emb[mask, 0], z_emb[mask, 1],
                          c=colors[i], label=labels_map[i],
                          alpha=0.6, s=20, edgecolors='white', linewidths=0.5)
        
        step_label = 'Final' if result['step'] == float('inf') else f"Step {result['step']}"
        ax.set_title(f"{step_label}\nAcc: {result['acc_T']:.1%} | Sep: {result['separation_T']:.3f}",
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('t-SNE Dim 1', fontsize=8)
        ax.set_ylabel('t-SNE Dim 2', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_checkpoints, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(exp_dir, 'checkpoint_progression.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()
    
    # 2. Metrics over training
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    steps = [r['step'] for r in results if r['step'] != float('inf')]
    acc_0_vals = [r['acc_0'] for r in results if r['step'] != float('inf')]
    acc_T_vals = [r['acc_T'] for r in results if r['step'] != float('inf')]
    sep_0_vals = [r['separation_0'] for r in results if r['step'] != float('inf')]
    sep_T_vals = [r['separation_T'] for r in results if r['step'] != float('inf')]
    
    if steps:
        # Accuracy at t=0
        axes[0, 0].plot(steps, acc_0_vals, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        axes[0, 0].set_xlabel('Training Step', fontsize=12)
        axes[0, 0].set_ylabel('Accuracy at t=0', fontsize=12)
        axes[0, 0].set_title('Initial Accuracy Evolution', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # Accuracy at t=final
        axes[0, 1].plot(steps, acc_T_vals, 'o-', linewidth=2, markersize=8, color='#A23B72')
        axes[0, 1].set_xlabel('Training Step', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy at t=9', fontsize=12)
        axes[0, 1].set_title('Final Accuracy Evolution', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # Separation at t=0
        axes[1, 0].plot(steps, sep_0_vals, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        axes[1, 0].set_xlabel('Training Step', fontsize=12)
        axes[1, 0].set_ylabel('Euclidean Distance', fontsize=12)
        axes[1, 0].set_title('Latent Separation at t=0', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Separation at t=final
        axes[1, 1].plot(steps, sep_T_vals, 'o-', linewidth=2, markersize=8, color='#A23B72')
        axes[1, 1].set_xlabel('Training Step', fontsize=12)
        axes[1, 1].set_ylabel('Euclidean Distance', fontsize=12)
        axes[1, 1].set_title('Latent Separation at t=9', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    metrics_path = os.path.join(exp_dir, 'checkpoint_metrics.png')
    plt.savefig(metrics_path, dpi=150)
    print(f"Saved: {metrics_path}")
    plt.close()
    
    # 3. Save metrics to JSON
    import json
    metrics_json = {
        'checkpoints': [
            {
                'name': r['name'],
                'step': int(r['step']) if r['step'] != float('inf') else 'final',
                'acc_t0': float(r['acc_0']),
                'acc_t9': float(r['acc_T']),
                'separation_t0': float(r['separation_0']),
                'separation_t9': float(r['separation_T'])
            }
            for r in results
        ]
    }
    
    json_path = os.path.join(exp_dir, 'checkpoint_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Saved: {json_path}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for r in results:
        step_label = 'Final' if r['step'] == float('inf') else f"Step {r['step']:5d}"
        print(f"{step_label}: Acc {r['acc_T']:.2%} | Sep {r['separation_T']:.4f}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose all checkpoints in an experiment")
    parser.add_argument("--exp_dir", type=str, required=True,
                       help="Path to experiment directory")
    parser.add_argument("--data_path", type=str, default="data_release/hh_rlhf/gpt2",
                       help="Path to HH-RLHF dataset")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=10)
    args = parser.parse_args()
    
    main(
        exp_dir=args.exp_dir,
        data_path=args.data_path,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        seq_length=args.seq_length
    )

