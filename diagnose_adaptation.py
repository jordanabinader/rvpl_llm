"""
Diagnostic script to verify why accuracy is 100%

This script checks if the latent z at t=0 is already perfectly predictive
of the user type, which would explain why no adaptation is needed.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from hidden_context.data_utils.sequential_pets_dataset import SequentialPetsDataset, sequential_collate_fn
from hidden_context.recurrent_vae_utils import RecurrentVAEModel

def main():
    # Paths
    model_path = "experiments/streaming_vpl/both_seq10_latent512_beta0.1_seed0/final_checkpoint/model.pt"
    data_path = "data_release/simple_pets/gpt2"
    data_subset = "both"
    
    print("="*80)
    print("Diagnostic: Checking if z at t=0 is already predictive")
    print("="*80)
    
    # Load dataset
    test_dataset = SequentialPetsDataset(
        data_path=data_path,
        data_subset=data_subset,
        split="test",
        seq_length=10,
        epoch_size=100,
        seed=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=sequential_collate_fn
    )
    
    # Load model
    model = RecurrentVAEModel(
        encoder_embed_dim=768,
        decoder_embed_dim=768,
        hidden_dim=512,
        latent_dim=512
    )
    model.load_model(model_path)
    model.eval()
    
    # Collect statistics
    t0_latents = []
    t9_latents = []
    user_types = []
    t0_accuracies = []
    t9_accuracies = []
    
    with torch.no_grad():
        for batch in test_loader:
            embeddings_chosen = batch['embeddings_chosen']
            embeddings_rejected = batch['embeddings_rejected']
            user_type = batch['user_type']
            
            batch_size = embeddings_chosen.shape[0]
            h_curr = torch.zeros(batch_size, model.latent_dim)
            
            # Process sequence
            for t in range(10):
                e_chosen = embeddings_chosen[:, t, :]
                e_rejected = embeddings_rejected[:, t, :]
                
                # Update belief
                mu, logvar, h_curr = model.encoder(e_chosen, e_rejected, h_curr)
                z = mu
                
                # Predict rewards
                r_chosen, r_rejected = model.decoder(e_chosen, e_rejected, z)
                
                # Store results
                if t == 0:
                    t0_latents.append(z.cpu().numpy())
                    user_types.append(user_type.cpu().numpy())
                    t0_accuracies.append((r_chosen > r_rejected).float().cpu().numpy())
                elif t == 9:
                    t9_latents.append(z.cpu().numpy())
                    t9_accuracies.append((r_chosen > r_rejected).float().cpu().numpy())
    
    # Aggregate
    t0_latents = np.vstack(t0_latents)  # [N, latent_dim]
    t9_latents = np.vstack(t9_latents)
    user_types = np.concatenate(user_types)  # [N]
    t0_accuracies = np.concatenate(t0_accuracies)
    t9_accuracies = np.concatenate(t9_accuracies)
    
    print("\n" + "="*80)
    print("Results:")
    print("="*80)
    
    # Accuracy
    print(f"\nAccuracy at t=0: {t0_accuracies.mean():.2%}")
    print(f"Accuracy at t=9: {t9_accuracies.mean():.2%}")
    print(f"Improvement: {(t9_accuracies.mean() - t0_accuracies.mean()):.2%}")
    
    # Latent analysis
    print(f"\nLatent Statistics:")
    print(f"  t=0 mean norm: {np.linalg.norm(t0_latents, axis=1).mean():.3f}")
    print(f"  t=9 mean norm: {np.linalg.norm(t9_latents, axis=1).mean():.3f}")
    
    # Separability analysis
    dog_lovers = (user_types == 0)
    cat_lovers = (user_types == 1)
    
    if dog_lovers.sum() > 0 and cat_lovers.sum() > 0:
        # Compute mean latents for each user type
        t0_dog_mean = t0_latents[dog_lovers].mean(axis=0)
        t0_cat_mean = t0_latents[cat_lovers].mean(axis=0)
        t9_dog_mean = t9_latents[dog_lovers].mean(axis=0)
        t9_cat_mean = t9_latents[cat_lovers].mean(axis=0)
        
        # Distance between means
        t0_separation = np.linalg.norm(t0_dog_mean - t0_cat_mean)
        t9_separation = np.linalg.norm(t9_dog_mean - t9_cat_mean)
        
        print(f"\nUser Type Separation (Euclidean distance between mean latents):")
        print(f"  t=0: {t0_separation:.3f}")
        print(f"  t=9: {t9_separation:.3f}")
        print(f"  Ratio (t9/t0): {t9_separation/t0_separation:.2f}x")
        
        if t9_separation / t0_separation < 1.5:
            print("\n⚠️  WARNING: Latents at t=0 are already highly separated!")
            print("   This means the model identifies user types from the first observation.")
            print("   No meaningful adaptation is occurring over the sequence.")
        else:
            print("\n✓ Good: Latents become more separated over time (proper adaptation).")
        
        # Linear separability test
        from sklearn.linear_model import LogisticRegression
        
        clf_t0 = LogisticRegression(max_iter=1000)
        clf_t0.fit(t0_latents, user_types)
        t0_linear_acc = clf_t0.score(t0_latents, user_types)
        
        clf_t9 = LogisticRegression(max_iter=1000)
        clf_t9.fit(t9_latents, user_types)
        t9_linear_acc = clf_t9.score(t9_latents, user_types)
        
        print(f"\nLinear Separability (user type classification from z):")
        print(f"  t=0: {t0_linear_acc:.2%}")
        print(f"  t=9: {t9_linear_acc:.2%}")
        
        if t0_linear_acc > 0.95:
            print("\n❌ ISSUE CONFIRMED: z at t=0 already perfectly predicts user type!")
            print("   The first observation is sufficient to identify the user.")
            print("   This explains the 100% accuracy at all timesteps.")
            print("\n   Recommendations:")
            print("   1. Use more subtle preference distinctions (not explicit dog/cat keywords)")
            print("   2. Add noise to observations (stochastic preferences)")
            print("   3. Use more user types (4-8 instead of 2)")
            print("   4. Reduce latent_dim to force information compression")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

