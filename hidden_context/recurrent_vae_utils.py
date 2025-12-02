"""
Recurrent VAE utilities for Streaming VPL

Implements:
- RecurrentVPLEncoder: GRU-based sequential belief updates
- RecurrentVAEModel: Full model wrapping encoder and decoder
- Recursive KL divergence functions
- RecurrentVAETrainer: Training loop with KL annealing
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback
from transformers.optimization import get_cosine_schedule_with_warmup
from typing import Dict, Any

# Import PairEncoder and Decoder from existing vae_utils
from .vae_utils import PairEncoder, Decoder, HyperDecoder


def recursive_kl_div(curr_mu, curr_logvar, prev_mu, prev_logvar):
    """
    Compute KL divergence between two Gaussian distributions.
    
    KL(N(curr_mu, curr_var) || N(prev_mu, prev_var))
    = 0.5 * [log(var2/var1) + (var1 + (mu1-mu2)^2) / var2 - 1]
    
    Args:
        curr_mu: Current belief mean [batch, latent_dim]
        curr_logvar: Current belief log variance [batch, latent_dim]
        prev_mu: Previous belief mean [batch, latent_dim]
        prev_logvar: Previous belief log variance [batch, latent_dim]
    
    Returns:
        KL divergence scalar (averaged over batch and dimensions)
    """
    curr_var = torch.exp(curr_logvar)
    prev_var = torch.exp(prev_logvar)
    
    kl = 0.5 * (
        (prev_logvar - curr_logvar)  # log(var2/var1)
        + (curr_var + (curr_mu - prev_mu).pow(2)) / (prev_var + 1e-8)  # (var1 + delta^2) / var2
        - 1
    )
    return kl.sum(dim=-1).mean()


def standard_kl_div(mu, logvar):
    """
    Compute KL divergence to standard normal N(0,1).
    Used for t=0 (initial timestep).
    
    KL(N(mu, var) || N(0, 1)) = 0.5 * [mu^2 + var - 1 - log(var)]
    
    Args:
        mu: Belief mean [batch, latent_dim]
        logvar: Belief log variance [batch, latent_dim]
    
    Returns:
        KL divergence scalar (averaged over batch and dimensions)
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


class RecurrentVPLEncoder(nn.Module):
    """
    Recurrent encoder for Streaming VPL.
    
    Architecture:
    1. PairEncoder: Compares (chosen, rejected) → comparison feature (with LeakyReLU)
    2. GRU: Updates belief state based on comparison feature
    3. Projection: Maps hidden state → (mu, logvar) for VAE
    """
    
    def __init__(self, embed_dim: int, latent_dim: int, hidden_dim: int):
        """
        Args:
            embed_dim: Dimension of input embeddings
            latent_dim: Dimension of latent space (z)
            hidden_dim: Dimension of hidden layer in pair encoder
        """
        super(RecurrentVPLEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Step 1: Comparison logic (with non-linearity)
        # Reuse PairEncoder from vae_utils.py which has LeakyReLU
        # self.pair_encoder = PairEncoder(embed_dim, hidden_dim, latent_dim)
        # Todo
        # Instead of mapping to latent_dim directly, map to a smaller 'observation dim'
        # This forces the GRU to do the heavy lifting of integration, not just copying.
        self.obs_dim = 64 # significantly smaller than 512
        
        # Reuse PairEncoder logic but map to obs_dim
        self.pair_encoder = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.obs_dim), # Output smaller feature
            nn.LeakyReLU(0.2)
        )
        
        # Step 2: Memory logic (belief update)
        self.gru = nn.GRUCell(self.obs_dim, latent_dim)
        
        # Step 3: VAE projection
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, e_chosen, e_rejected, h_prev):
        """
        Process one timestep of the sequence.
        
        Args:
            e_chosen: Embeddings of chosen option [batch, embed_dim]
            e_rejected: Embeddings of rejected option [batch, embed_dim]
            h_prev: Previous hidden state [batch, latent_dim]
        
        Returns:
            mu: Belief mean [batch, latent_dim]
            logvar: Belief log variance [batch, latent_dim]
            h_curr: Current hidden state [batch, latent_dim]
        """
        pair_embed = torch.cat([e_chosen, e_rejected], dim=-1)
        # Step 1: Encode the comparison (observation)
        obs_feat = self.pair_encoder(pair_embed)  # [batch, latent_dim]
        
        # Step 2: Update hidden state (belief update via GRU)
        h_curr = self.gru(obs_feat, h_prev)  # [batch, latent_dim]
        
        # Step 3: Project to latent distribution parameters
        mu = self.fc_mu(h_curr)
        logvar = self.fc_logvar(h_curr)
        
        return mu, logvar, h_curr


class RecurrentVAEModel(nn.Module):
    """
    Full Recurrent VAE model for Streaming VPL.
    
    Processes sequences of preference comparisons and maintains
    a hidden belief state that adapts over time.
    """
    
    def __init__(
        self,
        encoder_embed_dim: int,
        decoder_embed_dim: int,
        hidden_dim: int,
        latent_dim: int
    ):
        """
        Args:
            encoder_embed_dim: Dimension of embeddings for encoder (context)
            decoder_embed_dim: Dimension of embeddings for decoder (target)
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of latent space
        """
        super(RecurrentVAEModel, self).__init__()
        
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Recurrent encoder
        self.encoder = RecurrentVPLEncoder(encoder_embed_dim, latent_dim, hidden_dim)
        
        # Decoder (reuse from vae_utils.py)
        # self.decoder = Decoder(decoder_embed_dim + latent_dim, hidden_dim)
        self.decoder = HyperDecoder(decoder_embed_dim, latent_dim)
    
    def reparameterization(self, mean, logvar):
        """
        Reparameterization trick for VAE.
        
        Args:
            mean: [batch, latent_dim]
            logvar: [batch, latent_dim]
        
        Returns:
            z: Sampled latent vector [batch, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        # Normalize to unit sphere scaled by sqrt(dim)
        z = F.normalize(z, p=2, dim=-1) * math.sqrt(z.shape[-1])
        return z
    
    def forward(
        self,
        embeddings_chosen,
        embeddings_rejected,
        return_trajectories=False,
        **kwargs  # Catch any extra arguments from Trainer (e.g., labels, user_type)
    ):
        """
        Forward pass through the entire sequence.
        
        Args:
            embeddings_chosen: [batch, seq_len, embed_dim]
            embeddings_rejected: [batch, seq_len, embed_dim]
            return_trajectories: If True, return per-timestep values
        
        Returns:
            If return_trajectories=True:
                Dict with 'mu', 'logvar', 'z', 'rewards_chosen', 'rewards_rejected'
                each of shape [batch, seq_len, ...]
            Else:
                Final timestep values only
        """
        batch_size, seq_len, embed_dim = embeddings_chosen.shape
        device = embeddings_chosen.device
        
        # Initialize hidden state
        h_curr = torch.zeros(batch_size, self.latent_dim).to(device)
        
        # Storage for trajectories
        if return_trajectories:
            all_mu = []
            all_logvar = []
            all_z = []
            all_rewards_chosen = []
            all_rewards_rejected = []
        
        # Process sequence
        for t in range(seq_len):
            e_chosen = embeddings_chosen[:, t, :]
            e_rejected = embeddings_rejected[:, t, :]
            
            # Update belief
            mu, logvar, h_curr = self.encoder(e_chosen, e_rejected, h_curr)
            
            # Sample z
            if self.training:
                z = self.reparameterization(mu, logvar)
            else:
                z = mu  # Use mean during evaluation
            
            # Predict rewards
            r_chosen, r_rejected = self.decoder(e_chosen, e_rejected, z)
            
            if return_trajectories:
                all_mu.append(mu)
                all_logvar.append(logvar)
                all_z.append(z)
                all_rewards_chosen.append(r_chosen)
                all_rewards_rejected.append(r_rejected)
        
        if return_trajectories:
            return {
                'mu': torch.stack(all_mu, dim=1),  # [batch, seq_len, latent_dim]
                'logvar': torch.stack(all_logvar, dim=1),
                'z': torch.stack(all_z, dim=1),
                'rewards_chosen': torch.stack(all_rewards_chosen, dim=1),  # [batch, seq_len, 1]
                'rewards_rejected': torch.stack(all_rewards_rejected, dim=1)
            }
        else:
            return mu, logvar, z, r_chosen, r_rejected
    
    def save_model(self, path: str):
        """Save model to disk."""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model from disk."""
        self.load_state_dict(torch.load(path))

class CyclicalKLAnnealer:
    """
    Cyclical KL annealing to repeatedly warm up the latent space.
    """
    def __init__(self, max_beta: float, total_steps: int, n_cycles: int = 4, ratio: float = 0.5):
        """
        Args:
            max_beta: Maximum KL weight
            total_steps: Total training steps
            n_cycles: Number of cycles to repeat
            ratio: Portion of cycle used for increasing beta (vs holding flat)
        """
        self.max_beta = max_beta
        self.total_steps = total_steps
        self.n_cycles = n_cycles
        self.ratio = ratio
        self.period = total_steps // n_cycles
        self.current_step = 0
    
    def get_beta(self) -> float:
        cycle_idx = self.current_step // self.period
        cycle_step = self.current_step % self.period
        
        # Normalized progress within cycle (0 to 1)
        tau = cycle_step / self.period
        
        if tau > self.ratio:
            return self.max_beta
        else:
            # Linear increase during ratio portion
            return self.max_beta * (tau / self.ratio)
    
    def step(self):
        self.current_step += 1


class KLAnnealer:
    """
    KL annealing to prevent posterior collapse.
    
    Linearly increases beta from beta_start to beta_end over total_steps.
    """
    
    def __init__(self, beta_start: float, beta_end: float, total_steps: int):
        """
        Args:
            beta_start: Initial KL weight (typically 0.0)
            beta_end: Final KL weight (e.g., 0.1)
            total_steps: Number of steps to anneal over
        """
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_steps = total_steps
        self.current_step = 0
    
    def get_beta(self) -> float:
        """Get current beta value."""
        if self.current_step >= self.total_steps:
            return self.beta_end
        
        # Linear annealing
        progress = self.current_step / self.total_steps
        return self.beta_start + (self.beta_end - self.beta_start) * progress
    
    def step(self):
        """Increment step counter."""
        self.current_step += 1


class RecurrentVAETrainer(Trainer):
    """
    Custom trainer with temporal weighting and cyclical annealing.
    """
    
    def __init__(
        self,
        *args,
        seq_length: int = 10,
        latent_dim: int = 512,
        beta_max: float = 0.1,
        beta_cycles: int = 4,
        temporal_gamma: float = 1.1, # Weight later timesteps more
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.temporal_gamma = temporal_gamma
        
        # Estimate total steps based on dataset size and batch size
        # This is an approximation; Trainer usually handles this but we need it for annealing
        if self.args.max_steps > 0:
            total_steps = self.args.max_steps
        else:
             # Fallback estimation
            total_steps = len(self.train_dataset) * self.args.num_train_epochs // self.args.per_device_train_batch_size
            
        self.kl_annealer = CyclicalKLAnnealer(beta_max, total_steps, n_cycles=beta_cycles)
        
        print(f"RecurrentVAETrainer (Refactored) initialized:")
        print(f"  Seq Length: {seq_length}")
        print(f"  Temporal Gamma: {temporal_gamma}")
        print(f"  Cyclical Annealing: Max Beta {beta_max}, Cycles {beta_cycles}")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        embeddings_chosen = inputs['embeddings_chosen']
        embeddings_rejected = inputs['embeddings_rejected']
        
        batch_size, seq_len, embed_dim = embeddings_chosen.shape
        device = embeddings_chosen.device
        
        h_curr = torch.zeros(batch_size, self.latent_dim).to(device)
        prev_mu = torch.zeros(batch_size, self.latent_dim).to(device)
        prev_logvar = torch.zeros(batch_size, self.latent_dim).to(device)
        
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        
        if self.model.training and not return_outputs:
            self.kl_annealer.step()

        beta = self.kl_annealer.get_beta()
        
        if return_outputs:
            all_rewards_chosen, all_rewards_rejected = [], []
            all_mu, all_logvar = [], []
        
        for t in range(seq_len):
            e_chosen = embeddings_chosen[:, t, :]
            e_rejected = embeddings_rejected[:, t, :]
            
            # Encoder
            curr_mu, curr_logvar, h_curr = model.encoder(e_chosen, e_rejected, h_curr)
            curr_mu = torch.clamp(curr_mu, -1, 1)
            curr_logvar = torch.clamp(curr_logvar, -1, 1)
            
            # Sample
            std = torch.exp(0.5 * curr_logvar)
            eps = torch.randn_like(std)
            z = curr_mu + eps * std
            z = F.normalize(z, p=2, dim=-1) * math.sqrt(z.shape[-1])
            
            # Decoder
            r_chosen, r_rejected = model.decoder(e_chosen, e_rejected, z)
            
            # Losses
            # 1. Temporal Weighting: Care more about later timesteps
            time_weight = self.temporal_gamma ** t
            recon_loss = -F.logsigmoid(r_chosen - r_rejected).mean()
            
            # 2. KL Divergence
            if t == 0:
                kl_loss = standard_kl_div(curr_mu, curr_logvar)
            else:
                kl_loss = recursive_kl_div(curr_mu, curr_logvar, prev_mu, prev_logvar)
            
            # 3. KL Threshold (Free Bits) - prevents crushing small useful info
            # Only penalize if KL > 0.05
            kl_threshold = 0.05
            kl_loss_clipped = torch.max(kl_loss, torch.tensor(kl_threshold).to(device))
            # We subtract threshold so the gradient is zero when below threshold, 
            # but we add it back to the metric so we see the real KL
            kl_term = beta * (kl_loss_clipped - kl_threshold) 

            # Step Loss
            step_loss = (time_weight * recon_loss) + kl_term
            
            total_loss += step_loss
            total_recon += recon_loss
            total_kl += kl_loss
            
            prev_mu = curr_mu.detach()
            prev_logvar = curr_logvar.detach()
            
            if return_outputs:
                all_rewards_chosen.append(r_chosen.detach())
                all_rewards_rejected.append(r_rejected.detach())
                all_mu.append(curr_mu.detach())
                all_logvar.append(curr_logvar.detach())
        
        # Normalize loss
        total_loss = total_loss / seq_len
        avg_recon = total_recon / seq_len
        avg_kl = total_kl / seq_len
        
        if not return_outputs and self.state.global_step % 10 == 0:
            self.log({
                'train_loss': total_loss.item(),
                'train_recon': avg_recon.item(),
                'train_kl': avg_kl.item(),
                'beta': beta
            })
        
        if return_outputs:
            return total_loss, {
                'rewards_chosen': torch.stack(all_rewards_chosen, dim=1),
                'rewards_rejected': torch.stack(all_rewards_rejected, dim=1),
                'mu': torch.stack(all_mu, dim=1),
                'logvar': torch.stack(all_logvar, dim=1),
            }
        else:
            return total_loss
    
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.03 * num_training_steps),
            num_training_steps=num_training_steps
        )
        self.lr_scheduler = scheduler
        return scheduler
    
    def on_step_end(self, args, state, control, **kwargs):
        self.kl_annealer.step()
        return super().on_step_end(args, state, control, **kwargs)

    @classmethod
    def compute_metrics(cls, eval_prediction):
        # ... (Metrics logic remains identical to previous version) ...
        # Copied for completeness
        rewards_chosen, rewards_rejected, mu, logvar = eval_prediction.predictions
        rewards_chosen = torch.from_numpy(rewards_chosen)
        rewards_rejected = torch.from_numpy(rewards_rejected)
        mu = torch.from_numpy(mu)
        logvar = torch.from_numpy(logvar)
        
        accuracy = (rewards_chosen > rewards_rejected).float().mean()
        
        seq_len = rewards_chosen.shape[1]
        per_timestep_acc = []
        for t in range(seq_len):
            acc_t = (rewards_chosen[:, t] > rewards_rejected[:, t]).float().mean()
            per_timestep_acc.append(acc_t.item())
        
        btl_loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
        kl_loss = standard_kl_div(mu[:, -1], logvar[:, -1])
        
        metrics = {
            'accuracy': accuracy.item(),
            'btl_loss': btl_loss.item(),
            'kl_loss': kl_loss.item(),
        }
        for t, acc in enumerate(per_timestep_acc):
            metrics[f'acc_t{t}'] = acc
        
        return metrics