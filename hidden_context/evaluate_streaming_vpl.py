"""
Evaluation script for Streaming VPL

Generates adaptation curves showing how accuracy improves as the model
observes more interactions from the same user within an episode.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import HfArgumentParser
import matplotlib.pyplot as plt
import wandb

# Import our custom modules
from .data_utils.sequential_pets_dataset import SequentialPetsDataset, sequential_collate_fn
from .data_utils.sequential_hh_dataset import SequentialHHDataset, sequential_hh_collate_fn
from .data_utils.sequential_prism_dataset import SequentialPRISMDataset, sequential_prism_collate_fn
from .recurrent_vae_utils import RecurrentVAEModel, TransformerVAEModel


@dataclass
class EvalArguments:
    """Arguments for evaluating Streaming VPL."""
    
    # Model arguments
    model_path: str = field(
        metadata={"help": "Path to trained model checkpoint (.pt file)"}
    )
    
    # Data arguments
    data_path: str = field(
        default="data_release/simple_pets/gpt2",
        metadata={"help": "Path to the Pets dataset directory"}
    )
    data_subset: str = field(
        default="harmless",
        metadata={"help": "Which subset to use: 'harmless' or 'helpful'"}
    )
    
    # Model architecture (must match training)
    encoder_embed_dim: int = field(default=768)
    decoder_embed_dim: int = field(default=768)
    latent_dim: int = field(default=512)
    hidden_dim: int = field(default=512)
    use_transformer: bool = field(
        default=False,
        metadata={"help": "Use Transformer architecture (must match training)"}
    )
    num_attention_heads: int = field(default=4)
    num_transformer_layers: int = field(default=2)
    
    # Evaluation arguments
    seq_length: int = field(
        default=10,
        metadata={"help": "Number of interactions per episode (T)"}
    )
    num_eval_episodes: int = field(
        default=200,
        metadata={"help": "Number of episodes to evaluate on"}
    )
    batch_size: int = field(default=16)
    use_mean: bool = field(
        default=False,
        metadata={"help": "Use mean (mu) instead of sampling during eval"}
    )
    
    # Output arguments
    output_dir: str = field(
        default="experiments/streaming_vpl/evaluation",
        metadata={"help": "Directory to save plots and results"}
    )
    
    # Misc
    seed: int = field(default=0)
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to run evaluation on"}
    )


def evaluate_adaptation(
    model: RecurrentVAEModel,
    dataloader: DataLoader,
    device: str,
    seq_length: int,
    use_mean: bool = False
) -> dict:
    """
    Evaluate model and track per-timestep accuracy.
    
    Args:
        model: Trained RecurrentVAEModel
        dataloader: DataLoader for test episodes
        device: Device to run on
        seq_length: Length of sequences
        use_mean: If True, use mean (mu) instead of sampling (z ~ q(z|...))
    
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    model.to(device)
    
    # Storage for per-timestep accuracies and uncertainty metrics
    accuracies_per_timestep = [[] for _ in range(seq_length)]
    all_rewards_chosen = [[] for _ in range(seq_length)]
    all_rewards_rejected = [[] for _ in range(seq_length)]
    all_variances = [[] for _ in range(seq_length)]  # Track uncertainty (variance)
    all_logvars = [[] for _ in range(seq_length)]  # Track raw logvar for plotting
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            embeddings_chosen = batch['embeddings_chosen'].to(device)
            embeddings_rejected = batch['embeddings_rejected'].to(device)
            mask = batch.get('mask', None)  # Get mask if available
            if mask is not None:
                mask = mask.to(device)
            
            # Handle both user_type (HH-RLHF, Pets) and user_ids (PRISM)
            if 'user_type' in batch:
                user_types = batch['user_type'].to(device)
            # user_ids not needed for evaluation, just for logging
            
            batch_size = embeddings_chosen.shape[0]
            
            # Check if using Transformer or Recurrent architecture
            is_transformer = isinstance(model, TransformerVAEModel)
            
            if is_transformer:
                # Transformer: encode all pairs first, then attend for each timestep
                all_pairs = []
                for t in range(seq_length):
                    e_chosen = embeddings_chosen[:, t, :]
                    e_rejected = embeddings_rejected[:, t, :]
                    # Apply contrastive encoding: concatenate chosen, rejected, interaction, and difference
                    pair = torch.cat([
                        e_chosen,
                        e_rejected,
                        e_chosen * e_rejected,  # Interaction term
                        e_chosen - e_rejected   # Difference term
                    ], dim=-1)
                    pair_encoded = model.encoder.pair_encoder(pair)
                    all_pairs.append(pair_encoded)
                sequence_pairs = torch.stack(all_pairs, dim=1)
                
                # Process each timestep with attention
                for t in range(seq_length):
                    # Skip if all samples are padded at this timestep
                    if mask is not None:
                        step_mask = mask[:, t]
                        if not step_mask.any():
                            continue
                    
                    # KEY FIX: At timestep t, use belief from observations 0...t-1 only
                    if t == 0:
                        # No observations yet, use prior
                        mu = torch.zeros(batch_size, model.latent_dim).to(device)
                        logvar = torch.zeros(batch_size, model.latent_dim).to(device)
                    else:
                        # Attend to observations 0 to t-1 (not including t)
                        mu, logvar = model.encoder(sequence_pairs[:, :t, :], current_timestep=t-1)
                    
                    # Sample z from belief (Thompson sampling) or use mean
                    if use_mean:
                        z = mu
                    else:
                        std = torch.exp(0.5 * logvar)
                        eps = torch.randn_like(std)
                        z = mu + std * eps
                    
                    # Get observation t (which belief hasn't seen yet)
                    e_chosen = embeddings_chosen[:, t, :]
                    e_rejected = embeddings_rejected[:, t, :]
                    
                    # Predict on observation t
                    r_chosen, r_rejected = model.decoder(e_chosen, e_rejected, z)
                    
                    # Compute accuracy (exclude padded samples entirely)
                    correct = (r_chosen > r_rejected).float().squeeze(-1)  # [batch]
                    
                    # Compute uncertainty: average variance across latent dimensions
                    variance = torch.exp(logvar)  # Convert log variance to variance
                    avg_variance = variance.mean(dim=-1)  # [batch]
                    
                    if mask is not None:
                        valid = step_mask.bool()  # [batch]
                        correct = correct[valid]  # keep only real timesteps
                        r_chosen = r_chosen[valid]
                        r_rejected = r_rejected[valid]
                        avg_variance = avg_variance[valid]
                        logvar_store = logvar[valid]
                    else:
                        logvar_store = logvar
                    
                    accuracies_per_timestep[t].append(correct.cpu())
                    all_rewards_chosen[t].append(r_chosen.cpu())
                    all_rewards_rejected[t].append(r_rejected.cpu())
                    all_variances[t].append(avg_variance.cpu())
                    all_logvars[t].append(logvar_store.cpu())
            else:
                # Recurrent: maintain LSTM hidden and cell states
                h_curr = torch.zeros(batch_size, model.latent_dim).to(device)
                c_curr = torch.zeros(batch_size, model.latent_dim).to(device)
                mu = torch.zeros(batch_size, model.latent_dim).to(device)
                logvar = torch.zeros(batch_size, model.latent_dim).to(device)
                
                # Process sequence timestep by timestep
                for t in range(seq_length):
                    # Skip if all samples are padded at this timestep
                    if mask is not None:
                        step_mask = mask[:, t]
                        if not step_mask.any():
                            continue
                    
                    # KEY FIX: At timestep t, use belief from observations 0...t-1
                    # (mu, logvar already set from previous iteration, or prior for t=0)
                    
                    # Thompson Sampling from current belief or use mean
                    if use_mean:
                        z = mu
                    else:
                        std = torch.exp(0.5 * logvar)
                        eps = torch.randn_like(std)
                        z = mu + std * eps
                    
                    # Get observation t (which belief hasn't seen yet)
                    e_chosen = embeddings_chosen[:, t, :]
                    e_rejected = embeddings_rejected[:, t, :]
                    
                    # Predict on observation t
                    r_chosen, r_rejected = model.decoder(e_chosen, e_rejected, z)
                    
                    # Compute accuracy (exclude padded samples entirely)
                    correct = (r_chosen > r_rejected).float().squeeze(-1)  # [batch]
                    
                    # Compute uncertainty: average variance across latent dimensions
                    variance = torch.exp(logvar)  # Convert log variance to variance
                    avg_variance = variance.mean(dim=-1)  # [batch]
                    
                    if mask is not None:
                        valid = step_mask.bool()  # [batch]
                        correct = correct[valid]  # keep only real timesteps
                        r_chosen_store = r_chosen[valid]
                        r_rejected_store = r_rejected[valid]
                        avg_variance = avg_variance[valid]
                        logvar_store = logvar[valid]
                    else:
                        r_chosen_store = r_chosen
                        r_rejected_store = r_rejected
                        logvar_store = logvar
                    
                    accuracies_per_timestep[t].append(correct.cpu())
                    all_rewards_chosen[t].append(r_chosen_store.cpu())
                    all_rewards_rejected[t].append(r_rejected_store.cpu())
                    all_variances[t].append(avg_variance.cpu())
                    all_logvars[t].append(logvar_store.cpu())
                    
                    # Now update belief with observation t for next iteration
                    mu, logvar, h_curr, c_curr = model.encoder(e_chosen, e_rejected, h_curr, c_curr)
    
    # Aggregate results
    mean_accuracies = []
    std_accuracies = []
    mean_variances = []
    std_variances = []
    valid_counts = []
    
    for t in range(seq_length):
        accs = torch.cat(accuracies_per_timestep[t])
        vars = torch.cat(all_variances[t])
        valid_counts.append(accs.numel())
        mean_accuracies.append(accs.mean().item())
        std_accuracies.append(accs.std().item())
        mean_variances.append(vars.mean().item())
        std_variances.append(vars.std().item())
    
    # Print valid sample counts per timestep for debugging
    print("\nValid samples per timestep:")
    for t in range(seq_length):
        print(f"  t={t}: {valid_counts[t]} samples, accuracy={mean_accuracies[t]:.2%}, variance={mean_variances[t]:.4f}")
    
    # Compute overall statistics
    # Only use timesteps with sufficient data (min 50 samples or 25% of episodes)
    MIN_SAMPLES = max(50, len(accuracies_per_timestep[0]) // 4)
    usable_timesteps = [t for t, n in enumerate(valid_counts) if n >= MIN_SAMPLES]
    
    if len(usable_timesteps) < 2:
        print(f"\n⚠️  Warning: Only {len(usable_timesteps)} timesteps have >= {MIN_SAMPLES} samples")
        print(f"    Using all timesteps for metrics (may be noisy)")
        usable_timesteps = list(range(seq_length))
    
    overall_accuracy = np.mean(mean_accuracies)
    initial_accuracy = mean_accuracies[0]
    
    # Use last usable timestep for "final" accuracy (not noisy tail)
    final_t = usable_timesteps[-1]
    final_accuracy = mean_accuracies[final_t]
    improvement = final_accuracy - initial_accuracy
    
    print(f"\nUsing t={final_t} (n={valid_counts[final_t]}) for final accuracy (not t={seq_length-1} with n={valid_counts[-1]})")
    
    # Compute uncertainty reduction (initial variance - final variance)
    initial_variance = mean_variances[0]
    final_variance = mean_variances[final_t]
    uncertainty_reduction = initial_variance - final_variance
    
    print(f"\nUncertainty Metrics:")
    print(f"  Initial variance: {initial_variance:.4f}")
    print(f"  Final variance: {final_variance:.4f}")
    print(f"  Uncertainty reduction: {uncertainty_reduction:.4f} ({-100*uncertainty_reduction/initial_variance:.1f}%)")
    
    return {
        'mean_accuracies': mean_accuracies,
        'std_accuracies': std_accuracies,
        'mean_variances': mean_variances,
        'std_variances': std_variances,
        'overall_accuracy': overall_accuracy,
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'improvement': improvement,
        'initial_variance': initial_variance,
        'final_variance': final_variance,
        'uncertainty_reduction': uncertainty_reduction,
    }


def plot_adaptation_curve(
    results: dict,
    seq_length: int,
    output_path: str
):
    """
    Create and save adaptation curve plot with accuracy and uncertainty.
    
    Args:
        results: Dictionary with evaluation results
        seq_length: Length of sequences
        output_path: Path to save plot
    """
    mean_accs = results['mean_accuracies']
    std_accs = results['std_accuracies']
    mean_vars = results['mean_variances']
    std_vars = results['std_variances']
    
    # Create figure with two subplots: accuracy and uncertainty
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    timesteps = list(range(seq_length))
    
    # ========== SUBPLOT 1: Accuracy ==========
    # Plot mean accuracy with error bars
    ax1.plot(timesteps, mean_accs, 'b-o', linewidth=2, markersize=8, label='Mean Accuracy')
    ax1.fill_between(
        timesteps,
        [m - s for m, s in zip(mean_accs, std_accs)],
        [m + s for m, s in zip(mean_accs, std_accs)],
        alpha=0.3,
        color='blue'
    )
    
    # Add baseline (random guessing)
    ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Random Baseline')
    
    # Formatting
    ax1.set_xlabel('Interaction Number (t)', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.set_title('Sequential Adaptation in Streaming VPL', fontsize=16, fontweight='bold')
    ax1.set_xticks(timesteps)
    ax1.set_ylim([0.4, 1.0])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Add annotation showing improvement
    improvement = results['improvement']
    ax1.text(
        0.98, 0.02,
        f"Improvement: {improvement:.1%}\n"
        f"Initial: {results['initial_accuracy']:.1%}\n"
        f"Final: {results['final_accuracy']:.1%}",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # ========== SUBPLOT 2: Uncertainty (Variance) ==========
    # Plot mean variance with error bars
    ax2.plot(timesteps, mean_vars, 'g-s', linewidth=2, markersize=8, label='Mean Variance')
    ax2.fill_between(
        timesteps,
        [m - s for m, s in zip(mean_vars, std_vars)],
        [m + s for m, s in zip(mean_vars, std_vars)],
        alpha=0.3,
        color='green'
    )
    
    # Formatting
    ax2.set_xlabel('Interaction Number (t)', fontsize=14)
    ax2.set_ylabel('Posterior Variance', fontsize=14)
    ax2.set_title('Uncertainty Reduction over Time', fontsize=16, fontweight='bold')
    ax2.set_xticks(timesteps)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    # Add annotation showing uncertainty reduction
    uncertainty_reduction = results['uncertainty_reduction']
    reduction_pct = -100 * uncertainty_reduction / results['initial_variance'] if results['initial_variance'] > 0 else 0
    ax2.text(
        0.98, 0.98,
        f"Uncertainty Reduction: {reduction_pct:.1f}%\n"
        f"Initial: {results['initial_variance']:.4f}\n"
        f"Final: {results['final_variance']:.4f}",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    """Main evaluation function."""
    # Parse arguments
    parser = HfArgumentParser(EvalArguments)
    args: EvalArguments = parser.parse_args_into_dataclasses()[0]
    
    # Try to load config from checkpoint directory
    import json
    checkpoint_dir = os.path.dirname(args.model_path)
    config_path = os.path.join(checkpoint_dir, "config.json")
    
    if os.path.exists(config_path):
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        # Override args with saved config (but allow command-line overrides for some fields)
        # Only override if the arg wasn't explicitly set
        import sys
        explicit_args = set()
        for i, arg in enumerate(sys.argv):
            if arg.startswith('--'):
                explicit_args.add(arg[2:])
        
        # Map of config keys to arg names
        config_to_arg = {
            'seq_length': 'seq_length',
            'latent_dim': 'latent_dim',
            'hidden_dim': 'hidden_dim',
            'encoder_embed_dim': 'encoder_embed_dim',
            'decoder_embed_dim': 'decoder_embed_dim',
            'use_transformer': 'use_transformer',
            'num_attention_heads': 'num_attention_heads',
            'num_transformer_layers': 'num_transformer_layers',
            'transformer_dropout': 'transformer_dropout',
        }
        
        for config_key, arg_name in config_to_arg.items():
            if config_key in saved_config and saved_config[config_key] is not None:
                if arg_name not in explicit_args:
                    setattr(args, arg_name, saved_config[config_key])
                    print(f"  {arg_name}: {saved_config[config_key]} (from config)")
                else:
                    print(f"  {arg_name}: {getattr(args, arg_name)} (from command line, overriding config)")
        print()
    else:
        print(f"Warning: No config.json found at {config_path}")
        print("Using command-line arguments only.")
        print()
    
    print("="*80)
    print("Streaming VPL Evaluation")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}/{args.data_subset}")
    print(f"Seq Length: {args.seq_length}")
    print(f"Latent Dim: {args.latent_dim}")
    print(f"Hidden Dim: {args.hidden_dim}")
    print(f"Num Episodes: {args.num_eval_episodes}")
    print(f"Device: {args.device}")
    print(f"Use Transformer: {args.use_transformer}")
    print("="*80 + "\n")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize W&B for evaluation logging
    # Extract experiment name from model path
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(args.model_path)))
    wandb.init(
        project="streaming-vpl-prism" if "prism" in args.data_path.lower() else "streaming-vpl",
        name=f"{exp_name}_eval",
        config={
            "model_path": args.model_path,
            "data_path": args.data_path,
            "data_subset": getattr(args, 'data_subset', 'N/A'),
            "seq_length": args.seq_length,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "num_eval_episodes": args.num_eval_episodes,
            "seed": args.seed,
        },
        tags=["evaluation", "adaptation_curve"]
    )
    
    # Load test dataset
    print("Loading test dataset...")
    
    # Detect dataset type based on path
    if "prism" in args.data_path.lower():
        print("Using PRISM dataset")
        test_dataset = SequentialPRISMDataset(
            data_path=args.data_path,
            split="test",
            seq_length=args.seq_length,
            epoch_size=args.num_eval_episodes,
            seed=args.seed
        )
        collate_fn = sequential_prism_collate_fn
    elif "hh" in args.data_path.lower():
        print("Using HH-RLHF dataset")
        test_dataset = SequentialHHDataset(
            data_path=args.data_path,
            data_subset=args.data_subset,
            split="test",
            seq_length=args.seq_length,
            epoch_size=args.num_eval_episodes,
            seed=args.seed
        )
        collate_fn = sequential_hh_collate_fn
    else:
        print("Using Pets dataset")
        test_dataset = SequentialPetsDataset(
            data_path=args.data_path,
            data_subset=args.data_subset,
            split="test",
            seq_length=args.seq_length,
            epoch_size=args.num_eval_episodes,
            seed=args.seed
        )
        collate_fn = sequential_collate_fn
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Test dataset: {len(test_dataset)} episodes")
    
    # Initialize model
    print("\nInitializing model...")
    if args.use_transformer:
        print("Using Transformer architecture")
        model = TransformerVAEModel(
            encoder_embed_dim=args.encoder_embed_dim,
            decoder_embed_dim=args.decoder_embed_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_heads=args.num_attention_heads,
            num_layers=args.num_transformer_layers
        )
    else:
        print("Using Recurrent (LSTM) architecture")
        model = RecurrentVAEModel(
            encoder_embed_dim=args.encoder_embed_dim,
            decoder_embed_dim=args.decoder_embed_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim
        )
    
    # Load trained weights
    print(f"Loading model from: {args.model_path}")
    model.load_model(args.model_path)
    
    # Evaluate
    print("\nEvaluating adaptation...")
    if args.use_mean:
        print("Using mean (mu) for predictions (no sampling)")
    else:
        print("Using Thompson sampling (sampling from posterior)")
    results = evaluate_adaptation(
        model=model,
        dataloader=test_loader,
        device=args.device,
        seq_length=args.seq_length,
        use_mean=args.use_mean
    )
    
    # Print results
    print("\n" + "="*80)
    print("Results:")
    print("="*80)
    print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
    print(f"Initial Accuracy (t=0): {results['initial_accuracy']:.2%}")
    print(f"Final Accuracy (t={args.seq_length-1}): {results['final_accuracy']:.2%}")
    print(f"Improvement: {results['improvement']:.2%}")
    print("\nPer-timestep accuracies:")
    for t, (mean, std) in enumerate(zip(results['mean_accuracies'], results['std_accuracies'])):
        print(f"  t={t}: {mean:.2%} ± {std:.2%}")
    print("="*80 + "\n")
    
    # Save results as JSON
    json_path = os.path.join(args.output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Create and save plot
    plot_path = os.path.join(args.output_dir, 'adaptation_curve.png')
    plot_adaptation_curve(results, args.seq_length, plot_path)
    
    # Log everything to W&B
    print("\nLogging results to W&B...")
    
    # Log summary metrics
    wandb.log({
        "eval/overall_accuracy": results['overall_accuracy'],
        "eval/initial_accuracy": results['initial_accuracy'],
        "eval/final_accuracy": results['final_accuracy'],
        "eval/improvement": results['improvement'],
        "eval/initial_variance": results['initial_variance'],
        "eval/final_variance": results['final_variance'],
        "eval/uncertainty_reduction": results['uncertainty_reduction'],
    })
    
    # Log per-timestep accuracies and variances as a table
    timestep_table = wandb.Table(
        columns=["timestep", "accuracy", "accuracy_std", "variance", "variance_std"],
        data=[[t, acc_mean, acc_std, var_mean, var_std] 
              for t, (acc_mean, acc_std, var_mean, var_std) in enumerate(
                  zip(results['mean_accuracies'], results['std_accuracies'],
                      results['mean_variances'], results['std_variances']))]
    )
    wandb.log({"eval/timestep_metrics": timestep_table})
    
    # Log adaptation curve as image
    wandb.log({"eval/adaptation_curve": wandb.Image(plot_path)})
    
    # Log per-timestep accuracies and variances as line plots
    for t, (acc, var) in enumerate(zip(results['mean_accuracies'], results['mean_variances'])):
        wandb.log({
            "eval/accuracy_by_timestep": acc,
            "eval/variance_by_timestep": var
        }, step=t)
    
    # Log improvement visualization
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Initial (t=0)', f'Final (t={args.seq_length-1})'], 
           [results['initial_accuracy'], results['final_accuracy']],
           color=['#e74c3c', '#27ae60'])
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1])
    ax.set_title('Adaptation: Initial vs Final Accuracy')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate([results['initial_accuracy'], results['final_accuracy']]):
        ax.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    wandb.log({"eval/initial_vs_final": wandb.Image(fig)})
    plt.close(fig)
    
    print("\nEvaluation complete!")
    
    # Check if adaptation is working
    if results['improvement'] > 0.1:
        adaptation_status = "✓ Strong adaptation observed! The model learns user preferences over time."
        wandb.run.summary["adaptation_status"] = "strong"
    elif results['improvement'] > 0.05:
        adaptation_status = "~ Moderate adaptation observed. Consider tuning hyperparameters."
        wandb.run.summary["adaptation_status"] = "moderate"
    else:
        adaptation_status = "✗ Weak adaptation. Model may not be learning sequential patterns effectively."
        wandb.run.summary["adaptation_status"] = "weak"
    
    print(adaptation_status)
    wandb.run.summary["adaptation_message"] = adaptation_status
    
    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()


