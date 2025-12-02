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

# Import our custom modules
from .data_utils.sequential_pets_dataset import SequentialPetsDataset, sequential_collate_fn
from .data_utils.sequential_hh_dataset import SequentialHHDataset, sequential_hh_collate_fn
from .recurrent_vae_utils import RecurrentVAEModel


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
    seq_length: int
) -> dict:
    """
    Evaluate model and track per-timestep accuracy.
    
    Args:
        model: Trained RecurrentVAEModel
        dataloader: DataLoader for test episodes
        device: Device to run on
        seq_length: Length of sequences
    
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    model.to(device)
    
    # Storage for per-timestep accuracies
    accuracies_per_timestep = [[] for _ in range(seq_length)]
    all_rewards_chosen = [[] for _ in range(seq_length)]
    all_rewards_rejected = [[] for _ in range(seq_length)]
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            embeddings_chosen = batch['embeddings_chosen'].to(device)
            embeddings_rejected = batch['embeddings_rejected'].to(device)
            user_types = batch['user_type'].to(device)
            
            batch_size = embeddings_chosen.shape[0]
            
            # Initialize LSTM hidden and cell states
            h_curr = torch.zeros(batch_size, model.latent_dim).to(device)
            c_curr = torch.zeros(batch_size, model.latent_dim).to(device)
            
            # Process sequence timestep by timestep
            for t in range(seq_length):
                e_chosen = embeddings_chosen[:, t, :]
                e_rejected = embeddings_rejected[:, t, :]
                
                # Update belief
                mu, logvar, h_curr, c_curr = model.encoder(e_chosen, e_rejected, h_curr, c_curr)
                
                # Fix #1: Thompson Sampling - sample from posterior instead of using mean
                # This allows exploration based on uncertainty
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + std * eps
                
                # Predict rewards
                r_chosen, r_rejected = model.decoder(e_chosen, e_rejected, z)
                
                # Compute accuracy
                correct = (r_chosen > r_rejected).float()
                accuracies_per_timestep[t].append(correct.cpu())
                
                # Store rewards for analysis
                all_rewards_chosen[t].append(r_chosen.cpu())
                all_rewards_rejected[t].append(r_rejected.cpu())
    
    # Aggregate results
    mean_accuracies = []
    std_accuracies = []
    
    for t in range(seq_length):
        accs = torch.cat(accuracies_per_timestep[t])
        mean_accuracies.append(accs.mean().item())
        std_accuracies.append(accs.std().item())
    
    # Compute overall statistics
    overall_accuracy = np.mean(mean_accuracies)
    initial_accuracy = mean_accuracies[0]
    final_accuracy = mean_accuracies[-1]
    improvement = final_accuracy - initial_accuracy
    
    return {
        'mean_accuracies': mean_accuracies,
        'std_accuracies': std_accuracies,
        'overall_accuracy': overall_accuracy,
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'improvement': improvement,
    }


def plot_adaptation_curve(
    results: dict,
    seq_length: int,
    output_path: str
):
    """
    Create and save adaptation curve plot.
    
    Args:
        results: Dictionary with evaluation results
        seq_length: Length of sequences
        output_path: Path to save plot
    """
    mean_accs = results['mean_accuracies']
    std_accs = results['std_accuracies']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    timesteps = list(range(seq_length))
    
    # Plot mean accuracy with error bars
    ax.plot(timesteps, mean_accs, 'b-o', linewidth=2, markersize=8, label='Mean Accuracy')
    ax.fill_between(
        timesteps,
        [m - s for m, s in zip(mean_accs, std_accs)],
        [m + s for m, s in zip(mean_accs, std_accs)],
        alpha=0.3,
        color='blue'
    )
    
    # Add baseline (random guessing)
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Random Baseline')
    
    # Formatting
    ax.set_xlabel('Interaction Number (t)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Sequential Adaptation in Streaming VPL', fontsize=16, fontweight='bold')
    ax.set_xticks(timesteps)
    ax.set_ylim([0.4, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Add annotation showing improvement
    improvement = results['improvement']
    ax.text(
        0.98, 0.02,
        f"Improvement: {improvement:.1%}\n"
        f"Initial: {results['initial_accuracy']:.1%}\n"
        f"Final: {results['final_accuracy']:.1%}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
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
    
    print("="*80)
    print("Streaming VPL Evaluation")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}/{args.data_subset}")
    print(f"Seq Length: {args.seq_length}")
    print(f"Num Episodes: {args.num_eval_episodes}")
    print(f"Device: {args.device}")
    print("="*80 + "\n")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test dataset
    print("Loading test dataset...")
    
    # Detect dataset type based on path
    if "hh" in args.data_path.lower():
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
    results = evaluate_adaptation(
        model=model,
        dataloader=test_loader,
        device=args.device,
        seq_length=args.seq_length
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
    
    print("\nEvaluation complete!")
    
    # Check if adaptation is working
    if results['improvement'] > 0.1:
        print("✓ Strong adaptation observed! The model learns user preferences over time.")
    elif results['improvement'] > 0.05:
        print("~ Moderate adaptation observed. Consider tuning hyperparameters.")
    else:
        print("✗ Weak adaptation. Model may not be learning sequential patterns effectively.")


if __name__ == "__main__":
    main()


