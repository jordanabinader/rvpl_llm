"""
Compare Beta Experiments Script

Comprehensive comparison of experiments with different beta values.
Generates multi-panel visualizations comparing:
1. Training curves (loss, reconstruction, KL divergence)
2. t-SNE progression at different checkpoints
3. Adaptation curves from evaluation
4. Latent separation metrics over training
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from PIL import Image


def find_experiment_dir(base_dir, beta_value):
    """Find experiment directory for a specific beta value."""
    pattern = f"*beta{beta_value}*"
    matches = glob.glob(os.path.join(base_dir, pattern))
    if not matches:
        return None
    # Return the most recent if multiple matches
    return sorted(matches)[-1]


def load_checkpoint_metrics(exp_dir):
    """Load checkpoint metrics JSON."""
    metrics_path = os.path.join(exp_dir, 'checkpoint_metrics.json')
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def load_evaluation_results(exp_dir):
    """Load evaluation results JSON."""
    results_path = os.path.join(exp_dir, 'evaluation', 'results.json')
    if not os.path.exists(results_path):
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_comparison(exp_dirs, beta_values, output_path):
    """Generate comprehensive comparison plot."""
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']
    
    # ========================================
    # Row 1: Checkpoint Metrics Over Training
    # ========================================
    
    # Load checkpoint metrics
    all_metrics = []
    for exp_dir, beta in zip(exp_dirs, beta_values):
        metrics = load_checkpoint_metrics(exp_dir)
        if metrics:
            all_metrics.append((beta, metrics))
    
    if all_metrics:
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        
        for idx, (beta, metrics) in enumerate(all_metrics):
            checkpoints = [c for c in metrics['checkpoints'] if c['step'] != 'final']
            if not checkpoints:
                continue
            
            steps = [c['step'] for c in checkpoints]
            acc_t0 = [c['acc_t0'] for c in checkpoints]
            acc_t9 = [c['acc_t9'] for c in checkpoints]
            sep_t9 = [c['separation_t9'] for c in checkpoints]
            
            color = colors[idx % len(colors)]
            
            # Accuracy at t=0
            ax1.plot(steps, acc_t0, 'o-', label=f'β={beta}', 
                    linewidth=2, markersize=6, color=color)
            
            # Accuracy at t=9
            ax2.plot(steps, acc_t9, 'o-', label=f'β={beta}',
                    linewidth=2, markersize=6, color=color)
            
            # Improvement
            improvement = np.array(acc_t9) - np.array(acc_t0)
            ax3.plot(steps, improvement * 100, 'o-', label=f'β={beta}',
                    linewidth=2, markersize=6, color=color)
            
            # Latent separation
            ax4.plot(steps, sep_t9, 'o-', label=f'β={beta}',
                    linewidth=2, markersize=6, color=color)
        
        # Format axes
        ax1.set_title('Initial Accuracy (t=0)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Training Step', fontsize=10)
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        ax2.set_title('Final Accuracy (t=9)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Training Step', fontsize=10)
        ax2.set_ylabel('Accuracy', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        ax3.set_title('Adaptation (t=9 - t=0)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Training Step', fontsize=10)
        ax3.set_ylabel('Improvement (%)', fontsize=10)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        ax4.set_title('Latent Separation (t=9)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Training Step', fontsize=10)
        ax4.set_ylabel('Euclidean Distance', fontsize=10)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
    
    # ========================================
    # Row 2: Evaluation Adaptation Curves
    # ========================================
    
    ax_adapt = fig.add_subplot(gs[1, :2])
    ax_table = fig.add_subplot(gs[1, 2:])
    ax_table.axis('off')
    
    # Load evaluation results
    eval_results = []
    for exp_dir, beta in zip(exp_dirs, beta_values):
        results = load_evaluation_results(exp_dir)
        if results:
            eval_results.append((beta, results))
    
    if eval_results:
        for idx, (beta, results) in enumerate(eval_results):
            mean_accs = results['mean_accuracies']
            std_accs = results['std_accuracies']
            timesteps = np.arange(len(mean_accs))
            
            color = colors[idx % len(colors)]
            
            ax_adapt.plot(timesteps, mean_accs, 'o-', label=f'β={beta}',
                         linewidth=2, markersize=6, color=color)
            ax_adapt.fill_between(timesteps, 
                                 np.array(mean_accs) - np.array(std_accs),
                                 np.array(mean_accs) + np.array(std_accs),
                                 alpha=0.2, color=color)
        
        ax_adapt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax_adapt.set_title('Adaptation Curves (Thompson Sampling)', fontsize=12, fontweight='bold')
        ax_adapt.set_xlabel('Interaction Number (t)', fontsize=10)
        ax_adapt.set_ylabel('Accuracy', fontsize=10)
        ax_adapt.legend(fontsize=9)
        ax_adapt.grid(True, alpha=0.3)
        ax_adapt.set_ylim([0, 1])
        
        # Create comparison table
        table_data = [['Beta', 'Initial', 'Final', 'Improvement', 'Separation']]
        for beta, results in eval_results:
            initial = results['initial_accuracy']
            final = results['final_accuracy']
            improvement = results['improvement']
            
            # Get final separation from checkpoint metrics
            metrics = load_checkpoint_metrics(exp_dirs[beta_values.index(beta)])
            sep = 'N/A'
            if metrics:
                final_ckpt = [c for c in metrics['checkpoints'] if c['step'] == 'final']
                if final_ckpt:
                    sep = f"{final_ckpt[0]['separation_t9']:.3f}"
            
            table_data.append([
                f'{beta}',
                f'{initial:.2%}',
                f'{final:.2%}',
                f'{improvement:.2%}',
                sep
            ])
        
        table = ax_table.table(cellText=table_data, cellLoc='center',
                              loc='center', bbox=[0, 0.3, 1, 0.7])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best results
        if len(table_data) > 1:
            improvements = [float(row[3].strip('%')) for row in table_data[1:]]
            best_idx = np.argmax(improvements) + 1
            for i in range(5):
                table[(best_idx, i)].set_facecolor('#E7F4E4')
    
    # ========================================
    # Row 3: t-SNE Progression Comparison
    # ========================================
    
    # Load checkpoint progression images
    for idx, (exp_dir, beta) in enumerate(zip(exp_dirs, beta_values)):
        progression_img = os.path.join(exp_dir, 'checkpoint_progression.png')
        
        if os.path.exists(progression_img):
            ax_tsne = fig.add_subplot(gs[2, idx])
            
            img = Image.open(progression_img)
            ax_tsne.imshow(img)
            ax_tsne.axis('off')
            ax_tsne.set_title(f'β={beta} t-SNE Progression', fontsize=12, fontweight='bold')
    
    # Overall title
    fig.suptitle('Beta Experiments Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comprehensive comparison: {output_path}")
    plt.close()


def generate_side_by_side_tsne(exp_dirs, beta_values, output_path):
    """Generate side-by-side t-SNE comparison for final checkpoints."""
    
    n_experiments = len(exp_dirs)
    fig, axes = plt.subplots(1, n_experiments, figsize=(8*n_experiments, 6))
    
    if n_experiments == 1:
        axes = [axes]
    
    for idx, (exp_dir, beta) in enumerate(zip(exp_dirs, beta_values)):
        # Load final checkpoint diagnosis image
        final_diagnosis = None
        for pattern in ['latent_diagnosis_final_checkpoint.png', 
                       'latent_diagnosis_hh_final.png',
                       '*final*.png']:
            matches = glob.glob(os.path.join(exp_dir, pattern))
            if matches:
                final_diagnosis = matches[0]
                break
        
        if final_diagnosis and os.path.exists(final_diagnosis):
            img = Image.open(final_diagnosis)
            axes[idx].imshow(img)
            axes[idx].axis('off')
            
            # Add metrics as subtitle
            results = load_evaluation_results(exp_dir)
            if results:
                subtitle = f"β={beta}\n"
                subtitle += f"Initial: {results['initial_accuracy']:.1%} → "
                subtitle += f"Final: {results['final_accuracy']:.1%}\n"
                subtitle += f"Improvement: {results['improvement']:.1%}"
                axes[idx].set_title(subtitle, fontsize=12, fontweight='bold')
        else:
            axes[idx].text(0.5, 0.5, f'β={beta}\n(No visualization found)', 
                          ha='center', va='center', fontsize=14)
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved side-by-side t-SNE comparison: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare beta experiments")
    parser.add_argument("--base_dir", type=str, 
                       default="experiments/streaming_vpl_hh",
                       help="Base directory containing experiments")
    parser.add_argument("--beta_values", type=float, nargs='+',
                       default=[0.001, 0.005],
                       help="Beta values to compare")
    parser.add_argument("--output_dir", type=str,
                       default="experiments",
                       help="Output directory for comparison plots")
    args = parser.parse_args()
    
    print("="*80)
    print("Beta Experiments Comparison")
    print("="*80)
    
    # Find experiment directories
    exp_dirs = []
    for beta in args.beta_values:
        exp_dir = find_experiment_dir(args.base_dir, beta)
        if exp_dir:
            print(f"Found β={beta}: {exp_dir}")
            exp_dirs.append(exp_dir)
        else:
            print(f"WARNING: No experiment found for β={beta}")
    
    if not exp_dirs:
        print("ERROR: No experiments found")
        return
    
    # Filter beta_values to only those with found directories
    beta_values = [args.beta_values[i] for i, exp_dir in enumerate(exp_dirs) if exp_dir]
    
    print(f"\nComparing {len(exp_dirs)} experiments")
    
    # Generate comparison plots
    output_path = os.path.join(args.output_dir, 'beta_comparison_report.png')
    plot_comparison(exp_dirs, beta_values, output_path)
    
    # Generate side-by-side t-SNE
    tsne_output = os.path.join(args.output_dir, 'beta_tsne_comparison.png')
    generate_side_by_side_tsne(exp_dirs, beta_values, tsne_output)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for exp_dir, beta in zip(exp_dirs, beta_values):
        results = load_evaluation_results(exp_dir)
        if results:
            print(f"\nβ={beta}:")
            print(f"  Initial Accuracy: {results['initial_accuracy']:.2%}")
            print(f"  Final Accuracy:   {results['final_accuracy']:.2%}")
            print(f"  Improvement:      {results['improvement']:.2%}")
            
            metrics = load_checkpoint_metrics(exp_dir)
            if metrics:
                final_ckpt = [c for c in metrics['checkpoints'] if c['step'] == 'final']
                if final_ckpt:
                    print(f"  Final Separation: {final_ckpt[0]['separation_t9']:.4f}")
    
    print("\n" + "="*80)
    print(f"Comparison plots saved to {args.output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()

