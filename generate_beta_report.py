"""
Generate Beta Experiment Report

Creates a comprehensive markdown report summarizing the results of beta experiments,
including metrics tables, embedded visualizations, and recommendations.
"""

import json
import os
import glob
import argparse
from datetime import datetime
from pathlib import Path


def find_experiment_dir(base_dir, beta_value):
    """Find experiment directory for a specific beta value."""
    pattern = f"*beta{beta_value}*"
    matches = glob.glob(os.path.join(base_dir, pattern))
    if not matches:
        return None
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


def generate_report(exp_dirs, beta_values, output_path):
    """Generate markdown report."""
    
    report = []
    
    # Header
    report.append("# Beta Experiments Results Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append(f"This report summarizes the results of experiments testing different KL penalty weights (beta) ")
    report.append(f"with free bits thresholding to address the 'empty latent space' problem.")
    report.append("")
    report.append(f"**Experiments Conducted:** {len(exp_dirs)}")
    report.append(f"**Beta Values Tested:** {', '.join(map(str, beta_values))}")
    report.append("")
    
    # Load all results
    all_results = []
    for exp_dir, beta in zip(exp_dirs, beta_values):
        eval_results = load_evaluation_results(exp_dir)
        checkpoint_metrics = load_checkpoint_metrics(exp_dir)
        
        if eval_results and checkpoint_metrics:
            final_ckpt = [c for c in checkpoint_metrics['checkpoints'] if c['step'] == 'final']
            separation = final_ckpt[0]['separation_t9'] if final_ckpt else 0.0
            
            all_results.append({
                'beta': beta,
                'initial_acc': eval_results['initial_accuracy'],
                'final_acc': eval_results['final_accuracy'],
                'improvement': eval_results['improvement'],
                'separation': separation,
                'exp_dir': exp_dir
            })
    
    # Quick comparison table
    report.append("### Quick Comparison")
    report.append("")
    report.append("| Beta | Initial Accuracy | Final Accuracy | Improvement | Latent Separation |")
    report.append("|------|------------------|----------------|-------------|-------------------|")
    
    for result in all_results:
        improvement_str = f"{result['improvement']:.2%}"
        if result['improvement'] > 0:
            improvement_str = f"**+{result['improvement']:.2%}** ✅"
        elif result['improvement'] < -0.05:
            improvement_str = f"**{result['improvement']:.2%}** ⚠️"
        
        separation_str = f"{result['separation']:.4f}"
        if result['separation'] > 0.01:
            separation_str = f"**{result['separation']:.4f}** ✅"
        
        report.append(f"| {result['beta']} | {result['initial_acc']:.2%} | {result['final_acc']:.2%} | {improvement_str} | {separation_str} |")
    
    report.append("")
    
    # Determine best performer
    if all_results:
        best_result = max(all_results, key=lambda x: x['improvement'])
        report.append(f"**Best Performer:** β={best_result['beta']} with {best_result['improvement']:.2%} improvement")
        report.append("")
    
    # Visualizations
    report.append("---")
    report.append("")
    report.append("## Visualizations")
    report.append("")
    
    # Comparison plot
    comparison_plot = "experiments/beta_comparison_report.png"
    if os.path.exists(comparison_plot):
        report.append("### Comprehensive Comparison")
        report.append("")
        report.append(f"![Beta Comparison]({comparison_plot})")
        report.append("")
    
    # t-SNE comparison
    tsne_plot = "experiments/beta_tsne_comparison.png"
    if os.path.exists(tsne_plot):
        report.append("### t-SNE Latent Space Comparison")
        report.append("")
        report.append(f"![t-SNE Comparison]({tsne_plot})")
        report.append("")
    
    # Detailed Results
    report.append("---")
    report.append("")
    report.append("## Detailed Results")
    report.append("")
    
    for result in all_results:
        report.append(f"### Experiment: β={result['beta']}")
        report.append("")
        report.append(f"**Directory:** `{result['exp_dir']}`")
        report.append("")
        
        # Metrics
        report.append("#### Evaluation Metrics")
        report.append("")
        report.append(f"- **Initial Accuracy (t=0):** {result['initial_acc']:.2%}")
        report.append(f"- **Final Accuracy (t=9):** {result['final_acc']:.2%}")
        report.append(f"- **Adaptation Improvement:** {result['improvement']:.2%}")
        report.append(f"- **Latent Separation (Euclidean):** {result['separation']:.4f}")
        report.append("")
        
        # Load checkpoint progression
        checkpoint_metrics = load_checkpoint_metrics(result['exp_dir'])
        if checkpoint_metrics:
            report.append("#### Training Progression")
            report.append("")
            report.append("| Checkpoint | Accuracy (t=0) | Accuracy (t=9) | Improvement | Separation (t=9) |")
            report.append("|------------|----------------|----------------|-------------|------------------|")
            
            for ckpt in checkpoint_metrics['checkpoints']:
                step = ckpt['step']
                step_str = 'Final' if step == 'final' else f"Step {step}"
                improvement = ckpt['acc_t9'] - ckpt['acc_t0']
                report.append(f"| {step_str} | {ckpt['acc_t0']:.2%} | {ckpt['acc_t9']:.2%} | {improvement:.2%} | {ckpt['separation_t9']:.4f} |")
            
            report.append("")
        
        # Checkpoint progression image
        prog_img = os.path.join(result['exp_dir'], 'checkpoint_progression.png')
        if os.path.exists(prog_img):
            report.append("#### Checkpoint Progression")
            report.append("")
            report.append(f"![Checkpoint Progression]({prog_img})")
            report.append("")
        
        # Adaptation curve
        adapt_img = os.path.join(result['exp_dir'], 'evaluation', 'adaptation_curve.png')
        if os.path.exists(adapt_img):
            report.append("#### Adaptation Curve")
            report.append("")
            report.append(f"![Adaptation Curve]({adapt_img})")
            report.append("")
    
    # Analysis & Recommendations
    report.append("---")
    report.append("")
    report.append("## Analysis & Recommendations")
    report.append("")
    
    if all_results:
        # Analyze improvements
        positive_adaptations = [r for r in all_results if r['improvement'] > 0.01]
        negative_adaptations = [r for r in all_results if r['improvement'] < -0.05]
        
        if positive_adaptations:
            report.append("### ✅ Positive Findings")
            report.append("")
            for r in positive_adaptations:
                report.append(f"- **β={r['beta']}** shows positive adaptation (+{r['improvement']:.2%})")
            report.append("")
        
        if negative_adaptations:
            report.append("### ⚠️ Issues Detected")
            report.append("")
            for r in negative_adaptations:
                report.append(f"- **β={r['beta']}** shows negative adaptation ({r['improvement']:.2%})")
            report.append("")
        
        # Check latent separation
        good_separation = [r for r in all_results if r['separation'] > 0.01]
        poor_separation = [r for r in all_results if r['separation'] < 0.005]
        
        if good_separation:
            report.append("### Latent Space Quality")
            report.append("")
            report.append("**Good Separation (>0.01):**")
            report.append("")
            for r in good_separation:
                report.append(f"- β={r['beta']}: {r['separation']:.4f} ✅")
            report.append("")
        
        if poor_separation:
            report.append("**Poor Separation (<0.005):**")
            report.append("")
            for r in poor_separation:
                report.append(f"- β={r['beta']}: {r['separation']:.4f} ⚠️")
            report.append("")
        
        # Recommendations
        report.append("### Recommendations")
        report.append("")
        
        if positive_adaptations and good_separation:
            best = max(positive_adaptations, key=lambda x: x['improvement'])
            report.append(f"1. **Use β={best['beta']}** for future experiments")
            report.append(f"   - Shows positive adaptation (+{best['improvement']:.2%})")
            report.append(f"   - Good latent separation ({best['separation']:.4f})")
            report.append("")
        
        if negative_adaptations:
            report.append("2. **For models showing negative adaptation:**")
            report.append("   - Further reduce beta (try β < 0.001)")
            report.append("   - Increase free bits threshold")
            report.append("   - Consider increasing observation encoder capacity (obs_dim)")
            report.append("")
        
        if poor_separation:
            report.append("3. **For models with poor latent separation:**")
            report.append("   - Reduce KL penalty further")
            report.append("   - Increase model capacity (hidden_dim, latent_dim)")
            report.append("   - Check if observations contain sufficient signal")
            report.append("")
        
        # Compare to baseline
        baseline_improvement = -0.12  # From DIAGNOSIS_REPORT.md
        better_than_baseline = [r for r in all_results if r['improvement'] > baseline_improvement]
        
        if better_than_baseline:
            report.append(f"4. **Comparison to Baseline (β=0.1):**")
            report.append(f"   - Baseline showed {baseline_improvement:.2%} degradation")
            report.append(f"   - {len(better_than_baseline)}/{len(all_results)} experiments improved over baseline ✅")
            report.append("")
    
    # Conclusion
    report.append("---")
    report.append("")
    report.append("## Conclusion")
    report.append("")
    
    if all_results:
        avg_improvement = sum(r['improvement'] for r in all_results) / len(all_results)
        avg_separation = sum(r['separation'] for r in all_results) / len(all_results)
        
        report.append(f"**Average Improvement:** {avg_improvement:.2%}")
        report.append(f"**Average Latent Separation:** {avg_separation:.4f}")
        report.append("")
        
        if avg_improvement > 0 and avg_separation > 0.01:
            report.append("✅ **SUCCESS:** Reducing beta and implementing free bits has successfully addressed ")
            report.append("the negative adaptation problem. The latent space now learns meaningful user representations.")
        elif avg_improvement > -0.05:
            report.append("⚠️ **PARTIAL SUCCESS:** Improvements observed but not yet optimal. ")
            report.append("Further tuning of beta and free bits recommended.")
        else:
            report.append("❌ **NEEDS WORK:** Models still showing negative adaptation. ")
            report.append("Consider more aggressive beta reduction or architectural changes.")
    
    report.append("")
    report.append("---")
    report.append("")
    report.append(f"*Report generated by `generate_beta_report.py`*")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate beta experiment report")
    parser.add_argument("--base_dir", type=str,
                       default="experiments/streaming_vpl_hh",
                       help="Base directory containing experiments")
    parser.add_argument("--beta_values", type=float, nargs='+',
                       default=[0.001, 0.005],
                       help="Beta values to include in report")
    parser.add_argument("--output", type=str,
                       default="BETA_EXPERIMENT_RESULTS.md",
                       help="Output markdown file path")
    args = parser.parse_args()
    
    print("="*80)
    print("Generating Beta Experiments Report")
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
    
    # Generate report
    generate_report(exp_dirs, beta_values, args.output)
    
    print("\n" + "="*80)
    print(f"Report generated: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()

