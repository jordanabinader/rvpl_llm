"""
Training script for Streaming VPL with Recurrent Belief Updates

This script trains a RecurrentVAEModel on the HH-RLHF (or Pets) dataset
using the new HyperDecoder architecture, Cyclical Annealing, and Temporal Weighting
to prevent posterior collapse.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import random
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    TrainerCallback,
)

# Import our custom modules
from .data_utils.sequential_pets_dataset import SequentialPetsDataset, sequential_collate_fn
from .data_utils.sequential_hh_dataset import SequentialHHDataset, sequential_hh_collate_fn
from .recurrent_vae_utils import RecurrentVAEModel, RecurrentVAETrainer


@dataclass
class ScriptArguments:
    """Arguments for training Streaming VPL."""
    
    # Data arguments
    data_path: str = field(
        default="data_release/hh_rlhf/gpt2",
        metadata={"help": "Path to the dataset directory"}
    )
    data_subset: str = field(
        default="both",
        metadata={"help": "Which subset to use: 'harmless', 'helpful', or 'both'"}
    )
    
    # Model arguments
    encoder_embed_dim: int = field(
        default=768,
        metadata={"help": "Dimension of embeddings for encoder (GPT-2 default)"}
    )
    decoder_embed_dim: int = field(
        default=768,
        metadata={"help": "Dimension of embeddings for decoder (GPT-2 default)"}
    )
    latent_dim: int = field(
        default=64,  # Changed from 512 (Fix #4: Inverted Funnel)
        metadata={"help": "Dimension of latent user preference vector"}
    )
    hidden_dim: int = field(
        default=256,  # Changed from 512 (Fix #4: Inverted Funnel)
        metadata={"help": "Dimension of hidden layers in encoder/decoder"}
    )
    
    # Sequential processing arguments
    seq_length: int = field(
        default=10,
        metadata={"help": "Number of interactions per episode (T)"}
    )
    epoch_size: int = field(
        default=1000,
        metadata={"help": "Number of episodes per epoch"}
    )
    
    # --- UPDATED: New Regularization Arguments ---
    beta_max: float = field(
        default=0.1,
        metadata={"help": "Maximum KL weight (peak of the cycle)"}
    )
    beta_cycles: int = field(
        default=4,
        metadata={"help": "Number of annealing cycles during training"}
    )
    temporal_gamma: float = field(
        default=1.1,
        metadata={"help": "Weight multiplier for subsequent timesteps (1.0 = equal weighting)"}
    )
    # ---------------------------------------------
    
    # Training arguments
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.001)
    num_train_epochs: int = field(default=20)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    eval_steps: int = field(
        default=100,
        metadata={"help": "Evaluate every N steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every N steps"}
    )
    
    # Misc arguments
    log_dir: str = field(
        default="experiments/streaming_vpl_hh",
        metadata={"help": "Directory to save logs and checkpoints"}
    )
    seed: int = field(default=0)
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 precision"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Use float16 precision"}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing"}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "Local rank for distributed training"}
    )
    resume_from_checkpoint: bool = field(
        default=False,
        metadata={"help": "Resume training from checkpoint"}
    )


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    
    # Set random seed
    set_seed(script_args.seed)
    
    # Set default dtype
    if script_args.bf16:
        torch.set_default_dtype(torch.bfloat16)
    elif script_args.fp16:
        torch.set_default_dtype(torch.float16)
    
    print("="*80)
    print("Streaming VPL Training (Refactored for Posterior Collapse)")
    print("="*80)
    print(f"Data: {script_args.data_path}/{script_args.data_subset}")
    print(f"Seq Length: {script_args.seq_length}")
    print(f"Latent Dim: {script_args.latent_dim}")
    print(f"Cyclical Annealing: Max Beta {script_args.beta_max}, Cycles {script_args.beta_cycles}")
    print(f"Temporal Weighting: Gamma {script_args.temporal_gamma}")
    print(f"Seed: {script_args.seed}")
    print("="*80)
    
    # Load datasets
    print("\nLoading datasets...")
    
    # Detect dataset type based on path
    if "hh" in script_args.data_path.lower():
        print("Using HH-RLHF dataset (helpful/harmless preferences)")
        train_dataset = SequentialHHDataset(
            data_path=script_args.data_path,
            data_subset=script_args.data_subset,
            split="train",
            seq_length=script_args.seq_length,
            epoch_size=script_args.epoch_size,
            seed=script_args.seed
        )
        
        eval_dataset = SequentialHHDataset(
            data_path=script_args.data_path,
            data_subset=script_args.data_subset,
            split="test",
            seq_length=script_args.seq_length,
            epoch_size=script_args.epoch_size // 5,  # Smaller eval set
            seed=script_args.seed + 1
        )
        collate_fn = sequential_hh_collate_fn
    else:
        print("Using Pets dataset (dog/cat preferences)")
        train_dataset = SequentialPetsDataset(
            data_path=script_args.data_path,
            data_subset=script_args.data_subset,
            split="train",
            seq_length=script_args.seq_length,
            epoch_size=script_args.epoch_size,
            seed=script_args.seed
        )
        
        eval_dataset = SequentialPetsDataset(
            data_path=script_args.data_path,
            data_subset=script_args.data_subset,
            split="test",
            seq_length=script_args.seq_length,
            epoch_size=script_args.epoch_size // 5,  # Smaller eval set
            seed=script_args.seed + 1
        )
        collate_fn = sequential_collate_fn
    
    print(f"Train dataset: {len(train_dataset)} episodes")
    print(f"Eval dataset: {len(eval_dataset)} episodes")
    
    # Create model
    print("\nInitializing model...")
    model = RecurrentVAEModel(
        encoder_embed_dim=script_args.encoder_embed_dim,
        decoder_embed_dim=script_args.decoder_embed_dim,
        hidden_dim=script_args.hidden_dim,
        latent_dim=script_args.latent_dim
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define training arguments
    output_dir = os.path.join(
        script_args.log_dir,
        f"{script_args.data_subset}_seq{script_args.seq_length}_"
        f"latent{script_args.latent_dim}_beta{script_args.beta_max}_"
        f"cycles{script_args.beta_cycles}_gamma{script_args.temporal_gamma}_"
        f"seed{script_args.seed}"
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        eval_strategy="steps",
        eval_steps=script_args.eval_steps,
        save_strategy="steps",
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        logging_strategy="steps",
        logging_steps=10,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="wandb",
        run_name=os.path.basename(output_dir),
        load_best_model_at_end=False,
    )
    
    print(f"\nOutput directory: {output_dir}")
    
    # Create trainer
    # --- UPDATED INITIALIZATION ---
    trainer = RecurrentVAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=RecurrentVAETrainer.compute_metrics,
        seq_length=script_args.seq_length,
        latent_dim=script_args.latent_dim,
        # New arguments:
        beta_max=script_args.beta_max,
        beta_cycles=script_args.beta_cycles,
        temporal_gamma=script_args.temporal_gamma
    )
    
    # Add callback for first-step evaluation
    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == 0:
                control.should_evaluate = True
    
    trainer.add_callback(EvaluateFirstStepCallback())
    
    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    trainer.train(script_args.resume_from_checkpoint)
    
    # Save final model
    print("\n" + "="*80)
    print("Saving final model...")
    print("="*80)
    
    final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    
    model_path = os.path.join(final_checkpoint_dir, "model.pt")
    model.save_model(model_path)
    
    print(f"Model saved to: {model_path}")
    print("\nTraining complete!")