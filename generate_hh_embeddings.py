"""
Generate embeddings for HH-RLHF data for Streaming VPL

This script adds pre-computed embeddings to the relabeled HH-RLHF dataset
to enable sequential preference learning with helpful/harmless user types.
"""

import os
import json
import gzip
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import torch
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


@dataclass
class ScriptArguments:
    """Arguments for generating embeddings."""
    
    input_dir: str = field(
        default="data/relabeled_hh_rlhf",
        metadata={"help": "Directory containing the relabeled HH-RLHF data"}
    )
    output_dir: str = field(
        default="data_release/hh_rlhf/gpt2",
        metadata={"help": "Directory to save data with embeddings"}
    )
    data_subset: str = field(
        default="both",
        metadata={"help": "Which subset to process: 'helpful', 'harmless', or 'both'"}
    )
    embed_dim: int = field(
        default=768,
        metadata={"help": "Dimension of embeddings (768 for GPT-2)"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for embedding generation"}
    )


def generate_embeddings(args):
    """Generate embeddings for HH-RLHF data."""
    
    print("="*80)
    print("Generating Embeddings for HH-RLHF Data")
    print("="*80)
    print(f"Input: {args.input_dir}/{args.data_subset}")
    print(f"Output: {args.output_dir}/{args.data_subset}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Max length: {args.max_length}")
    print("="*80 + "\n")
    
    # Load GPT-2 model
    print("Loading GPT-2 model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use bfloat16 on GPU, float32 on CPU (for compatibility)
    if device == "cuda":
        model = AutoModelForSequenceClassification.from_pretrained(
            "gpt2",
            num_labels=args.embed_dim,
            torch_dtype=torch.bfloat16
        )
    else:
        print("Using CPU - embeddings will be slower. Consider using GPU for faster generation.")
        model = AutoModelForSequenceClassification.from_pretrained(
            "gpt2",
            num_labels=args.embed_dim
        )
    
    model.score.weight.data *= 0.01  # Initialize final layer with small weights
    model.to(device)
    model.eval()
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Model loaded on {device}\n")
    
    # Process each split
    for split in ['train', 'test']:
        input_path = os.path.join(args.input_dir, args.data_subset, f"{split}.jsonl.gz")
        output_dir = os.path.join(args.output_dir, args.data_subset)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{split}.jsonl")
        
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping...")
            continue
        
        print(f"Processing {split} split...")
        
        # Load data
        data = []
        with gzip.open(input_path, 'rt') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        print(f"  Loaded {len(data)} samples")
        
        # Generate embeddings
        output_data = []
        for item in tqdm(data, desc=f"  Generating embeddings"):
            chosen_text = item['chosen']
            rejected_text = item['rejected']
            
            # Tokenize
            chosen_tokens = tokenizer(
                chosen_text,
                max_length=args.max_length,
                truncation=True,
                return_tensors="pt"
            )
            rejected_tokens = tokenizer(
                rejected_text,
                max_length=args.max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # Generate embeddings
            with torch.no_grad():
                chosen_emb = model(
                    input_ids=chosen_tokens['input_ids'].to(device),
                    attention_mask=chosen_tokens['attention_mask'].to(device)
                ).logits[0].float().cpu().numpy().tolist()
                
                rejected_emb = model(
                    input_ids=rejected_tokens['input_ids'].to(device),
                    attention_mask=rejected_tokens['attention_mask'].to(device)
                ).logits[0].float().cpu().numpy().tolist()
            
            # Add embeddings to item
            item['embeddings'] = {
                'embedding_chosen': chosen_emb,
                'embedding_rejected': rejected_emb
            }
            
            output_data.append(item)
        
        # Save with embeddings
        print(f"  Saving to {output_path}...")
        with open(output_path, 'w') as f:
            for item in output_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"  ✓ Saved {len(output_data)} samples\n")
    
    print("="*80)
    print("Embedding generation complete!")
    print("="*80)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    generate_embeddings(args)

