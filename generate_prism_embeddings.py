"""
Generate embeddings for PRISM alignment dataset.

This script:
1. Downloads PRISM conversations from HuggingFace
2. Extracts preference pairs from multi-turn conversations
3. Generates embeddings using GPT-2 or Llama
4. Saves processed data in JSONL format for training
"""

import os
import json
import gzip
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset


def parse_conversation_to_preferences(conversation_data: Dict) -> List[Dict]:
    """
    Parse a PRISM conversation into preference pairs.
    
    In PRISM, each turn has multiple model responses that are rated.
    We extract pairs where one response is clearly preferred over another.
    
    Based on actual PRISM structure:
    - conversation_history has entries with: 'turn', 'role', 'content', 'score', 'within_turn_id'
    - Multiple model responses can exist at the same turn (different within_turn_id)
    - User prompts alternate with model responses
    
    Args:
        conversation_data: Single conversation from PRISM dataset
        
    Returns:
        List of preference pairs with metadata
    """
    pairs = []
    
    conversation_id = conversation_data.get('conversation_id', '')
    user_id = conversation_data.get('user_id', '')
    conversation_type = conversation_data.get('conversation_type', '')
    conversation_history = conversation_data.get('conversation_history', [])
    
    # Group entries by turn number
    turns = {}
    for entry in conversation_history:
        turn = entry.get('turn', 0)
        if turn not in turns:
            turns[turn] = {'user': None, 'models': []}
        
        role = entry.get('role', '')
        if role == 'user':
            turns[turn]['user'] = entry.get('content', '')
        elif role == 'model':
            score = entry.get('score')
            if score is not None and entry.get('content'):  # Only include rated responses with content
                turns[turn]['models'].append({
                    'content': entry.get('content', ''),
                    'score': score,
                    'model_name': entry.get('model_name', 'unknown'),
                    'model_provider': entry.get('model_provider', 'unknown')
                })
    
    # Create preference pairs from each turn
    for turn_num, turn_data in turns.items():
        prompt = turn_data['user']
        models = turn_data['models']
        
        if prompt is None or len(models) < 2:
            continue  # Need prompt and at least 2 model responses to create a pair
        
        # Sort by score (descending)
        models_sorted = sorted(models, key=lambda x: x['score'], reverse=True)
        
        # Create pairs: best vs each worse option
        best = models_sorted[0]
        for worse in models_sorted[1:]:
            rating_diff = best['score'] - worse['score']
            
            # Only include pairs with sufficient rating difference
            if rating_diff >= 10.0:  # Will be configurable via min_rating_diff
                pairs.append({
                    'conversation_id': conversation_id,
                    'user_id': user_id,
                    'conversation_type': conversation_type,
                    'turn_number': turn_num,
                    'prompt': prompt,
                    'chosen': best['content'],
                    'rejected': worse['content'],
                    'rating_chosen': best['score'],
                    'rating_rejected': worse['score'],
                    'rating_difference': rating_diff,
                    'model_chosen': best['model_name'],
                    'model_rejected': worse['model_name']
                })
    
    return pairs


def generate_embeddings(
    texts: List[str],
    model,
    tokenizer,
    max_length: int = 1024,
    batch_size: int = 8,
    device: str = "cuda"
) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            last_hidden_state = outputs.hidden_states[-1]
            
            # Get last token embedding for each sequence
            batch_embeddings = []
            for j, seq_len in enumerate(tokens['attention_mask'].sum(dim=1)):
                # Use last non-padding token
                embedding = last_hidden_state[j, seq_len - 1].float().cpu().numpy()
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)


def process_prism_dataset(
    output_dir: str,
    model_type: str = "gpt2",
    embed_dim: int = 1024,
    max_length: int = 1024,
    train_test_split: float = 0.8,
    min_rating_diff: float = 10.0,
    seed: int = 0
):
    """
    Main processing function.
    
    Args:
        output_dir: Where to save processed data
        model_type: "gpt2" or "llama"
        embed_dim: Embedding dimension
        max_length: Max token length
        train_test_split: Fraction for train split
        min_rating_diff: Minimum rating difference to include pair
        seed: Random seed
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output directory
    os.makedirs(f"{output_dir}/{model_type}", exist_ok=True)
    
    # Load PRISM dataset from HuggingFace
    print("Downloading PRISM dataset from HuggingFace...")
    # The PRISM dataset has multiple configs: 'conversations', 'survey', 'utterances', 'metadata'
    # We want 'conversations' which has the full conversation trees
    conversations_data = load_dataset(
        "HannahRoseKirk/prism-alignment",
        "conversations",
        split="train"
    )
    
    print(f"Loaded {len(conversations_data)} conversations")
    
    # Extract preference pairs from all conversations
    print("Extracting preference pairs from conversations...")
    all_pairs = []
    
    for conv in tqdm(conversations_data):
        pairs = parse_conversation_to_preferences(conv)
        all_pairs.extend(pairs)
    
    print(f"Extracted {len(all_pairs)} preference pairs")
    
    # Filter by rating difference
    all_pairs = [p for p in all_pairs if p['rating_difference'] >= min_rating_diff]
    print(f"After filtering (min diff={min_rating_diff}): {len(all_pairs)} pairs")
    
    # Split into train/test
    np.random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * train_test_split)
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]
    
    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")
    
    # Load embedding model
    print(f"Loading {model_type} model for embeddings...")
    
    if model_type == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForSequenceClassification.from_pretrained(
            "gpt2",
            num_labels=embed_dim,
            torch_dtype=torch.bfloat16
        )
        model.score.weight.data *= 0.01
    elif model_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            add_eos_token=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.bfloat16
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Process train and test splits
    for split_name, pairs in [("train", train_pairs), ("test", test_pairs)]:
        print(f"\nProcessing {split_name} split...")
        
        # Prepare texts for embedding
        prompts = [f"{p['prompt']}\n\nAssistant: {p['chosen']}" for p in pairs]
        chosen_texts = [f"{p['prompt']}\n\nAssistant: {p['chosen']}" for p in pairs]
        rejected_texts = [f"{p['prompt']}\n\nAssistant: {p['rejected']}" for p in pairs]
        
        # Generate embeddings
        print("Generating embeddings for chosen responses...")
        chosen_embeddings = generate_embeddings(
            chosen_texts, model, tokenizer, max_length, device=device
        )
        
        print("Generating embeddings for rejected responses...")
        rejected_embeddings = generate_embeddings(
            rejected_texts, model, tokenizer, max_length, device=device
        )
        
        # Create final dataset with embeddings
        output_data = []
        for i, pair in enumerate(tqdm(pairs, desc="Creating final dataset")):
            output_data.append({
                'conversation_id': pair['conversation_id'],
                'user_id': pair['user_id'],
                'conversation_type': pair['conversation_type'],
                'turn_number': pair['turn_number'],
                'prompt': pair['prompt'],
                'chosen': pair['chosen'],
                'rejected': pair['rejected'],
                'rating_chosen': pair['rating_chosen'],
                'rating_rejected': pair['rating_rejected'],
                'rating_difference': pair['rating_difference'],
                'model_chosen': pair['model_chosen'],
                'model_rejected': pair['model_rejected'],
                'embeddings': {
                    'embedding_chosen': chosen_embeddings[i].tolist(),
                    'embedding_rejected': rejected_embeddings[i].tolist()
                }
            })
        
        # Save to JSONL
        output_file = f"{output_dir}/{model_type}/{split_name}.jsonl"
        print(f"Saving to {output_file}...")
        
        with open(output_file, 'w') as f:
            for item in output_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Saved {len(output_data)} items to {output_file}")
    
    # Print statistics
    print("\n" + "="*80)
    print("PRISM Dataset Processing Complete!")
    print("="*80)
    print(f"Output directory: {output_dir}/{model_type}/")
    print(f"Train samples: {len(train_pairs)}")
    print(f"Test samples: {len(test_pairs)}")
    print(f"Unique users: {len(set(p['user_id'] for p in all_pairs))}")
    print(f"Embedding dimension: {embed_dim}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PRISM dataset for VPL training")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_release/prism",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2",
        choices=["gpt2", "llama"],
        help="Model type for embeddings"
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=1024,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum token length"
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.8,
        help="Fraction of data for training"
    )
    parser.add_argument(
        "--min_rating_diff",
        type=float,
        default=10.0,
        help="Minimum rating difference to include preference pair"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    process_prism_dataset(
        output_dir=args.output_dir,
        model_type=args.model_type,
        embed_dim=args.embed_dim,
        max_length=args.max_length,
        train_test_split=args.train_test_split,
        min_rating_diff=args.min_rating_diff,
        seed=args.seed
    )

