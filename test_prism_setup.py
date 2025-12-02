"""
Test script to verify PRISM dataset can be loaded and processed correctly.

This script tests:
1. Loading PRISM dataset from HuggingFace
2. Parsing conversations to preference pairs
3. Basic statistics about the data
"""

import sys
from datasets import load_dataset
from collections import defaultdict
import numpy as np

def test_prism_loading():
    """Test basic loading of PRISM dataset."""
    print("="*80)
    print("PRISM Dataset Test")
    print("="*80)
    print()
    
    print("Step 1: Loading PRISM dataset from HuggingFace...")
    try:
        # Load the conversations subset (need to specify config)
        dataset = load_dataset("HannahRoseKirk/prism-alignment", "conversations", split="train")
        print(f"✓ Successfully loaded dataset")
        print(f"  Total conversations: {len(dataset)}")
        print()
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print("\nMake sure you have internet connection and the datasets library installed:")
        print("  pip install datasets")
        return False
    
    print("Step 2: Analyzing dataset structure...")
    sample = dataset[0]
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  User ID: {sample.get('user_id', 'N/A')}")
    print(f"  Conversation ID: {sample.get('conversation_id', 'N/A')}")
    print(f"  Conversation Type: {sample.get('conversation_type', 'N/A')}")
    print(f"  Conversation Turns: {sample.get('conversation_turns', 'N/A')}")
    print()
    
    print("Step 3: Analyzing conversation history structure...")
    conv_history = sample.get('conversation_history', [])
    print(f"  History entries: {len(conv_history)}")
    if len(conv_history) > 0:
        print(f"  First entry type: {type(conv_history[0])}")
        if isinstance(conv_history[0], dict):
            print(f"  First entry keys: {list(conv_history[0].keys())}")
            print(f"  First entry role: {conv_history[0].get('role', 'N/A')}")
    print()
    
    print("Step 4: Computing dataset statistics...")
    
    # User statistics
    user_ids = [item['user_id'] for item in dataset]
    unique_users = len(set(user_ids))
    
    # Conversation statistics
    turn_counts = [item['conversation_turns'] for item in dataset]
    conv_types = [item['conversation_type'] for item in dataset]
    
    print(f"  Unique users: {unique_users}")
    print(f"  Total conversations: {len(dataset)}")
    print(f"  Avg conversations per user: {len(dataset) / unique_users:.1f}")
    print()
    
    print(f"  Turn count statistics:")
    print(f"    Min: {min(turn_counts)}")
    print(f"    Max: {max(turn_counts)}")
    print(f"    Mean: {np.mean(turn_counts):.1f}")
    print(f"    Median: {np.median(turn_counts):.1f}")
    print()
    
    print(f"  Conversation types:")
    type_counts = defaultdict(int)
    for t in conv_types:
        type_counts[t] += 1
    for conv_type, count in sorted(type_counts.items()):
        print(f"    {conv_type}: {count} ({100*count/len(conv_types):.1f}%)")
    print()
    
    print("Step 5: Analyzing preference ratings...")
    
    # Count how many model responses have ratings
    total_responses = 0
    rated_responses = 0
    ratings = []
    
    sample_size = min(100, len(dataset))
    for i in range(sample_size):  # Sample first 100 conversations
        item = dataset[i]
        conv_history = item.get('conversation_history', [])
        for entry in conv_history:
            if entry.get('role') == 'model':
                total_responses += 1
                score = entry.get('score')
                if score is not None:
                    rated_responses += 1
                    ratings.append(score)
    
    print(f"  Sample of first {sample_size} conversations:")
    print(f"    Total model responses: {total_responses}")
    print(f"    Rated responses: {rated_responses}")
    if len(ratings) > 0:
        print(f"    Rating statistics:")
        print(f"      Min: {min(ratings)}")
        print(f"      Max: {max(ratings)}")
        print(f"      Mean: {np.mean(ratings):.1f}")
        print(f"      Median: {np.median(ratings):.1f}")
    print()
    
    print("Step 6: Estimating preference pairs...")
    
    # Count potential preference pairs
    total_pairs = 0
    turns_with_multiple_responses = 0
    
    for i in range(sample_size):
        item = dataset[i]
        conv_history = item.get('conversation_history', [])
        
        # Group by turns (consecutive user-model pairs)
        turn_responses = defaultdict(list)
        current_prompt = None
        turn_num = 0
        
        for entry in conv_history:
            if entry.get('role') == 'user':
                current_prompt = entry.get('content')
                turn_num += 1
            elif entry.get('role') == 'model' and current_prompt:
                score = entry.get('score')
                if score is not None:
                    turn_responses[turn_num].append(score)
        
        # Count pairs from each turn
        for turn, scores in turn_responses.items():
            if len(scores) >= 2:
                turns_with_multiple_responses += 1
                # Number of pairs is n*(n-1)/2 for n responses
                n = len(scores)
                total_pairs += n * (n - 1) // 2
    
    print(f"  From first {sample_size} conversations:")
    print(f"    Turns with multiple responses: {turns_with_multiple_responses}")
    print(f"    Potential preference pairs: {total_pairs}")
    print(f"    Estimated pairs for full dataset: ~{total_pairs * len(dataset) // sample_size}")
    print()
    
    print("="*80)
    print("✓ PRISM Dataset Test Passed!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Run preprocessing: bash generate_prism_embeddings.sh")
    print("  2. Train model: bash run_streaming_vpl_prism.sh")
    print()
    
    return True


if __name__ == "__main__":
    success = test_prism_loading()
    sys.exit(0 if success else 1)

