"""
Sequential HH-RLHF Dataset for Streaming VPL

This dataset samples episodes (sequences of T interactions) from helpful/harmless user pools.
Unlike the Pets dataset, this uses actual language preferences that should require
multiple observations to identify, making it more suitable for demonstrating adaptation.
"""

import json
import random
from typing import Dict, List, Literal, Optional
import torch
from torch.utils.data import Dataset
import numpy as np


class SequentialHHDataset(Dataset):
    """
    Dataset that samples sequential episodes for Streaming VPL on HH-RLHF data.
    
    Each episode contains T interactions from the same user type:
    - User Type 0: Prefers HELPFUL responses
    - User Type 1: Prefers HARMLESS responses
    
    Uses pre-computed embeddings from the JSONL files.
    """
    
    def __init__(
        self,
        data_path: str,
        data_subset: Literal["helpful", "harmless", "both"],
        split: Literal["train", "test"],
        seq_length: int = 10,
        epoch_size: int = 1000,
        seed: int = 0,
        min_pool_size: Optional[int] = None
    ):
        """
        Args:
            data_path: Path to data directory (e.g., "data_release/hh_rlhf/gpt2")
            data_subset: Which subset to use ("helpful", "harmless", or "both")
            split: "train" or "test"
            seq_length: Number of interactions per episode (T)
            epoch_size: Arbitrary epoch length (number of episodes per epoch)
            seed: Random seed for reproducibility
            min_pool_size: Minimum samples required per user type (defaults to seq_length)
        """
        self.data_path = data_path
        self.data_subset = data_subset
        self.split = split
        self.seq_length = seq_length
        self.epoch_size = epoch_size
        self.seed = seed
        self.min_pool_size = min_pool_size or seq_length
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Load data and cluster by preference type
        self._load_and_cluster_data()
        
        # Determine which pools have enough samples
        self.available_user_types = []
        for user_type in [0, 1]:
            if len(self.pools[user_type]) >= self.min_pool_size:
                self.available_user_types.append(user_type)
        
        if len(self.available_user_types) == 0:
            raise ValueError(
                f"No pools have enough samples (need at least {self.min_pool_size}). "
                f"Pool 0 (Helpful): {len(self.pools[0])}, Pool 1 (Harmless): {len(self.pools[1])}"
            )
        
        # Check if we have both user types
        if len(self.available_user_types) < 2:
            print("\n" + "="*80)
            print("WARNING: Dataset contains only ONE user type!")
            print(f"Helpful preferences (Pool 0): {len(self.pools[0])}")
            print(f"Harmless preferences (Pool 1): {len(self.pools[1])}")
            print("\nThis will result in high accuracy at t=0 (no adaptation to demonstrate).")
            print("For proper adaptation experiments, use data_subset='both' to load both types.")
            print("="*80 + "\n")
        
        print(f"Loaded {len(self.pools[0])} Helpful and {len(self.pools[1])} Harmless samples")
        print(f"Available user types: {self.available_user_types}")
        print(f"Split: {self.split}, Seq Length: {seq_length}, Epoch Size: {epoch_size}")
    
    def _load_and_cluster_data(self):
        """Load JSONL data and cluster into user pools by preference type."""
        # Load all data from specified subset(s)
        all_data = []
        file_path = f"{self.data_path}/{self.data_subset}/{self.split}.jsonl"
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    all_data.append(item)
        except FileNotFoundError:
            print(f"Error: {file_path} not found!")
            print(f"Make sure you've run generate_hh_embeddings.sh first.")
            raise
        
        print(f"Loaded {len(all_data)} total samples")
        
        # Cluster into pools by objective
        self.pools = {
            0: [],  # Helpful preference users
            1: []   # Harmless preference users
        }
        
        for item in all_data:
            # Check if embeddings exist
            if 'embeddings' not in item:
                continue
            
            user_type = self._identify_user_type(item)
            if user_type is not None:
                self.pools[user_type].append(item)
        
        # Debug: print sample from each pool if available
        if len(self.pools[0]) > 0:
            print(f"Helpful sample: {self.pools[0][0]['chosen'][:100]}...")
        if len(self.pools[1]) > 0:
            print(f"Harmless sample: {self.pools[1][0]['chosen'][:100]}...")
    
    def _identify_user_type(self, item: Dict) -> Optional[int]:
        """
        Identify user type based on the objective field.
        
        For HH-RLHF data:
        - objective='helpful' → User Type 0 (prefers helpful responses)
        - objective='harmless' → User Type 1 (prefers harmless responses)
        
        Returns:
            0 if Helpful preference, 1 if Harmless preference, None if unclear
        """
        objective = item.get('objective', '').lower()
        
        if 'helpful' in objective:
            return 0
        elif 'harmless' in objective or 'harmless' in objective:
            return 1
        else:
            return None
    
    def __len__(self) -> int:
        """Return arbitrary epoch size."""
        return self.epoch_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Sample one episode (sequence of T interactions from same user).
        
        Returns:
            Dictionary with:
                - embeddings_chosen: [T, embed_dim]
                - embeddings_rejected: [T, embed_dim]
                - labels: [T] (all 1s since chosen > rejected by definition)
                - user_type: scalar (0 or 1)
        """
        # Randomly select user type from available types
        user_type = random.choice(self.available_user_types)
        
        # Sample T unique items from this user's pool without replacement
        pool = self.pools[user_type]
        episode_items = random.sample(pool, self.seq_length)
        
        # Extract embeddings and stack
        embeddings_chosen = []
        embeddings_rejected = []
        labels = []
        
        for item in episode_items:
            # Get pre-computed embeddings
            embed_chosen = item['embeddings']['embedding_chosen']
            embed_rejected = item['embeddings']['embedding_rejected']
            
            embeddings_chosen.append(embed_chosen)
            embeddings_rejected.append(embed_rejected)
            labels.append(1)  # Chosen is always preferred
        
        return {
            'embeddings_chosen': embeddings_chosen,  # List of lists
            'embeddings_rejected': embeddings_rejected,
            'labels': labels,
            'user_type': user_type
        }


def sequential_hh_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function to batch multiple episodes.
    
    Args:
        batch: List of episodes from __getitem__
        
    Returns:
        Dictionary with batched tensors:
            - embeddings_chosen: [batch_size, seq_len, embed_dim]
            - embeddings_rejected: [batch_size, seq_len, embed_dim]
            - labels: [batch_size, seq_len]
            - user_type: [batch_size]
    """
    batch_size = len(batch)
    seq_len = len(batch[0]['embeddings_chosen'])
    embed_dim = len(batch[0]['embeddings_chosen'][0])
    
    # Initialize tensors
    embeddings_chosen = torch.zeros(batch_size, seq_len, embed_dim)
    embeddings_rejected = torch.zeros(batch_size, seq_len, embed_dim)
    labels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    user_types = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    for i, episode in enumerate(batch):
        embeddings_chosen[i] = torch.tensor(episode['embeddings_chosen'])
        embeddings_rejected[i] = torch.tensor(episode['embeddings_rejected'])
        labels[i] = torch.tensor(episode['labels'])
        user_types[i] = episode['user_type']
    
    return {
        'embeddings_chosen': embeddings_chosen,
        'embeddings_rejected': embeddings_rejected,
        'labels': labels,
        'user_type': user_types
    }

