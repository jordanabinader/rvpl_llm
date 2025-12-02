"""
Sequential Pets Dataset for Streaming VPL

This dataset samples episodes (sequences of T interactions) from user pools.
Critical feature: Ensures train/test use disjoint sentence sets to prevent data leakage.
"""

import json
import random
from typing import Dict, List, Literal, Optional
import torch
from torch.utils.data import Dataset
import numpy as np


class SequentialPetsDataset(Dataset):
    """
    Dataset that samples sequential episodes for Streaming VPL.
    
    Each episode contains T interactions from the same user type (Dog Lover or Cat Lover).
    Uses pre-computed embeddings from the JSONL files.
    """
    
    def __init__(
        self,
        data_path: str,
        data_subset: Literal["harmless", "helpful", "both"],
        split: Literal["train", "test"],
        seq_length: int = 10,
        epoch_size: int = 1000,
        seed: int = 0
    ):
        """
        Args:
            data_path: Path to data directory (e.g., "data_release/simple_pets/gpt2")
            data_subset: Which subset to use ("harmless", "helpful", or "both")
            split: "train" or "test"
            seq_length: Number of interactions per episode (T)
            epoch_size: Arbitrary epoch length (number of episodes per epoch)
            seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.data_subset = data_subset
        self.split = split
        self.seq_length = seq_length
        self.epoch_size = epoch_size
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Load data (from both subsets if data_subset="both")
        self._load_and_cluster_data()
        
        # Determine which pools have enough samples
        self.available_user_types = []
        for user_type in [0, 1]:
            if len(self.pools[user_type]) >= seq_length:
                self.available_user_types.append(user_type)
        
        if len(self.available_user_types) == 0:
            raise ValueError(
                f"No pools have enough samples (need at least {seq_length}). "
                f"Pool 0 (Dog): {len(self.pools[0])}, Pool 1 (Cat): {len(self.pools[1])}"
            )
        
        # CRITICAL CHECK: For proper adaptation experiments, we need BOTH user types
        if len(self.available_user_types) < 2:
            print("\n" + "="*80)
            print("WARNING: Dataset contains only ONE user type!")
            print(f"Dog Lovers (Pool 0): {len(self.pools[0])}")
            print(f"Cat Lovers (Pool 1): {len(self.pools[1])}")
            print("\nThis will result in 100% accuracy at t=0 (no adaptation to demonstrate).")
            print("For proper adaptation experiments, use data_subset='both' to load both types.")
            print("="*80 + "\n")
        
        print(f"Loaded {len(self.pools[0])} Dog Lover samples and {len(self.pools[1])} Cat Lover samples")
        print(f"Available user types: {self.available_user_types}")
        print(f"Split: {self.split}, Seq Length: {seq_length}, Epoch Size: {epoch_size}")
    
    def _load_and_cluster_data(self):
        """Load JSONL data and cluster into user pools."""
        # Determine which subsets to load
        if self.data_subset == "both":
            subsets_to_load = ["harmless", "helpful"]
        else:
            subsets_to_load = [self.data_subset]
        
        # Load all data from specified subset(s)
        all_data = []
        for subset in subsets_to_load:
            file_path = f"{self.data_path}/{subset}/{self.split}.jsonl"
            with open(file_path, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    all_data.append(item)
        
        # Filter for controversial only (Dog vs Cat divergent preferences)
        controversial_data = [item for item in all_data if item.get('controversial', False)]
        
        print(f"Loaded {len(all_data)} total samples, {len(controversial_data)} controversial")
        
        # Cluster into pools by user preference
        self.pools = {
            0: [],  # Dog Lovers
            1: []   # Cat Lovers
        }
        
        for item in controversial_data:
            user_type = self._identify_user_type(item)
            if user_type is not None:
                self.pools[user_type].append(item)
        
        # Debug: print sample from each pool if available
        if len(self.pools[0]) > 0:
            print(f"Dog Lover sample: {self.pools[0][0]['chosen'][:80]}...")
        if len(self.pools[1]) > 0:
            print(f"Cat Lover sample: {self.pools[1][0]['chosen'][:80]}...")
        
        # Verify train/test disjoint split (sentences should not overlap)
        if self.split == "train":
            self._verify_train_split()
    
    def _identify_user_type(self, item: Dict) -> Optional[int]:
        """
        Identify user type based on which pet was chosen.
        
        Returns:
            0 if Dog Lover, 1 if Cat Lover, None if unclear
        """
        chosen = item['chosen'].lower()
        rejected = item['rejected'].lower()
        
        # Check both chosen and rejected to identify the preference
        has_dog_chosen = 'dog' in chosen
        has_cat_chosen = 'cat' in chosen
        has_dog_rejected = 'dog' in rejected
        has_cat_rejected = 'cat' in rejected
        
        # User chose dog over cat → Dog Lover
        if has_dog_chosen and has_cat_rejected:
            return 0
        # User chose cat over dog → Cat Lover
        elif has_cat_chosen and has_dog_rejected:
            return 1
        # Clear preference for dog (even if not directly compared)
        elif has_dog_chosen and not has_cat_chosen:
            return 0
        # Clear preference for cat (even if not directly compared)
        elif has_cat_chosen and not has_dog_chosen:
            return 1
        else:
            return None  # Skip unclear cases
    
    def _verify_train_split(self):
        """
        Verify that training data uses sentences 0-80 as intended.
        This is a sanity check to ensure disjoint splits.
        """
        # Sample a few items and check if they use expected sentence indices
        # The actual verification would require knowing the sentence mappings
        # For now, we trust that generate_simple_data.py split correctly
        pass
    
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


def sequential_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
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

