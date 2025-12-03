"""
Sequential PRISM Dataset for Streaming VPL

This dataset uses the PRISM alignment dataset which has:
- Real user conversations with LLMs
- User identifiers (can track across multiple conversations)
- Preference ratings on a 1-100 scale
- Multi-turn conversations (2-22 turns)

Unlike HH-RLHF, this dataset naturally has:
1. User identity across multiple interactions
2. Sequential turn numbers within conversations
3. Cardinal preference ratings (not just binary)
"""

import json
import random
from typing import Dict, List, Literal, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict


class SequentialPRISMDataset(Dataset):
    """
    Dataset that samples sequential episodes for Streaming VPL on PRISM data.
    
    Each episode contains T interactions from the same real user.
    Uses pre-computed embeddings from the JSONL files.
    """
    
    def __init__(
        self,
        data_path: str,
        split: Literal["train", "test"],
        seq_length: int = 10,
        epoch_size: int = 1000,
        seed: int = 0,
        min_interactions_per_user: Optional[int] = None,
        use_hard_negatives: bool = True,
        hard_negative_ratio: float = 0.6,
        rating_threshold: float = 10.0,  # Minimum rating difference for preference
        use_turn_order: bool = True,  # Whether to preserve turn order
    ):
        """
        Args:
            data_path: Path to processed PRISM data (JSONL with embeddings)
            split: "train" or "test"
            seq_length: Number of interactions per episode (T)
            epoch_size: Arbitrary epoch length (number of episodes per epoch)
            seed: Random seed for reproducibility
            min_interactions_per_user: Minimum interactions required per user
            use_hard_negatives: If True, preferentially sample hard examples
            hard_negative_ratio: Proportion of hard negatives to sample
            rating_threshold: Minimum rating difference to consider a clear preference
            use_turn_order: If True, sample interactions in chronological order
        """
        self.data_path = data_path
        self.split = split
        self.seq_length = seq_length
        self.epoch_size = epoch_size
        self.seed = seed
        self.min_interactions_per_user = min_interactions_per_user or 1
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_ratio = hard_negative_ratio
        self.rating_threshold = rating_threshold
        self.use_turn_order = use_turn_order
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Load data and organize by user
        self._load_and_organize_data()
        
        # Filter users with enough interactions
        self.available_users = [
            user_id for user_id, interactions in self.user_pools.items()
            if len(interactions) >= self.min_interactions_per_user
        ]
        
        if len(self.available_users) == 0:
            raise ValueError(
                f"No users have enough interactions (need at least {self.min_interactions_per_user})."
            )
        
        print(f"Loaded {len(self.user_pools)} total users")
        print(f"Available users with >={self.min_interactions_per_user} interactions: {len(self.available_users)}")
        print(f"Total interactions: {sum(len(pool) for pool in self.user_pools.values())}")
        print(f"Split: {self.split}, Seq Length: {seq_length}, Epoch Size: {epoch_size}")
    
    def _load_and_organize_data(self):
        """Load JSONL data and organize by user."""
        file_path = f"{self.data_path}/{self.split}.jsonl"
        
        try:
            with open(file_path, 'r') as f:
                all_data = [json.loads(line.strip()) for line in f]
        except FileNotFoundError:
            print(f"Error: {file_path} not found!")
            print(f"Make sure you've run the PRISM preprocessing script first.")
            raise
        
        print(f"Loaded {len(all_data)} total interactions")
        
        # Organize by user_id
        self.user_pools = defaultdict(list)
        
        for item in all_data:
            # Check if embeddings exist
            if 'embeddings' not in item:
                continue
            
            user_id = item.get('user_id')
            if user_id is None:
                continue
            
            # Compute difficulty if using hard negatives
            if self.use_hard_negatives:
                item['difficulty'] = self._compute_difficulty(item)
            
            self.user_pools[user_id].append(item)
        
        # Sort interactions by turn number for each user if using turn order
        if self.use_turn_order:
            for user_id in self.user_pools:
                self.user_pools[user_id].sort(
                    key=lambda x: (x.get('conversation_id', ''), x.get('turn_number', 0))
                )
        
        # Convert to regular dict
        self.user_pools = dict(self.user_pools)
        
        # Print sample statistics
        if len(self.user_pools) > 0:
            sample_user = list(self.user_pools.keys())[0]
            sample_interactions = self.user_pools[sample_user]
            print(f"Sample user {sample_user}: {len(sample_interactions)} interactions")
            if len(sample_interactions) > 0:
                print(f"Sample interaction keys: {list(sample_interactions[0].keys())}")
    
    def _compute_difficulty(self, item: Dict) -> float:
        """
        Compute difficulty score for a sample.
        
        For PRISM, "hard" pairs are those where:
        - Rating difference is small (close call)
        - Responses are similar in length
        - Embeddings are similar
        
        Returns:
            difficulty: Higher score = harder example (0.0 to 1.0)
        """
        # Rating similarity (harder when ratings are close)
        rating_diff = abs(item.get('rating_chosen', 50) - item.get('rating_rejected', 50))
        rating_similarity = 1.0 - min(rating_diff / 100.0, 1.0)  # Normalize to [0, 1]
        
        # Length similarity
        chosen = item.get('chosen', '')
        rejected = item.get('rejected', '')
        len_chosen = len(chosen)
        len_rejected = len(rejected)
        max_len = max(len_chosen, len_rejected, 1)
        min_len = min(len_chosen, len_rejected, 1)
        length_ratio = min_len / max_len
        
        # Embedding similarity
        embedding_sim = 0.5
        if 'embeddings' in item:
            embed_c = np.array(item['embeddings']['embedding_chosen'])
            embed_r = np.array(item['embeddings']['embedding_rejected'])
            dot_product = np.dot(embed_c, embed_r)
            norm_c = np.linalg.norm(embed_c)
            norm_r = np.linalg.norm(embed_r)
            if norm_c > 0 and norm_r > 0:
                embedding_sim = dot_product / (norm_c * norm_r)
                embedding_sim = (embedding_sim + 1) / 2  # Scale to [0, 1]
        
        # Combined difficulty: weight rating similarity more heavily
        difficulty = 0.4 * rating_similarity + 0.3 * length_ratio + 0.3 * embedding_sim
        
        return difficulty
    
    def __len__(self) -> int:
        """Return arbitrary epoch size."""
        return self.epoch_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Sample one episode (sequence of up to T interactions from same user).
        Supports variable-length sequences with masking.
        
        Returns:
            Dictionary with:
                - embeddings_chosen: list of embeddings (will be padded in collate_fn)
                - embeddings_rejected: list of embeddings (will be padded in collate_fn)
                - labels: list (all 1s since chosen > rejected by definition)
                - user_id: string identifier
                - turn_numbers: list (turn indices if available)
                - actual_length: int (actual number of interactions, ≤T)
                - mask: list of 0/1 (1 for real data, 0 for padding)
        """
        # Randomly select user from available users
        user_id = random.choice(self.available_users)
        
        # Get user's interaction pool
        pool = self.user_pools[user_id]
        actual_length = len(pool)
        
        # Sample interactions based on what's available
        if actual_length >= self.seq_length:
            # User has enough interactions - sample seq_length of them
            if self.use_turn_order:
                max_start = actual_length - self.seq_length
                start_idx = random.randint(0, max_start)
                episode_items = pool[start_idx:start_idx + self.seq_length]
            elif self.use_hard_negatives and 'difficulty' in pool[0]:
                difficulties = np.array([item['difficulty'] for item in pool])
                weights = difficulties + 0.01
                weights = weights / weights.sum()
                indices = np.random.choice(
                    actual_length,
                    size=self.seq_length,
                    replace=False,
                    p=weights
                )
                episode_items = [pool[i] for i in indices]
            else:
                episode_items = random.sample(pool, self.seq_length)
            actual_seq_length = self.seq_length
        else:
            # User has fewer than seq_length interactions - use all of them
            # No padding here, will be done in collate_fn
            episode_items = pool
            actual_seq_length = actual_length
        
        # Extract embeddings and metadata
        embeddings_chosen = []
        embeddings_rejected = []
        labels = []
        turn_numbers = []
        
        for item in episode_items:
            embed_chosen = item['embeddings']['embedding_chosen']
            embed_rejected = item['embeddings']['embedding_rejected']
            
            embeddings_chosen.append(embed_chosen)
            embeddings_rejected.append(embed_rejected)
            labels.append(1)  # Chosen is always preferred
            turn_numbers.append(item.get('turn_number', 0))
        
        # Create mask: 1 for real data, 0 for padding
        mask = [1] * actual_seq_length + [0] * (self.seq_length - actual_seq_length)
        
        return {
            'embeddings_chosen': embeddings_chosen,
            'embeddings_rejected': embeddings_rejected,
            'labels': labels,
            'user_id': user_id,
            'turn_numbers': turn_numbers,
            'actual_length': actual_seq_length,
            'mask': mask
        }


def sequential_prism_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function to batch multiple episodes with variable lengths.
    Handles padding for shorter sequences.
    
    Args:
        batch: List of episodes from __getitem__
        
    Returns:
        Dictionary with batched tensors:
            - embeddings_chosen: [batch_size, max_seq_len, embed_dim] (padded)
            - embeddings_rejected: [batch_size, max_seq_len, embed_dim] (padded)
            - labels: [batch_size, max_seq_len] (padded with 0s)
            - user_ids: List of user_id strings
            - turn_numbers: [batch_size, max_seq_len] (padded with 0s)
            - mask: [batch_size, max_seq_len] (1 for real data, 0 for padding)
            - actual_lengths: [batch_size] (actual sequence lengths)
    """
    batch_size = len(batch)
    
    # Get max sequence length (from mask)
    max_seq_len = len(batch[0]['mask'])
    
    # Get embedding dimension from first real embedding
    embed_dim = len(batch[0]['embeddings_chosen'][0])
    
    # Initialize tensors with zeros (padding)
    embeddings_chosen = torch.zeros(batch_size, max_seq_len, embed_dim)
    embeddings_rejected = torch.zeros(batch_size, max_seq_len, embed_dim)
    labels = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    turn_numbers = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    actual_lengths = torch.zeros(batch_size, dtype=torch.long)
    user_ids = []
    
    # Fill tensors with actual data
    for i, episode in enumerate(batch):
        actual_len = episode['actual_length']
        actual_lengths[i] = actual_len
        
        # Only fill up to actual_length
        embeddings_chosen[i, :actual_len] = torch.tensor(episode['embeddings_chosen'])
        embeddings_rejected[i, :actual_len] = torch.tensor(episode['embeddings_rejected'])
        labels[i, :actual_len] = torch.tensor(episode['labels'])
        turn_numbers[i, :actual_len] = torch.tensor(episode['turn_numbers'])
        mask[i] = torch.tensor(episode['mask'], dtype=torch.bool)
        user_ids.append(episode['user_id'])
    
    return {
        'embeddings_chosen': embeddings_chosen,
        'embeddings_rejected': embeddings_rejected,
        'labels': labels,
        'user_ids': user_ids,
        'turn_numbers': turn_numbers,
        'mask': mask,
        'actual_lengths': actual_lengths
    }

