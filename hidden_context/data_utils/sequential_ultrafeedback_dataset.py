"""
Sequential UltraFeedback Dataset for Streaming VPL

This dataset samples episodes (sequences of T interactions) from different user preference pools
in the UltraFeedback dataset. Unlike HH-RLHF, this uses actual multi-turn conversations with
context, making it more realistic for modeling user adaptation.

Uses a rolling window approach over feedbacks from the same user.
"""

import json
import random
from typing import Dict, List, Literal, Optional
import torch
from torch.utils.data import Dataset
import numpy as np


class SequentialUltraFeedbackDataset(Dataset):
    """
    Dataset that samples sequential episodes for Streaming VPL on UltraFeedback data.
    
    Each episode contains T interactions from the same user/aspect:
    - Different subdirectories (1, 2, 4, 8) represent different user preference groups
    
    Uses pre-computed embeddings from the JSONL files and supports rolling windows
    over conversation contexts.
    """
    
    def __init__(
        self,
        data_path: str,
        data_subset: Literal["1", "2", "4", "8", "all"],
        split: Literal["train", "test"],
        seq_length: int = 10,
        epoch_size: int = 1000,
        seed: int = 0,
        min_pool_size: Optional[int] = None,
        use_hard_negatives: bool = True,
        hard_negative_ratio: float = 0.6,
        use_rolling_window: bool = True,
        min_contexts: int = 2
    ):
        """
        Args:
            data_path: Path to data directory (e.g., "data_release/P_4_survey_100/gpt2")
            data_subset: Which subset to use ("1", "2", "4", "8", or "all")
            split: "train" or "test"
            seq_length: Number of interactions per episode (T)
            epoch_size: Arbitrary epoch length (number of episodes per epoch)
            seed: Random seed for reproducibility
            min_pool_size: Minimum samples required per user type (defaults to seq_length)
            use_hard_negatives: If True, preferentially sample hard examples
            hard_negative_ratio: Proportion of hard negatives to sample
            use_rolling_window: If True, use rolling window over contexts
            min_contexts: Minimum number of contexts required for rolling window
        """
        self.data_path = data_path
        self.data_subset = data_subset
        self.split = split
        self.seq_length = seq_length
        self.epoch_size = epoch_size
        self.seed = seed
        self.min_pool_size = min_pool_size or seq_length
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_ratio = hard_negative_ratio
        self.use_rolling_window = use_rolling_window
        self.min_contexts = min_contexts
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Determine which subsets to load
        if data_subset == "all":
            self.subsets = ["1", "2", "4", "8"]
        else:
            self.subsets = [data_subset]
        
        # Load data and cluster by user type (subset)
        self._load_and_cluster_data()
        
        # Determine which pools have enough samples
        self.available_user_types = []
        for user_type in self.pools.keys():
            if len(self.pools[user_type]) >= self.min_pool_size:
                self.available_user_types.append(user_type)
        
        if len(self.available_user_types) == 0:
            raise ValueError(
                f"No pools have enough samples (need at least {self.min_pool_size}). "
                f"Pool sizes: {[(k, len(v)) for k, v in self.pools.items()]}"
            )
        
        # Check if we have multiple user types
        if len(self.available_user_types) < 2:
            print("\n" + "="*80)
            print("WARNING: Dataset contains only ONE user type!")
            for user_type in self.pools.keys():
                print(f"User type {user_type}: {len(self.pools[user_type])} samples")
            print("\nThis will result in high accuracy at t=0 (no adaptation to demonstrate).")
            print("For proper adaptation experiments, use data_subset='all' to load all types.")
            print("="*80 + "\n")
        
        total_samples = sum(len(pool) for pool in self.pools.values())
        print(f"Loaded {total_samples} total samples across {len(self.pools)} user types")
        for user_type in sorted(self.pools.keys()):
            print(f"  User type {user_type}: {len(self.pools[user_type])} samples")
        print(f"Available user types: {sorted(self.available_user_types)}")
        print(f"Split: {self.split}, Seq Length: {seq_length}, Epoch Size: {epoch_size}")
        print(f"Rolling window: {use_rolling_window}, Min contexts: {min_contexts}")
    
    def _load_and_cluster_data(self):
        """Load JSONL data and cluster into user pools by subset."""
        self.pools = {}
        
        for subset in self.subsets:
            file_path = f"{self.data_path}/{subset}/{self.split}.jsonl"
            
            try:
                with open(file_path, 'r') as f:
                    subset_data = []
                    for line in f:
                        item = json.loads(line.strip())
                        # Check if embeddings exist
                        if 'embeddings' not in item:
                            continue
                        
                        # Compute difficulty if using hard negatives
                        if self.use_hard_negatives:
                            item['difficulty'] = self._compute_difficulty(item)
                        
                        subset_data.append(item)
                    
                    # Use subset as user type
                    user_type = int(subset)
                    self.pools[user_type] = subset_data
                    print(f"Loaded {len(subset_data)} samples from subset {subset}")
                    
            except FileNotFoundError:
                print(f"Warning: {file_path} not found, skipping subset {subset}")
                continue
    
    def _compute_difficulty(self, item: Dict) -> float:
        """
        Compute difficulty score for a sample.
        
        "Hard" pairs are those where the distinction is not obvious:
        - Similar embeddings between chosen and rejected
        - Similar context embeddings if available
        
        Returns:
            difficulty: Higher score = harder example (0.0 to 1.0)
        """
        embeddings = item.get('embeddings', {})
        
        # Main embedding similarity
        embed_c = np.array(embeddings.get('embedding_chosen', []))
        embed_r = np.array(embeddings.get('embedding_rejected', []))
        
        if len(embed_c) == 0 or len(embed_r) == 0:
            return 0.5  # Default moderate difficulty
        
        # Cosine similarity
        dot_product = np.dot(embed_c, embed_r)
        norm_c = np.linalg.norm(embed_c)
        norm_r = np.linalg.norm(embed_r)
        
        if norm_c > 0 and norm_r > 0:
            embedding_sim = dot_product / (norm_c * norm_r)
            embedding_sim = (embedding_sim + 1) / 2  # Scale to [0, 1]
        else:
            embedding_sim = 0.5
        
        # If contexts available, also consider context similarity
        context_sim = 0.5
        if 'contexts' in embeddings and len(embeddings['contexts']) > 0:
            # Average similarity across contexts
            context_sims = []
            for ctx in embeddings['contexts']:
                ctx_c = np.array(ctx.get('embedding_chosen', []))
                ctx_r = np.array(ctx.get('embedding_rejected', []))
                if len(ctx_c) > 0 and len(ctx_r) > 0:
                    dot = np.dot(ctx_c, ctx_r)
                    norm_c = np.linalg.norm(ctx_c)
                    norm_r = np.linalg.norm(ctx_r)
                    if norm_c > 0 and norm_r > 0:
                        sim = dot / (norm_c * norm_r)
                        context_sims.append((sim + 1) / 2)
            if context_sims:
                context_sim = np.mean(context_sims)
        
        # Combined difficulty
        difficulty = 0.7 * embedding_sim + 0.3 * context_sim
        
        return difficulty
    
    def _extract_rolling_window(self, item: Dict, window_size: int) -> List[Dict]:
        """
        Extract a rolling window of contexts from an item.
        
        For items with multiple contexts, creates a sequence by selecting
        contexts in order, creating a temporal progression.
        
        Args:
            item: Data item with embeddings and contexts
            window_size: Number of interactions to extract
            
        Returns:
            List of pseudo-items representing the rolling window
        """
        embeddings = item.get('embeddings', {})
        contexts = embeddings.get('contexts', [])
        
        if len(contexts) < self.min_contexts or not self.use_rolling_window:
            # Not enough contexts, return the main item repeated
            return [item] * window_size
        
        # Create rolling window over contexts
        window_items = []
        
        # Start with contexts if available
        num_contexts = len(contexts)
        for i in range(min(window_size, num_contexts)):
            pseudo_item = {
                'embeddings': {
                    'embedding_chosen': contexts[i].get('embedding_chosen', embeddings.get('embedding_chosen', [])),
                    'embedding_rejected': contexts[i].get('embedding_rejected', embeddings.get('embedding_rejected', []))
                },
                'difficulty': item.get('difficulty', 0.5),
                'original_id': item.get('original_id', -1),
                'context_idx': i
            }
            window_items.append(pseudo_item)
        
        # If we need more items, use the final response
        while len(window_items) < window_size:
            pseudo_item = {
                'embeddings': {
                    'embedding_chosen': embeddings.get('embedding_chosen', []),
                    'embedding_rejected': embeddings.get('embedding_rejected', [])
                },
                'difficulty': item.get('difficulty', 0.5),
                'original_id': item.get('original_id', -1),
                'context_idx': -1  # Final response
            }
            window_items.append(pseudo_item)
        
        return window_items[:window_size]
    
    def __len__(self) -> int:
        """Return arbitrary epoch size."""
        return self.epoch_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Sample one episode (sequence of T interactions from same user).
        
        Uses rolling window over contexts if available.
        
        Returns:
            Dictionary with:
                - embeddings_chosen: [T, embed_dim]
                - embeddings_rejected: [T, embed_dim]
                - labels: [T] (all 1s since chosen > rejected by definition)
                - user_type: scalar
        """
        # Randomly select user type from available types
        user_type = random.choice(self.available_user_types)
        
        # Get pool for this user type
        pool = self.pools[user_type]
        
        # Sample items based on difficulty if using hard negatives
        if self.use_hard_negatives and pool and 'difficulty' in pool[0]:
            # Extract difficulty scores
            difficulties = np.array([item.get('difficulty', 0.5) for item in pool])
            
            # Normalize to probabilities (higher difficulty = higher prob)
            weights = difficulties + 0.01
            weights = weights / weights.sum()
            
            # Sample base items
            num_base_items = max(1, self.seq_length // 3)  # Sample fewer base items for rolling window
            indices = np.random.choice(
                len(pool),
                size=min(num_base_items, len(pool)),
                replace=False,
                p=weights
            )
            base_items = [pool[i] for i in indices]
        else:
            # Standard random sampling
            num_base_items = max(1, self.seq_length // 3)
            base_items = random.sample(pool, min(num_base_items, len(pool)))
        
        # Expand with rolling windows if enabled
        episode_items = []
        for base_item in base_items:
            if self.use_rolling_window:
                window_items = self._extract_rolling_window(base_item, self.seq_length // len(base_items) + 1)
                episode_items.extend(window_items)
            else:
                episode_items.append(base_item)
        
        # Ensure we have exactly seq_length items
        if len(episode_items) < self.seq_length:
            # Pad by repeating
            while len(episode_items) < self.seq_length:
                episode_items.append(random.choice(episode_items))
        episode_items = episode_items[:self.seq_length]
        
        # Extract embeddings
        embeddings_chosen = []
        embeddings_rejected = []
        labels = []
        
        for item in episode_items:
            embed_c = item['embeddings']['embedding_chosen']
            embed_r = item['embeddings']['embedding_rejected']
            
            embeddings_chosen.append(embed_c)
            embeddings_rejected.append(embed_r)
            labels.append(1)  # Chosen is always preferred
        
        return {
            'embeddings_chosen': embeddings_chosen,
            'embeddings_rejected': embeddings_rejected,
            'labels': labels,
            'user_type': user_type
        }


def sequential_ultrafeedback_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
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

