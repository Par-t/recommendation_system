"""
PyTorch Dataset for recommendation training.

Handles loading preprocessed data and negative sampling.
"""

import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """
    Dataset for (user, positive_item, negative_item) triplets.
    
    Negative items are sampled on-the-fly: for each positive interaction,
    we sample a random item the user hasn't interacted with.
    """
    
    def __init__(self, parquet_path: str, num_items: int, user_positive_items: dict = None):
        """
        Args:
            parquet_path: Path to parquet file with interactions
            num_items: Total number of items (for negative sampling range)
            user_positive_items: Dict mapping user_idx -> set of positive item_idxs.
                                 If None, built from the parquet file.
        """
        df = pd.read_parquet(parquet_path)
        self.users = df["user_idx"].values
        self.items = df["item_idx"].values
        self.num_items = num_items
        
        # Build user -> positive items mapping for negative sampling
        if user_positive_items is not None:
            self.user_positive_items = user_positive_items
        else:
            self.user_positive_items = {}
            for user, item in zip(self.users, self.items):
                if user not in self.user_positive_items:
                    self.user_positive_items[user] = set()
                self.user_positive_items[user].add(item)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]
        
        # Sample negative item (one the user hasn't interacted with)
        neg_item = random.randint(0, self.num_items - 1)
        while neg_item in self.user_positive_items.get(user, set()):
            neg_item = random.randint(0, self.num_items - 1)
        
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )


def load_stats(processed_dir: str) -> dict:
    """Load dataset statistics from stats.json."""
    import json
    with open(Path(processed_dir) / "stats.json") as f:
        return json.load(f)
