"""
Neural Collaborative Filtering model.

Simple architecture: user/item embeddings with dot-product scoring.
"""

import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    Neural Collaborative Filtering with dot-product scoring.
    
    Architecture:
        user_idx -> user_embedding -> \
                                       dot product -> score
        item_idx -> item_embedding -> /
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute scores for (user, item) pairs.
        
        Args:
            user_idx: (batch_size,) tensor of user indices
            item_idx: (batch_size,) tensor of item indices
            
        Returns:
            (batch_size,) tensor of scores
        """
        user_emb = self.user_embedding(user_idx)  # (batch, dim)
        item_emb = self.item_embedding(item_idx)  # (batch, dim)
        
        # Dot product scoring
        scores = (user_emb * item_emb).sum(dim=1)  # (batch,)
        return scores
    
    def bpr_loss(
        self,
        user_idx: torch.Tensor,
        pos_item_idx: torch.Tensor,
        neg_item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bayesian Personalized Ranking loss.
        
        Encourages: score(user, pos_item) > score(user, neg_item)
        """
        pos_scores = self.forward(user_idx, pos_item_idx)
        neg_scores = self.forward(user_idx, neg_item_idx)
        
        # BPR loss: -log(sigmoid(pos - neg))
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        return loss
