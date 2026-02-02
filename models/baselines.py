"""
Baseline models for recommendation evaluation.
All implement same interface as NCF for compatibility with evaluate_model.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class PopularityBaseline(nn.Module):
    """Recommends most popular items to all users."""
    
    def __init__(self, popularity_scores: np.ndarray):
        super().__init__()
        self.popularity = torch.tensor(popularity_scores, dtype=torch.float32)
    
    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return self.popularity[items]
    
    @classmethod
    def from_train_data(cls, train_path: str, num_items: int) -> "PopularityBaseline":
        train_df = pd.read_parquet(train_path)
        counts = train_df["item_idx"].value_counts()
        scores = np.zeros(num_items)
        scores[counts.index] = counts.values
        scores = scores / scores.max()  # Normalize to [0, 1]
        return cls(scores)


class RandomBaseline(nn.Module):
    """Random scores. Sanity check baseline."""
    
    def __init__(self, num_items: int, seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.scores = torch.tensor(rng.rand(num_items), dtype=torch.float32)
    
    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return self.scores[items]


class MatrixFactorizationBaseline(nn.Module):
    """Matrix Factorization via truncated SVD."""
    
    def __init__(self, user_factors: np.ndarray, item_factors: np.ndarray):
        super().__init__()
        self.user_factors = torch.tensor(user_factors, dtype=torch.float32)
        self.item_factors = torch.tensor(item_factors, dtype=torch.float32)
    
    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_factors[users] * self.item_factors[items]).sum(dim=-1)
    
    @classmethod
    def from_train_data(cls, train_path: str, num_users: int, num_items: int, n_factors: int = 32) -> "MatrixFactorizationBaseline":
        train_df = pd.read_parquet(train_path)
        
        # Sparse interaction matrix
        rows, cols = train_df["user_idx"].values, train_df["item_idx"].values
        matrix = csr_matrix((np.ones(len(train_df)), (rows, cols)), shape=(num_users, num_items))
        
        # SVD
        k = min(n_factors, min(num_users, num_items) - 1)
        U, sigma, Vt = svds(matrix.astype(float), k=k)
        sqrt_sigma = np.sqrt(sigma)
        
        return cls(U * sqrt_sigma, Vt.T * sqrt_sigma)


def evaluate_baseline(baseline_name: str, data_dir: str, k_values: list = [10, 20], log_mlflow: bool = True) -> dict:
    """Evaluate a baseline and optionally log to MLflow."""
    import mlflow
    from evaluation.metrics import evaluate_model
    
    with open(f"{data_dir}/stats.json") as f:
        stats = json.load(f)
    num_users, num_items = stats["num_users"], stats["num_items"]
    
    # Create baseline
    if baseline_name == "popularity":
        baseline = PopularityBaseline.from_train_data(f"{data_dir}/train.parquet", num_items)
    elif baseline_name == "random":
        baseline = RandomBaseline(num_items)
    elif baseline_name == "mf":
        baseline = MatrixFactorizationBaseline.from_train_data(f"{data_dir}/train.parquet", num_users, num_items)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")
    
    # Load test data grouped by user
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    user_items_test = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    
    results = evaluate_model(baseline, user_items_test, num_items, k_values)
    
    if log_mlflow:
        mlflow.set_experiment("baselines")
        with mlflow.start_run(run_name=f"baseline_{baseline_name}"):
            mlflow.log_params({"baseline": baseline_name, "num_users": num_users, "num_items": num_items})
            mlflow.log_metrics(results)
    
    return results


if __name__ == "__main__":
    import argparse
    import logging
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--baseline", type=str, choices=["popularity", "random", "mf", "all"], default="all")
    parser.add_argument("--k-values", type=str, default="10,20")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    
    k_values = [int(k) for k in args.k_values.split(",")]
    baselines = ["popularity", "random", "mf"] if args.baseline == "all" else [args.baseline]
    
    for name in baselines:
        logger.info(f"Evaluating {name}...")
        results = evaluate_baseline(name, args.data_dir, k_values, log_mlflow=not args.no_mlflow)
        for metric, score in results.items():
            logger.info(f"  {metric}: {score:.4f}")
