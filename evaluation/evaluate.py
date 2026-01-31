"""
Evaluate a trained model on the test set.

Usage:
    python evaluation/evaluate.py --model-path outputs/model.pt --data-dir data/processed
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from evaluation.metrics import evaluate_model
from models.ncf import NCF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to processed data")
    parser.add_argument("--k-values", type=str, default="10,20", help="Comma-separated K values")
    return parser.parse_args()


def load_test_data(data_dir: str) -> dict:
    """Load test set and group by user."""
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    
    user_items = {}
    for _, row in test_df.iterrows():
        user = row["user_idx"]
        item = row["item_idx"]
        if user not in user_items:
            user_items[user] = set()
        user_items[user].add(item)
    
    return user_items


def main():
    args = parse_args()
    k_values = [int(k) for k in args.k_values.split(",")]
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model = NCF(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded model from {args.model_path}")
    
    # Load test data
    user_items_test = load_test_data(args.data_dir)
    logger.info(f"Test set: {len(user_items_test)} users")
    
    # Evaluate
    results = evaluate_model(
        model=model,
        user_items_test=user_items_test,
        num_items=checkpoint["num_items"],
        k_values=k_values,
    )
    
    # Print results
    logger.info("Results:")
    for metric, score in results.items():
        logger.info(f"  {metric}: {score:.4f}")
    
    # Save results
    output_path = Path(args.model_path).parent / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
