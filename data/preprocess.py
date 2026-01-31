"""
Data preprocessing pipeline for recommendation system.

Loads raw Amazon review data, filters sparse users/items,
and creates temporal train/val/test splits.

Usage:
    python data/preprocess.py --input data/All_Beauty.jsonl --output-dir data/processed
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Amazon review data")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save processed files"
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=5,
        help="Minimum interactions per user/item (default: 5)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of data for training (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of data for validation (default: 0.15)"
    )
    return parser.parse_args()


def load_jsonl(path: str) -> pd.DataFrame:
    """Load JSONL file and extract relevant columns."""
    logger.info(f"Loading data from {path}")
    
    records = []
    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
            records.append({
                "user_id": record["user_id"],
                "item_id": record["parent_asin"],
                "timestamp": record["timestamp"]
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df):,} interactions")
    return df


def filter_sparse(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    """
    Iteratively filter users and items with fewer than min_interactions.
    
    This is iterative because removing items may cause users to fall below
    the threshold, and vice versa.
    """
    logger.info(f"Filtering users/items with < {min_interactions} interactions")
    
    prev_len = len(df) + 1
    iteration = 0
    
    while len(df) < prev_len:
        prev_len = len(df)
        iteration += 1
        
        # Filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        df = df[df["user_id"].isin(valid_users)]
        
        # Filter items
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_interactions].index
        df = df[df["item_id"].isin(valid_items)]
        
        logger.info(f"  Iteration {iteration}: {len(df):,} interactions remaining")
    
    logger.info(f"After filtering: {len(df):,} interactions")
    return df


def create_id_mappings(df: pd.DataFrame) -> tuple[dict, dict]:
    """Create string -> integer ID mappings for users and items."""
    unique_users = df["user_id"].unique()
    unique_items = df["item_id"].unique()
    
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    
    logger.info(f"Created mappings: {len(user_to_idx):,} users, {len(item_to_idx):,} items")
    return user_to_idx, item_to_idx


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by global timestamp cutoffs.
    
    Sort all interactions by time, then split into train/val/test
    based on the specified ratios.
    """
    logger.info("Performing temporal split")
    
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]
    
    logger.info(f"  Train: {len(train_df):,} ({len(train_df)/n*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df):,} ({len(val_df)/n*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df):,} ({len(test_df)/n*100:.1f}%)")
    
    return train_df, val_df, test_df


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process data
    df = load_jsonl(args.input)
    df = filter_sparse(df, args.min_interactions)
    
    # Create ID mappings
    user_to_idx, item_to_idx = create_id_mappings(df)
    
    # Apply mappings
    df["user_idx"] = df["user_id"].map(user_to_idx)
    df["item_idx"] = df["item_id"].map(item_to_idx)
    
    # Temporal split
    train_df, val_df, test_df = temporal_split(df, args.train_ratio, args.val_ratio)
    
    # Save splits as parquet
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    logger.info(f"Saved splits to {output_dir}")
    
    # Save ID mappings
    with open(output_dir / "user_to_idx.json", "w") as f:
        json.dump(user_to_idx, f)
    with open(output_dir / "item_to_idx.json", "w") as f:
        json.dump(item_to_idx, f)
    logger.info("Saved ID mappings")
    
    # Save statistics (artifact for MLflow later)
    stats = {
        "raw_interactions": len(load_jsonl(args.input)),
        "filtered_interactions": len(df),
        "num_users": len(user_to_idx),
        "num_items": len(item_to_idx),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "min_interactions": args.min_interactions,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats: {stats}")


if __name__ == "__main__":
    main()
