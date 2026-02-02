"""
Data preprocessing pipeline for recommendation system.

Supports JSONL and CSV input formats.
Split strategies: temporal (global time cutoff) or leave-one-out (per-user).

Usage:
    python data/preprocess.py --input data/reviews.jsonl --output-dir data/processed
    python data/preprocess.py --input data/reviews.csv --output-dir data/processed --split-strategy leave-one-out
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import mlflow
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess recommendation data")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw data file (JSONL or CSV)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save processed files"
    )
    parser.add_argument(
        "--min-user",
        type=int,
        default=5,
        help="Minimum interactions per user (default: 5)"
    )
    parser.add_argument(
        "--min-item",
        type=int,
        default=5,
        help="Minimum interactions per item (default: 5)"
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["temporal", "leave-one-out"],
        default="temporal",
        help="Split strategy: temporal or leave-one-out (default: temporal)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train ratio for temporal split (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Val ratio for temporal split (default: 0.15)"
    )
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    """Load data from JSONL, CSV, or CSV.GZ. Returns DataFrame with user_id, item_id, timestamp."""
    path = Path(path)
    logger.info(f"Loading data from {path}")
    
    suffixes = path.suffixes  # e.g. ['.csv', '.gz'] for file.csv.gz
    
    if path.suffix == ".jsonl":
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
    elif path.suffix == ".csv" or suffixes == [".csv", ".gz"]:
        # Pandas auto-detects gzip from .gz extension
        df = pd.read_csv(path)
        # Expect columns: user_id, item_id, timestamp (or rename common variants)
        col_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if "user" in col_lower:
                col_map[col] = "user_id"
            elif "item" in col_lower or "product" in col_lower or "asin" in col_lower:
                col_map[col] = "item_id"
            elif "time" in col_lower or "date" in col_lower:
                col_map[col] = "timestamp"
        df = df.rename(columns=col_map)
        df = df[["user_id", "item_id", "timestamp"]]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .jsonl, .csv, or .csv.gz")
    
    logger.info(f"Loaded {len(df):,} interactions")
    return df


def filter_sparse(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Iteratively filter users/items below thresholds."""
    logger.info(f"Filtering: min_user={min_user}, min_item={min_item}")
    
    prev_len = len(df) + 1
    iteration = 0
    
    while len(df) < prev_len:
        prev_len = len(df)
        iteration += 1
        
        # Filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        df = df[df["user_id"].isin(valid_users)]
        
        # Filter items
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item].index
        df = df[df["item_id"].isin(valid_items)]
        
        logger.info(f"  Iteration {iteration}: {len(df):,} remaining")
    
    logger.info(f"After filtering: {len(df):,} interactions")
    return df


def create_id_mappings(df: pd.DataFrame) -> tuple[dict, dict]:
    """Create string -> integer ID mappings."""
    unique_users = df["user_id"].unique()
    unique_items = df["item_id"].unique()
    
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    
    logger.info(f"Mappings: {len(user_to_idx):,} users, {len(item_to_idx):,} items")
    return user_to_idx, item_to_idx


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by global timestamp cutoffs."""
    logger.info("Temporal split")
    
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]
    
    logger.info(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df


def leave_one_out_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leave-one-out split: per user, hold out last interaction for test,
    second-to-last for validation, rest for training.
    
    Users with < 3 interactions go entirely to training.
    """
    logger.info("Leave-one-out split")
    
    df_sorted = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    
    train_rows, val_rows, test_rows = [], [], []
    
    for user_id, group in df_sorted.groupby("user_id"):
        interactions = group.to_dict("records")
        n = len(interactions)
        
        if n >= 3:
            train_rows.extend(interactions[:-2])
            val_rows.append(interactions[-2])
            test_rows.append(interactions[-1])
        else:
            # Not enough interactions for val/test, put all in train
            train_rows.extend(interactions)
    
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    
    logger.info(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df


def generate_config_id(args) -> str:
    """Generate unique config ID from preprocessing parameters."""
    config = {
        "min_user": args.min_user,
        "min_item": args.min_item,
        "split_strategy": args.split_strategy,
        "train_ratio": args.train_ratio if args.split_strategy == "temporal" else None,
    }
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def main():
    args = parse_args()
    
    # Generate config ID for tracking
    config_id = generate_config_id(args)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start MLflow run for data lineage
    mlflow.set_experiment("preprocessing")
    with mlflow.start_run(run_name=f"preprocess_{config_id}"):
        mlflow.set_tag("config_id", config_id)
        # Log parameters
        mlflow.log_params({
            "input_file": Path(args.input).name,
            "split_strategy": args.split_strategy,
            "min_user": args.min_user,
            "min_item": args.min_item,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
        })
        
        # Load and filter
        df = load_data(args.input)
        raw_count = len(df)
        df = filter_sparse(df, args.min_user, args.min_item)
        
        # ID mappings
        user_to_idx, item_to_idx = create_id_mappings(df)
        df["user_idx"] = df["user_id"].map(user_to_idx)
        df["item_idx"] = df["item_id"].map(item_to_idx)
        
        # Split
        if args.split_strategy == "temporal":
            train_df, val_df, test_df = temporal_split(df, args.train_ratio, args.val_ratio)
        else:
            train_df, val_df, test_df = leave_one_out_split(df)
        
        # Log metrics
        mlflow.log_metrics({
            "raw_interactions": raw_count,
            "filtered_interactions": len(df),
            "num_users": len(user_to_idx),
            "num_items": len(item_to_idx),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "retention_rate": len(df) / raw_count,
        })
        
        # Save splits
        train_df.to_parquet(output_dir / "train.parquet", index=False)
        val_df.to_parquet(output_dir / "val.parquet", index=False)
        test_df.to_parquet(output_dir / "test.parquet", index=False)
        logger.info(f"Saved splits to {output_dir}")
        
        # Save mappings
        with open(output_dir / "user_to_idx.json", "w") as f:
            json.dump(user_to_idx, f)
        with open(output_dir / "item_to_idx.json", "w") as f:
            json.dump(item_to_idx, f)
        
        # Save stats
        stats = {
            "config_id": config_id,
            "raw_interactions": raw_count,
            "filtered_interactions": len(df),
            "num_users": len(user_to_idx),
            "num_items": len(item_to_idx),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "split_strategy": args.split_strategy,
            "min_user": args.min_user,
            "min_item": args.min_item,
        }
        if args.split_strategy == "temporal":
            stats["train_ratio"] = args.train_ratio
            stats["val_ratio"] = args.val_ratio
        
        with open(output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        # Log stats file as artifact
        mlflow.log_artifact(output_dir / "stats.json")
        
        # Clear output for user
        logger.info(f"\n{'='*60}")
        logger.info(f"CONFIG_ID: {config_id}")
        logger.info(f"OUTPUT: {output_dir}")
        logger.info(f"{'='*60}")
        logger.info(f"To run training ablations on this data:")
        logger.info(f"  python experiments/sweep.py --data-dir {output_dir}")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
