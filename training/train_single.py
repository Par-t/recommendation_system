"""
Single-process training script for NCF model.

Usage:
    python training/train_single.py --data-dir data/processed --epochs 10
"""

import argparse
import logging
import time
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from data.dataset import InteractionDataset, load_stats
from models.ncf import NCF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NCF model")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to processed data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for model artifacts")
    return parser.parse_args()


def train_epoch(model: NCF, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    
    for users, pos_items, neg_items in dataloader:
        optimizer.zero_grad()
        loss = model.bpr_loss(users, pos_items, neg_items)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    args = parse_args()
    
    # Load stats
    stats = load_stats(args.data_dir)
    num_users = stats["num_users"]
    num_items = stats["num_items"]
    
    logger.info(f"Dataset: {num_users} users, {num_items} items")
    
    # Create dataset and dataloader
    train_dataset = InteractionDataset(
        parquet_path=f"{args.data_dir}/train.parquet",
        num_items=num_items,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Create model and optimizer
    model = NCF(num_users, num_items, args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # MLflow tracking
    mlflow.set_experiment("ncf-training")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": args.embedding_dim,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "training_mode": "single",
        })
        
        # Training loop
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(args.epochs):
            epoch_start = time.time()
            loss = train_epoch(model, train_loader, optimizer)
            epoch_time = time.time() - epoch_start
            
            mlflow.log_metrics({
                "train_loss": loss,
                "epoch_time": epoch_time,
            }, step=epoch)
            
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f} - Time: {epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        throughput = len(train_dataset) * args.epochs / total_time
        
        mlflow.log_metrics({
            "total_time": total_time,
            "throughput_samples_per_sec": throughput,
        })
        
        logger.info(f"Training complete in {total_time:.2f}s ({throughput:.0f} samples/sec)")
        
        # Save model
        model_path = output_dir / "model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": args.embedding_dim,
        }, model_path)
        
        mlflow.log_artifact(model_path)
        logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
