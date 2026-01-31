"""
Distributed Data Parallel training script for NCF model.

Uses PyTorch DDP with gloo backend (CPU) for multi-process training.

Usage:
    torchrun --nproc_per_node=2 training/train_ddp.py --data-dir data/processed --epochs 10
"""

import argparse
import logging
import os
import time
from pathlib import Path

import mlflow
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.dataset import InteractionDataset, load_stats
from models.ncf import NCF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Rank %(rank)s] %(message)s"
)


class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
    
    def filter(self, record):
        record.rank = self.rank
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NCF model with DDP")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to processed data")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per process")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for model artifacts")
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed process group."""
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def cleanup_distributed():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def train_epoch(
    model: DDP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    
    for users, pos_items, neg_items in dataloader:
        optimizer.zero_grad()
        loss = model.module.bpr_loss(users, pos_items, neg_items)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    args = parse_args()
    
    # Setup distributed
    rank, world_size = setup_distributed()
    
    # Setup logging with rank
    logger = logging.getLogger(__name__)
    logger.addFilter(RankFilter(rank))
    
    # Load stats
    stats = load_stats(args.data_dir)
    num_users = stats["num_users"]
    num_items = stats["num_items"]
    
    if rank == 0:
        logger.info(f"Dataset: {num_users} users, {num_items} items")
        logger.info(f"World size: {world_size} processes")
    
    # Create dataset
    train_dataset = InteractionDataset(
        parquet_path=f"{args.data_dir}/train.parquet",
        num_items=num_items,
    )
    
    # Distributed sampler - splits data across processes
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )
    
    # Create model and wrap in DDP
    model = NCF(num_users, num_items, args.embedding_dim)
    model = DDP(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Only rank 0 handles MLflow and saving
    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mlflow.set_experiment("ncf-training")
        mlflow.start_run()
        mlflow.log_params({
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": args.embedding_dim,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "training_mode": "ddp",
            "world_size": world_size,
        })
    
    # Training loop
    if rank == 0:
        logger.info("Starting distributed training...")
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Important: set epoch for sampler to shuffle differently each epoch
        sampler.set_epoch(epoch)
        
        epoch_start = time.time()
        loss = train_epoch(model, train_loader, optimizer)
        epoch_time = time.time() - epoch_start
        
        # Sync loss across all processes for accurate logging
        loss_tensor = torch.tensor([loss])
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
        
        if rank == 0:
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "epoch_time": epoch_time,
            }, step=epoch)
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    
    # Calculate throughput (total samples across all processes)
    total_samples = len(train_dataset) * args.epochs
    throughput = total_samples / total_time
    
    if rank == 0:
        mlflow.log_metrics({
            "total_time": total_time,
            "throughput_samples_per_sec": throughput,
        })
        logger.info(f"Training complete in {total_time:.2f}s ({throughput:.0f} samples/sec)")
        
        # Save model (from rank 0 only)
        model_path = output_dir / "model.pt"
        torch.save({
            "model_state_dict": model.module.state_dict(),  # .module to unwrap DDP
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": args.embedding_dim,
        }, model_path)
        
        mlflow.log_artifact(model_path)
        logger.info(f"Model saved to {model_path}")
        mlflow.end_run()
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
