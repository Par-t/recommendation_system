"""
Automated experiment sweep.

Modes:
1. Config file: Define preprocessing + training configs in YAML
   python experiments/sweep.py --input data/All_Beauty.jsonl --config experiments/config.yaml

2. Ablation: Run training configs on existing preprocessed data
   python experiments/sweep.py --data-dir data/processed_a3f2c1d8 --config experiments/config.yaml

All runs tagged with config_id for lineage tracking in MLflow.
"""

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import mlflow
import yaml

from experiments.resource_monitor import pre_training_check, check_resources

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load experiment config from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_config_id(config: dict) -> str:
    """Generate unique ID from preprocessing config."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def run_preprocessing(input_file: str, config: dict, config_id: str) -> str:
    """Run preprocessing and return output directory."""
    output_dir = f"data/processed_{config_id}"
    
    cmd = [
        "python", "data/preprocess.py",
        "--input", input_file,
        "--output-dir", output_dir,
        "--min-user", str(config["min_user"]),
        "--min-item", str(config["min_item"]),
        "--split-strategy", config["split_strategy"],
    ]
    if config["split_strategy"] == "temporal":
        cmd.extend(["--train-ratio", str(config.get("train_ratio", 0.7))])
    
    logger.info(f"Preprocessing: {config}")
    subprocess.run(cmd, check=True)
    
    # Tag the MLflow run with config_id
    mlflow.set_experiment("preprocessing")
    runs = mlflow.search_runs(experiment_names=["preprocessing"], max_results=1, order_by=["start_time DESC"])
    if len(runs) > 0:
        run_id = runs.iloc[0]["run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("config_id", config_id)
    
    return output_dir


def run_baselines(data_dir: str, config_id: str):
    """Run all baselines."""
    logger.info(f"Running baselines on {data_dir}")
    
    for baseline in ["popularity", "random", "mf"]:
        cmd = ["python", "models/baselines.py", "--data-dir", data_dir, "--baseline", baseline]
        subprocess.run(cmd, check=True)
        
        mlflow.set_experiment("baselines")
        runs = mlflow.search_runs(experiment_names=["baselines"], max_results=1, order_by=["start_time DESC"])
        if len(runs) > 0:
            run_id = runs.iloc[0]["run_id"]
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tag("config_id", config_id)


def run_training(data_dir: str, train_config: dict, config_id: str, use_ddp: bool = True, nproc: int = 2):
    """Run NCF training (DDP by default for speed)."""
    logger.info(f"Training: {train_config} (ddp={use_ddp}, nproc={nproc})")
    
    if use_ddp:
        cmd = [
            "torchrun", f"--nproc_per_node={nproc}",
            "training/train_ddp.py",
        ]
    else:
        cmd = ["python", "training/train_single.py"]
    
    cmd.extend([
        "--data-dir", data_dir,
        "--embedding-dim", str(train_config["embedding_dim"]),
        "--lr", str(train_config["lr"]),
        "--epochs", str(train_config["epochs"]),
    ])
    subprocess.run(cmd, check=True)
    
    mlflow.set_experiment("training")
    runs = mlflow.search_runs(experiment_names=["training"], max_results=1, order_by=["start_time DESC"])
    if len(runs) > 0:
        run_id = runs.iloc[0]["run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("config_id", config_id)
            mlflow.set_tag("data_dir", data_dir)


def run_ablation(data_dir: str, train_configs: list, config_id: str = None, 
                 include_baselines: bool = False, use_ddp: bool = True, nproc: int = 2):
    """Run training ablations on preprocessed data."""
    if config_id is None:
        with open(f"{data_dir}/stats.json") as f:
            stats = json.load(f)
        config_id = stats.get("config_id", Path(data_dir).name.replace("processed_", ""))
    
    logger.info(f"Running ablation: {data_dir} (config_id={config_id}, ddp={use_ddp})")
    
    for i, train_config in enumerate(train_configs):
        # Health check between runs
        is_safe, msg = check_resources()
        if not is_safe:
            logger.warning(f"Resources strained: {msg}. Pausing 30s...")
            time.sleep(30)
        
        logger.info(f"  Training {i+1}/{len(train_configs)}: {train_config}")
        run_training(data_dir, train_config, config_id, use_ddp, nproc)
    
    if include_baselines:
        run_baselines(data_dir, config_id)


def run_sweep(input_file: str, preprocess_configs: list, train_configs: list, 
              include_baselines: bool = False, use_ddp: bool = True, nproc: int = 2):
    """Run full sweep: preprocess + ablation for each config."""
    # Pre-flight resource check
    if use_ddp:
        should_proceed, adjusted_nproc = pre_training_check(nproc)
        if not should_proceed:
            logger.error("System resources critical. Aborting sweep.")
            return
        nproc = adjusted_nproc
    
    logger.info(f"Starting sweep: {len(preprocess_configs)} preprocess configs (ddp={use_ddp}, nproc={nproc})")
    start_time = time.time()
    
    for i, preprocess_config in enumerate(preprocess_configs):
        config_id = generate_config_id(preprocess_config)
        logger.info(f"\n{'='*60}\nConfig {i+1}/{len(preprocess_configs)}: {config_id}\n{'='*60}")
        
        data_dir = run_preprocessing(input_file, preprocess_config, config_id)
        run_ablation(data_dir, train_configs, config_id, include_baselines, use_ddp, nproc)
        
        # Clean up processed data (reproducible from params, no need to keep)
        logger.info(f"Cleaning up: {data_dir}")
        shutil.rmtree(data_dir)
    
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}\nSweep complete in {elapsed/60:.1f} minutes\n{'='*60}")
    logger.info(f"View results: mlflow ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment sweep or ablation")
    parser.add_argument("--input", type=str, help="Input data file (for full sweep)")
    parser.add_argument("--data-dir", type=str, help="Preprocessed data dir (for ablation only)")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--include-baselines", action="store_true", help="Also run baselines")
    parser.add_argument("--no-ddp", action="store_true", help="Use single-node training instead of DDP")
    parser.add_argument("--nproc", type=int, default=2, help="Number of processes for DDP (default: 2)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_configs = config["training"]
    use_ddp = not args.no_ddp
    
    if args.data_dir:
        run_ablation(args.data_dir, train_configs, include_baselines=args.include_baselines, 
                     use_ddp=use_ddp, nproc=args.nproc)
    elif args.input:
        preprocess_configs = config["preprocessing"]
        run_sweep(args.input, preprocess_configs, train_configs, args.include_baselines, 
                  use_ddp=use_ddp, nproc=args.nproc)
    else:
        parser.error("Provide either --input (full sweep) or --data-dir (ablation)")
