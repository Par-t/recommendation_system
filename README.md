# Distributed Recommendation System

MLOps-focused recommendation pipeline: distributed training, experiment tracking, containerized serving.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Data

Download the Amazon Reviews dataset (JSONL or CSV format) and place it in the `data/` directory:

```
data/
├── All_Beauty.jsonl         # Small dataset for testing
└── Electronics.csv.gz       # Large dataset for benchmarking
```

## Pipeline

### 1. Preprocess

Supports JSONL, CSV, and CSV.GZ formats. Two split strategies: temporal (default) or leave-one-out.

```bash
# Basic (temporal split)
python data/preprocess.py --input data/All_Beauty.jsonl --output-dir data/processed

# With custom thresholds
python data/preprocess.py --input data/All_Beauty.jsonl --output-dir data/processed --min-user 3 --min-item 3

# Leave-one-out split
python data/preprocess.py --input data/All_Beauty.jsonl --output-dir data/processed --split-strategy leave-one-out

# Large CSV dataset
python data/preprocess.py --input data/Electronics.csv.gz --output-dir data/processed_electronics --min-user 3 --min-item 3
```

Preprocessing is tracked in MLflow under the `preprocessing` experiment.

### 2. Train (Single Node)
```bash
python training/train_single.py --data-dir data/processed --epochs 10
```

### 3. Train (Distributed)
```bash
torchrun --nproc_per_node=2 training/train_ddp.py --data-dir data/processed --epochs 10
```

### 4. Baselines
```bash
python models/baselines.py --data-dir data/processed --baseline all
```
Evaluates popularity, random, and matrix factorization baselines. Results logged to MLflow `baselines` experiment.

### 5. Evaluate NCF
```bash
python evaluation/evaluate.py --model-path outputs/model.pt --data-dir data/processed
```

### 6. Automated Experiment Sweep

Run hyperparameter search across preprocessing and training configs:

```bash
python experiments/sweep.py --input data/All_Beauty.jsonl --config experiments/config.yaml
```

**Features:**
- YAML-based config for preprocessing and training hyperparameters
- Uses DDP by default for faster iteration
- Resource monitoring (auto-adjusts based on CPU/memory)
- Cleans up processed data after each config (reproducible from params)
- All runs tagged with `config_id` for MLflow lineage tracking

**Modes:**
```bash
# Full sweep (preprocess + train for each config)
python experiments/sweep.py --input data/All_Beauty.jsonl --config experiments/config.yaml

# Ablation only (on existing preprocessed data)
python experiments/sweep.py --data-dir data/processed --config experiments/config.yaml
```

### 7. View Experiments
```bash
mlflow ui  # http://127.0.0.1:5000
```

## Serving

### Local
```bash
uvicorn serving.app:app --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d '{"user_id": "USER_ID", "top_k": 5}'
```

### Docker
```bash
# Build
docker build -f docker/Dockerfile.serving -t recommendation-api .

# Run (mount data and model)
docker run -p 8000:8000 \
  -v $(pwd)/data/processed:/app/data/processed \
  -v $(pwd)/outputs:/app/outputs \
  recommendation-api
```

### Docker Training
```bash
docker build -f docker/Dockerfile.training -t recommendation-training .

docker run \
  -v $(pwd)/data/processed:/app/data/processed \
  -v $(pwd)/outputs:/app/outputs \
  recommendation-training
```

## Project Structure

```
├── data/
│   ├── preprocess.py       # Raw data → parquet splits
│   └── dataset.py          # PyTorch Dataset with negative sampling
├── models/
│   ├── ncf.py              # Neural Collaborative Filtering
│   └── baselines.py        # Popularity, Random, MF baselines
├── training/
│   ├── train_single.py     # Single-node + MLflow
│   └── train_ddp.py        # Distributed (PyTorch DDP)
├── evaluation/
│   ├── metrics.py          # Recall@K, NDCG@K
│   └── evaluate.py         # Model evaluation
├── experiments/
│   ├── sweep.py            # Automated hyperparameter sweep
│   ├── resource_monitor.py # CPU/memory monitoring
│   └── config.yaml         # Experiment configurations
├── serving/
│   └── app.py              # FastAPI inference
└── docker/
    ├── Dockerfile.serving
    └── Dockerfile.training
```

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Temporal split | Prevents data leakage |
| Leave-one-out split | Alternative for per-user evaluation |
| Configurable filtering | Balance data size vs. quality |
| Baselines (Pop, MF) | Benchmark to validate NCF improvement |
| BPR loss | Ranking > rating prediction |
| Volume mounts | Data stays external to images |
| MLflow | Tracks preprocessing, baselines, and training experiments |
| DDP with gloo | CPU-based distributed training |
| YAML experiment config | Research-grade, not hardcoded presets |
| DDP by default in sweep | Faster hyperparameter search |
| Separate DDP benchmark | One-time comparison, not every sweep |
| Resource monitoring | Prevents system overload during sweeps |