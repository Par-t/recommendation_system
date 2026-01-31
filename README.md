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

Download the Amazon Reviews dataset (JSONL format) and place it in the `data/` directory:

```
data/
└── amazon-reviews_dataset.jsonl
```

## Pipeline

### 1. Preprocess
```bash
python data/preprocess.py --input data/amazon-reviews_dataset.jsonl --output-dir data/processed
```

### 2. Train (Single Node)
```bash
python training/train_single.py --data-dir data/processed --epochs 10
```

### 3. Train (Distributed)
```bash
torchrun --nproc_per_node=2 training/train_ddp.py --data-dir data/processed --epochs 10
```

### 4. Evaluate
```bash
python evaluation/evaluate.py --model-path outputs/model.pt --data-dir data/processed
```

### 5. View Experiments
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
│   └── ncf.py              # Neural Collaborative Filtering
├── training/
│   ├── train_single.py     # Single-node + MLflow
│   └── train_ddp.py        # Distributed (PyTorch DDP)
├── evaluation/
│   ├── metrics.py          # Recall@K, NDCG@K
│   └── evaluate.py         # Model evaluation
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
| BPR loss | Ranking > rating prediction |
| Volume mounts | Data stays external to images |
| MLflow | Reproducible experiment tracking |
| DDP with gloo | CPU-based distributed training |
