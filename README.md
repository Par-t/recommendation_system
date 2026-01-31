# Distributed Recommendation System

An MLOps-focused recommendation system demonstrating distributed training, experiment tracking, and Kubernetes-based serving.

**Focus**: ML systems engineering — not chasing SOTA accuracy.

## Project Structure

```
recommendation/
├── data/
│   ├── preprocess.py      # Data pipeline
│   └── dataset.py         # PyTorch Dataset
├── models/
│   └── ncf.py             # Neural Collaborative Filtering
├── training/
│   └── train_single.py    # Single-node training with MLflow
├── evaluation/
│   ├── metrics.py         # Recall@K, NDCG@K
│   └── evaluate.py        # Model evaluation script
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Usage

### 1. Preprocess Data

Download the Amazon Reviews dataset (e.g., All_Beauty) and place the JSONL file in `data/`.

```bash
python data/preprocess.py \
    --input data/All_Beauty.jsonl \
    --output-dir data/processed \
    --min-interactions 5
```

### 2. Train Model

```bash
python training/train_single.py \
    --data-dir data/processed \
    --epochs 10 \
    --batch-size 256 \
    --embedding-dim 64
```

### 3. Evaluate

```bash
python evaluation/evaluate.py \
    --model-path outputs/model.pt \
    --data-dir data/processed
```

### 4. View Experiments

```bash
mlflow ui
# Open http://127.0.0.1:5000
```

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Temporal split | Prevents data leakage (no future information in training) |
| BPR loss | Optimizes ranking, not rating prediction |
| Filter <5 interactions | Removes noise from sparse users/items |
| MLflow tracking | Reproducible experiments with logged params/metrics |
| Editable install (`pip install -e .`) | Production-style package structure |

## Metrics

- **Recall@K**: Fraction of relevant items in top-K recommendations
- **NDCG@K**: Ranking quality (rewards relevant items appearing earlier)
- **Throughput**: Samples processed per second during training

## Roadmap

- [x] Data preprocessing pipeline
- [x] NCF model implementation
- [x] Single-node training with MLflow
- [x] Evaluation metrics (Recall@K, NDCG@K)
- [ ] Distributed training (PyTorch DDP)
- [ ] FastAPI serving endpoint
- [ ] Docker containerization
- [ ] Kubernetes deployment
