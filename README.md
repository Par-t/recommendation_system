# Distributed Recommendation System

An MLOps-focused recommendation system demonstrating distributed training, experiment tracking, and Kubernetes-based serving.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the Amazon Reviews dataset (e.g., All_Beauty) and place the JSONL file in `data/`.

### Preprocessing

```bash
python data/preprocess.py \
    --input data/All_Beauty.jsonl \
    --output-dir data/processed \
    --min-interactions 5
```

**What it does:**
- Extracts `user_id`, `item_id`, `timestamp` from raw reviews
- Filters users/items with fewer than 5 interactions
- Creates temporal train/val/test splits (70/15/15)
- Generates integer ID mappings for embeddings

**Outputs** (in `data/processed/`):
- `train.parquet`, `val.parquet`, `test.parquet` — data splits
- `user_to_idx.json`, `item_to_idx.json` — ID mappings
- `stats.json` — dataset statistics
