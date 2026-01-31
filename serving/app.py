"""
FastAPI inference service for recommendations.

Usage:
    uvicorn serving.app:app --host 0.0.0.0 --port 8000
"""

import json
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models.ncf import NCF

# Config
MODEL_PATH = "outputs/model.pt"
DATA_DIR = "data/processed"

app = FastAPI(
    title="Recommendation API",
    description="NCF-based recommendation service",
    version="1.0.0",
)

# Global model (loaded once at startup)
model = None
num_items = None
idx_to_item = None


class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = 10


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list[str]


@app.on_event("startup")
def load_model():
    """Load model and mappings on startup."""
    global model, num_items, idx_to_item
    
    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = NCF(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    num_items = checkpoint["num_items"]
    
    # Load ID mappings
    with open(f"{DATA_DIR}/user_to_idx.json") as f:
        user_to_idx = json.load(f)
    with open(f"{DATA_DIR}/item_to_idx.json") as f:
        item_to_idx = json.load(f)
    
    # Reverse mapping: idx -> item_id
    idx_to_item = {v: k for k, v in item_to_idx.items()}
    
    # Store user mapping for lookups
    app.state.user_to_idx = user_to_idx


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """Get top-K recommendations for a user."""
    user_to_idx = app.state.user_to_idx
    
    # Validate user
    if request.user_id not in user_to_idx:
        raise HTTPException(status_code=404, detail=f"User '{request.user_id}' not found")
    
    user_idx = user_to_idx[request.user_id]
    
    # Score all items for this user
    with torch.no_grad():
        user_tensor = torch.full((num_items,), user_idx, dtype=torch.long)
        item_tensor = torch.arange(num_items, dtype=torch.long)
        scores = model(user_tensor, item_tensor).numpy()
    
    # Get top-K item indices
    top_k_indices = scores.argsort()[-request.top_k:][::-1]
    
    # Convert indices to item IDs
    recommendations = [idx_to_item[idx] for idx in top_k_indices]
    
    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
    )
