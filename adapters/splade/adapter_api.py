#!/usr/bin/env python3
"""
SPLADE v2 Adapter HTTP API
Exposes the standard adapter interface via HTTP
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import time
from splade_adapter import SPLADEAdapter

app = FastAPI(title="SPLADE v2 Adapter", version="1.0.0")

# Initialize adapter
adapter = SPLADEAdapter()

class SearchRequest(BaseModel):
    query: str
    budget: float
    k: int
    index_id: str

class SearchResponse(BaseModel):
    candidates: List[Dict[str, Any]]
    timings: Dict[str, float]
    metadata: Dict[str, Any]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "adapter": "splade_v2"}

@app.post("/build_index")
async def build_index(dataset: str):
    """Build search index for dataset"""
    try:
        start_time = time.time()
        index_id = adapter.build_index(dataset)
        elapsed = (time.time() - start_time) * 1000
        return {
            "index_id": index_id,
            "build_time_ms": elapsed,
            "dataset": dataset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search with budget constraint"""
    try:
        result = adapter.search(request.query, request.budget, request.k, request.index_id)
        return SearchResponse(
            candidates=result.candidates,
            timings=result.timings,
            metadata=result.metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system_info")
async def get_system_info():
    """Get system metadata"""
    return adapter.get_system_info()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)