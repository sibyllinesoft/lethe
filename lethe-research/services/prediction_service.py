#!/usr/bin/env python3
"""
FastAPI-based ML Prediction Service for Lethe
Workstream A, Phase 2.1: Replace subprocess bridge with HTTP service

This service provides ML predictions via REST API, replacing the subprocess-based
approach with a more robust, scalable, and testable HTTP interface.

Performance target: <50ms for prediction requests
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import the existing ultra-fast predictor
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "experiments"))
    from iter3_fast_prediction import create_ultra_fast_predictor, UltraFastPredictor
except ImportError as e:
    logging.error(f"Failed to import prediction module: {e}")
    UltraFastPredictor = None


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for ML predictions."""
    query: str = Field(..., description="The search query to analyze")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for prediction")


class QueryFeatures(BaseModel):
    """Extracted features from the query."""
    query_len: int
    has_code_symbol: bool
    has_error_token: bool
    has_path_or_file: bool
    has_numeric_id: bool
    bm25_top1: float
    ann_top1: float
    overlap_ratio: float
    hyde_k: int
    word_count: int
    char_count: int
    complexity_score: float


class MLPrediction(BaseModel):
    """ML prediction response."""
    alpha: float = Field(..., description="Fusion parameter alpha")
    beta: float = Field(..., description="Fusion parameter beta")
    plan: str = Field(..., description="Retrieval plan: explore, verify, or exploit")
    prediction_time_ms: float = Field(..., description="Time taken for prediction in milliseconds")
    model_loaded: bool = Field(..., description="Whether models were successfully loaded")
    features: Optional[QueryFeatures] = Field(None, description="Extracted query features")
    error: Optional[str] = Field(None, description="Error message if prediction failed")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: bool
    uptime_seconds: float


# Global predictor instance
predictor: Optional[UltraFastPredictor] = None
service_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: load models on startup."""
    global predictor
    
    logging.info("Starting ML Prediction Service...")
    
    if UltraFastPredictor is None:
        logging.error("UltraFastPredictor not available - fallback mode only")
        predictor = None
    else:
        try:
            # Initialize predictor with models directory
            models_dir = Path(__file__).parent.parent / "models"
            predictor = create_ultra_fast_predictor(str(models_dir))
            
            # Test prediction to ensure models load
            start_time = time.time()
            test_result = predictor.predict_all("test query", {})
            load_time = (time.time() - start_time) * 1000
            
            if test_result.get('model_loaded', False):
                logging.info(f"Models loaded successfully in {load_time:.1f}ms")
            else:
                logging.warning("Models failed to load, using fallback mode")
                
        except Exception as e:
            logging.error(f"Failed to initialize predictor: {e}")
            predictor = None
    
    yield
    
    logging.info("Shutting down ML Prediction Service...")


# Create FastAPI app
app = FastAPI(
    title="Lethe ML Prediction Service",
    description="High-performance ML predictions for Lethe retrieval system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=predictor is not None,
        uptime_seconds=time.time() - service_start_time
    )


@app.post("/predict", response_model=MLPrediction)
async def predict_parameters(request: PredictionRequest, req: Request = None):
    """
    Predict optimal retrieval parameters for a query.
    
    This endpoint replaces the subprocess-based prediction system with
    a direct HTTP API call, providing better error handling, monitoring,
    and performance characteristics.
    """
    start_time = time.time()
    
    try:
        if predictor is None:
            # Fallback mode - use heuristic defaults
            logger.warning("Using fallback prediction (no models loaded)")
            prediction = MLPrediction(
                alpha=0.7,
                beta=0.5,
                plan="exploit",
                prediction_time_ms=(time.time() - start_time) * 1000,
                model_loaded=False,
                error="Models not available, using fallbacks"
            )
            return prediction
        
        # Make prediction using loaded models
        result = predictor.predict_all(request.query, request.context)
        prediction_time = (time.time() - start_time) * 1000
        
        # Convert features if present
        features = None
        if 'features' in result and result['features']:
            features = QueryFeatures(**result['features'])
        
        prediction = MLPrediction(
            alpha=result.get('alpha', 0.7),
            beta=result.get('beta', 0.5),
            plan=result.get('plan', 'exploit'),
            prediction_time_ms=prediction_time,
            model_loaded=result.get('model_loaded', True),
            features=features,
            error=result.get('error')
        )
        
        # Log performance metrics
        if prediction_time > 100:
            logger.warning(f"Slow prediction: {prediction_time:.1f}ms for query: {request.query[:50]}...")
        else:
            logger.debug(f"Prediction completed in {prediction_time:.1f}ms")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        
        # Return fallback prediction with error information
        return MLPrediction(
            alpha=0.7,
            beta=0.5,
            plan="exploit",
            prediction_time_ms=(time.time() - start_time) * 1000,
            model_loaded=False,
            error=str(e)
        )


@app.get("/models/info")
async def models_info():
    """Get information about loaded models."""
    if predictor is None:
        return {"models_loaded": False, "error": "No models available"}
    
    try:
        # Get model information from predictor
        info = {
            "models_loaded": True,
            "models_dir": str(predictor.models_dir) if hasattr(predictor, 'models_dir') else "unknown",
            "load_time_ms": getattr(predictor, '_load_time_ms', 0),
            "feature_weights": getattr(predictor, 'feature_weights', {}),
            "plan_rules": getattr(predictor, 'plan_rules', {})
        }
        return info
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return {"models_loaded": True, "error": str(e)}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests for monitoring."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.1f}ms"
    )
    
    return response


def main():
    """Main entry point for running the service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lethe ML Prediction Service")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    logger.info(f"Starting Lethe ML Prediction Service on {args.host}:{args.port}")
    
    # Run the service
    uvicorn.run(
        "prediction_service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=False  # Disable reload in production
    )


if __name__ == "__main__":
    main()