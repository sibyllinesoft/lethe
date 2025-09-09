use axum::{
    extract::{Path, Query as QueryParams, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use lethe_domain::{EnhancedQueryOptions, EnhancedQueryResult};
use serde::{Deserialize, Serialize};
use validator::Validate;
use crate::{error::{ApiError, ApiResult}, state::AppState};
use std::collections::HashMap;

/// Query request payload
#[derive(Debug, Deserialize, Validate)]
pub struct QueryRequest {
    #[validate(length(min = 1, max = 1000, message = "Query must be between 1 and 1000 characters"))]
    pub query: String,
    
    pub session_id: Option<String>,
    
    #[validate(range(min = 1, max = 100, message = "k must be between 1 and 100"))]
    pub k: Option<usize>,
    
    pub include_metadata: Option<bool>,
    pub enable_hyde: Option<bool>,
    pub override_strategy: Option<String>,
    pub context: Option<HashMap<String, serde_json::Value>>,
}

/// Query response
#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub result: EnhancedQueryResult,
    pub request_id: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Query parameters for GET requests
#[derive(Debug, Deserialize)]
pub struct QueryQuery {
    pub q: String,
    pub session_id: Option<String>,
    pub k: Option<usize>,
    pub include_metadata: Option<bool>,
}

/// Enhanced query endpoint (POST)
pub async fn query_enhanced(
    State(state): State<AppState>,
    Json(request): Json<QueryRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate request
    request.validate().map_err(ApiError::from)?;

    // Convert request to domain options
    let options = EnhancedQueryOptions {
        session_id: request.session_id.unwrap_or_else(|| "default".to_string()),
        k: request.k.unwrap_or(10),
        include_metadata: request.include_metadata.unwrap_or(true),
        enable_hyde: request.enable_hyde,
        override_strategy: request.override_strategy.and_then(|s| {
            match s.as_str() {
                "bm25" => Some(lethe_domain::RetrievalStrategy::BM25Only),
                "vector" => Some(lethe_domain::RetrievalStrategy::VectorOnly),
                "hybrid" => Some(lethe_domain::RetrievalStrategy::Hybrid),
                "hyde" => Some(lethe_domain::RetrievalStrategy::HydeEnhanced),
                "multi_step" => Some(lethe_domain::RetrievalStrategy::MultiStep),
                "adaptive" => Some(lethe_domain::RetrievalStrategy::Adaptive),
                _ => None,
            }
        }),
        context: request.context,
    };

    // Process query through pipeline
    let result = state.query_pipeline
        .process_query(&request.query, &options)
        .await
        .map_err(|e| ApiError::internal(format!("Query processing failed: {}", e)))?;

    let response = QueryResponse {
        result,
        request_id: None, // TODO: Extract from request headers
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Simple query endpoint (GET)
pub async fn query_simple(
    State(state): State<AppState>,
    params: QueryParams<QueryQuery>,
) -> ApiResult<impl IntoResponse> {
    let QueryQuery { q, session_id, k, include_metadata } = params.0;

    // Validate query
    if q.is_empty() || q.len() > 1000 {
        return Err(ApiError::validation("Query must be between 1 and 1000 characters"));
    }

    let options = EnhancedQueryOptions {
        session_id: session_id.unwrap_or_else(|| "default".to_string()),
        k: k.unwrap_or(10),
        include_metadata: include_metadata.unwrap_or(true),
        enable_hyde: None,
        override_strategy: None,
        context: None,
    };

    // Process query through pipeline
    let result = state.query_pipeline
        .process_query(&q, &options)
        .await
        .map_err(|e| ApiError::internal(format!("Query processing failed: {}", e)))?;

    let response = QueryResponse {
        result,
        request_id: None,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Query by session endpoint
pub async fn query_by_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(request): Json<QueryRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate request
    request.validate().map_err(ApiError::from)?;

    // Use session from path, override request session_id
    let options = EnhancedQueryOptions {
        session_id: session_id.clone(),
        k: request.k.unwrap_or(10),
        include_metadata: request.include_metadata.unwrap_or(true),
        enable_hyde: request.enable_hyde,
        override_strategy: request.override_strategy.and_then(|s| {
            match s.as_str() {
                "bm25" => Some(lethe_domain::RetrievalStrategy::BM25Only),
                "vector" => Some(lethe_domain::RetrievalStrategy::VectorOnly),
                "hybrid" => Some(lethe_domain::RetrievalStrategy::Hybrid),
                "hyde" => Some(lethe_domain::RetrievalStrategy::HydeEnhanced),
                "multi_step" => Some(lethe_domain::RetrievalStrategy::MultiStep),
                "adaptive" => Some(lethe_domain::RetrievalStrategy::Adaptive),
                _ => None,
            }
        }),
        context: request.context,
    };

    // Process query through pipeline
    let result = state.query_pipeline
        .process_query(&request.query, &options)
        .await
        .map_err(|e| ApiError::internal(format!("Query processing failed: {}", e)))?;

    let response = QueryResponse {
        result,
        request_id: None,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Batch query endpoint
#[derive(Debug, Deserialize, Validate)]
pub struct BatchQueryRequest {
    #[validate(length(min = 1, max = 10, message = "Must provide between 1 and 10 queries"))]
    pub queries: Vec<QueryRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchQueryResponse {
    pub results: Vec<QueryResponse>,
    pub total_queries: usize,
    pub successful: usize,
    pub failed: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub async fn batch_query(
    State(state): State<AppState>,
    Json(request): Json<BatchQueryRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate request
    request.validate().map_err(ApiError::from)?;

    let mut results = Vec::new();
    let mut successful = 0;
    let mut failed = 0;

    for query_request in request.queries {
        // Validate individual query
        if let Err(e) = query_request.validate() {
            tracing::warn!(error = %e, "Invalid query in batch request");
            failed += 1;
            continue;
        }

        let options = EnhancedQueryOptions {
            session_id: query_request.session_id.unwrap_or_else(|| "default".to_string()),
            k: query_request.k.unwrap_or(10),
            include_metadata: query_request.include_metadata.unwrap_or(true),
            enable_hyde: query_request.enable_hyde,
            override_strategy: query_request.override_strategy.and_then(|s| {
                match s.as_str() {
                    "bm25" => Some(lethe_domain::RetrievalStrategy::BM25Only),
                    "vector" => Some(lethe_domain::RetrievalStrategy::VectorOnly),
                    "hybrid" => Some(lethe_domain::RetrievalStrategy::Hybrid),
                    "hyde" => Some(lethe_domain::RetrievalStrategy::HydeEnhanced),
                    "multi_step" => Some(lethe_domain::RetrievalStrategy::MultiStep),
                    "adaptive" => Some(lethe_domain::RetrievalStrategy::Adaptive),
                    _ => None,
                }
            }),
            context: query_request.context,
        };

        match state.query_pipeline.process_query(&query_request.query, &options).await {
            Ok(result) => {
                results.push(QueryResponse {
                    result,
                    request_id: None,
                    timestamp: chrono::Utc::now(),
                });
                successful += 1;
            }
            Err(e) => {
                tracing::error!(error = %e, query = %query_request.query, "Query processing failed in batch");
                failed += 1;
            }
        }
    }

    let response = BatchQueryResponse {
        results,
        total_queries: request.queries.len(),
        successful,
        failed,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_request_validation() {
        let valid_request = QueryRequest {
            query: "What is machine learning?".to_string(),
            session_id: Some("test".to_string()),
            k: Some(5),
            include_metadata: Some(true),
            enable_hyde: Some(false),
            override_strategy: Some("hybrid".to_string()),
            context: None,
        };
        assert!(valid_request.validate().is_ok());

        let invalid_request = QueryRequest {
            query: "".to_string(), // Empty query
            session_id: None,
            k: Some(0), // Invalid k
            include_metadata: None,
            enable_hyde: None,
            override_strategy: None,
            context: None,
        };
        assert!(invalid_request.validate().is_err());
    }

    #[test]
    fn test_batch_query_request_validation() {
        let valid_batch = BatchQueryRequest {
            queries: vec![
                QueryRequest {
                    query: "Query 1".to_string(),
                    session_id: None,
                    k: Some(5),
                    include_metadata: None,
                    enable_hyde: None,
                    override_strategy: None,
                    context: None,
                },
            ],
        };
        assert!(valid_batch.validate().is_ok());

        let empty_batch = BatchQueryRequest {
            queries: vec![],
        };
        assert!(empty_batch.validate().is_err());
    }
}