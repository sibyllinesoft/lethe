use crate::{
    error::{ApiError, ApiResult},
    state::AppState,
};
use axum::{
    extract::{Path, Query as QueryParams, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use lethe_shared::EmbeddingVector;
use serde::{Deserialize, Serialize};
use validator::Validate;

/// Embedding creation request
#[derive(Debug, Deserialize, Serialize, Validate, Clone)]
pub struct CreateEmbeddingRequest {
    #[validate(length(min = 1, message = "Chunk ID cannot be empty"))]
    pub chunk_id: String,

    #[validate(length(
        min = 1,
        max = 10000,
        message = "Text must be between 1 and 10000 characters"
    ))]
    pub text: String,
}

/// Embedding response
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub chunk_id: String,
    pub embedding: EmbeddingVector,
    pub dimension: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Embeddings list query parameters
#[derive(Debug, Deserialize)]
pub struct EmbeddingsQuery {
    pub session_id: Option<String>,
    pub limit: Option<usize>,
}

/// Embeddings list response
#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub embeddings: Vec<(String, EmbeddingVector)>,
    pub total_count: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Similarity search request
#[derive(Debug, Deserialize, Validate)]
pub struct SimilaritySearchRequest {
    #[validate(length(
        min = 1,
        max = 10000,
        message = "Query text must be between 1 and 10000 characters"
    ))]
    pub query: String,

    #[validate(range(min = 1, max = 100, message = "k must be between 1 and 100"))]
    pub k: Option<i32>,
}

/// Similarity search response
#[derive(Debug, Serialize)]
pub struct SimilaritySearchResponse {
    pub results: Vec<SimilarityResult>,
    pub query: String,
    pub k: i32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Individual similarity search result
#[derive(Debug, Serialize)]
pub struct SimilarityResult {
    pub chunk_id: String,
    pub similarity_score: f32,
}

/// Create embedding for a chunk
pub async fn create_embedding(
    State(state): State<AppState>,
    Json(request): Json<CreateEmbeddingRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate request
    request.validate().map_err(ApiError::from)?;

    // Generate embedding using the embedding service
    let embeddings = state
        .embedding_service
        .embed(&[request.text.clone()])
        .await
        .map_err(|e| ApiError::internal(format!("Failed to generate embedding: {}", e)))?;

    let embedding = embeddings
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::internal("Embedding service returned no vectors"))?;

    // Store embedding in repository
    state
        .embedding_repository
        .create_embedding(&request.chunk_id, &embedding)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to store embedding: {}", e)))?;

    let response = EmbeddingResponse {
        chunk_id: request.chunk_id,
        embedding: embedding.clone(),
        dimension: embedding.dimension,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::CREATED, Json(response)))
}

/// Get embedding for a chunk
pub async fn get_embedding(
    State(state): State<AppState>,
    Path(chunk_id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let embedding = state
        .embedding_repository
        .get_embedding(&chunk_id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get embedding: {}", e)))?
        .ok_or_else(|| ApiError::not_found(format!("Embedding for chunk {}", chunk_id)))?;

    let response = EmbeddingResponse {
        chunk_id: chunk_id.clone(),
        embedding: embedding.clone(),
        dimension: embedding.dimension,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Delete embedding for a chunk
pub async fn delete_embedding(
    State(state): State<AppState>,
    Path(chunk_id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let deleted = state
        .embedding_repository
        .delete_embedding(&chunk_id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to delete embedding: {}", e)))?;

    if !deleted {
        return Err(ApiError::not_found(format!(
            "Embedding for chunk {}",
            chunk_id
        )));
    }

    Ok((StatusCode::NO_CONTENT, ()))
}

/// List embeddings
pub async fn list_embeddings(
    State(state): State<AppState>,
    params: QueryParams<EmbeddingsQuery>,
) -> ApiResult<impl IntoResponse> {
    let EmbeddingsQuery {
        session_id,
        limit: _,
    } = params.0;

    if let Some(session_id) = session_id {
        let embeddings = state
            .embedding_repository
            .get_embeddings_by_session(&session_id)
            .await
            .map_err(|e| {
                ApiError::internal(format!("Failed to get embeddings by session: {}", e))
            })?;

        let response = EmbeddingsResponse {
            total_count: embeddings.len(),
            embeddings,
            timestamp: chrono::Utc::now(),
        };

        Ok((StatusCode::OK, Json(response)))
    } else {
        Err(ApiError::bad_request("session_id parameter is required"))
    }
}

/// Get embeddings by session
pub async fn get_embeddings_by_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let embeddings = state
        .embedding_repository
        .get_embeddings_by_session(&session_id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get embeddings by session: {}", e)))?;

    let response = EmbeddingsResponse {
        total_count: embeddings.len(),
        embeddings,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Similarity search using text query
pub async fn similarity_search(
    State(state): State<AppState>,
    Json(request): Json<SimilaritySearchRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate request
    request.validate().map_err(ApiError::from)?;

    let k = request.k.unwrap_or(10);

    // Generate query embedding
    let query_vectors = state
        .embedding_service
        .embed(&[request.query.clone()])
        .await
        .map_err(|e| ApiError::internal(format!("Failed to generate query embedding: {}", e)))?;

    let query_embedding = query_vectors
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::internal("Embedding service returned no vectors"))?;

    // Perform similarity search
    let similar_embeddings = state
        .embedding_repository
        .search_similar_embeddings(&query_embedding, k)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to perform similarity search: {}", e)))?;

    let results = similar_embeddings
        .into_iter()
        .map(|(chunk_id, score)| SimilarityResult {
            chunk_id,
            similarity_score: score,
        })
        .collect();

    let response = SimilaritySearchResponse {
        results,
        query: request.query,
        k,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Batch create embeddings
#[derive(Debug, Deserialize, Serialize, Validate)]
pub struct BatchCreateEmbeddingsRequest {
    #[validate(length(
        min = 1,
        max = 100,
        message = "Must provide between 1 and 100 embeddings"
    ))]
    pub embeddings: Vec<CreateEmbeddingRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchCreateEmbeddingsResponse {
    pub embeddings: Vec<EmbeddingResponse>,
    pub created_count: usize,
    pub failed_count: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub async fn batch_create_embeddings(
    State(state): State<AppState>,
    Json(request): Json<BatchCreateEmbeddingsRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate request
    request.validate().map_err(ApiError::from)?;

    let mut embedding_responses = Vec::new();
    let mut embeddings_to_store: Vec<(String, EmbeddingVector)> = Vec::new();
    let mut failed_count = 0;

    let embedding_requests = request.embeddings;

    for embedding_request in embedding_requests {
        if let Err(e) = embedding_request.validate() {
            tracing::warn!(error = %e, "Invalid embedding request in batch");
            failed_count += 1;
            continue;
        }

        match state
            .embedding_service
            .embed(&[embedding_request.text.clone()])
            .await
        {
            Ok(vectors) => {
                if let Some(embedding) = vectors.into_iter().next() {
                    embeddings_to_store
                        .push((embedding_request.chunk_id.clone(), embedding.clone()));
                    embedding_responses.push(EmbeddingResponse {
                        chunk_id: embedding_request.chunk_id,
                        embedding: embedding.clone(),
                        dimension: embedding.dimension,
                        timestamp: chrono::Utc::now(),
                    });
                } else {
                    tracing::warn!(chunk_id = %embedding_request.chunk_id, "Embedding service returned no vectors in batch");
                    failed_count += 1;
                }
            }
            Err(e) => {
                tracing::error!(error = %e, chunk_id = %embedding_request.chunk_id, "Failed to generate embedding in batch");
                failed_count += 1;
            }
        }
    }

    if !embeddings_to_store.is_empty() {
        if let Err(e) = state
            .embedding_repository
            .batch_create_embeddings(&embeddings_to_store)
            .await
        {
            tracing::error!(error = %e, "Failed to batch store embeddings");
            return Err(ApiError::internal("Failed to store embeddings"));
        }
    }

    let response = BatchCreateEmbeddingsResponse {
        created_count: embedding_responses.len(),
        embeddings: embedding_responses,
        failed_count,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::CREATED, Json(response)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_embedding_request_validation() {
        let valid_request = CreateEmbeddingRequest {
            chunk_id: "chunk-1".to_string(),
            text: "This is some text to embed.".to_string(),
        };
        assert!(valid_request.validate().is_ok());

        let invalid_request = CreateEmbeddingRequest {
            chunk_id: "".to_string(), // Empty chunk ID
            text: "".to_string(),     // Empty text
        };
        assert!(invalid_request.validate().is_err());
    }

    #[test]
    fn test_similarity_search_request_validation() {
        let valid_request = SimilaritySearchRequest {
            query: "Find similar documents".to_string(),
            k: Some(10),
        };
        assert!(valid_request.validate().is_ok());

        let invalid_request = SimilaritySearchRequest {
            query: "".to_string(), // Empty query
            k: Some(0),            // Invalid k
        };
        assert!(invalid_request.validate().is_err());
    }

    #[test]
    fn test_batch_create_embeddings_validation() {
        let valid_batch = BatchCreateEmbeddingsRequest {
            embeddings: vec![CreateEmbeddingRequest {
                chunk_id: "chunk-1".to_string(),
                text: "Text 1".to_string(),
            }],
        };
        assert!(valid_batch.validate().is_ok());

        let empty_batch = BatchCreateEmbeddingsRequest { embeddings: vec![] };
        assert!(empty_batch.validate().is_err());
    }
}
