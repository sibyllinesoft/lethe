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
use lethe_shared::Chunk;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

/// Chunk creation request
#[derive(Debug, Deserialize, Serialize, Validate, Clone)]
pub struct CreateChunkRequest {
    pub id: String,
    pub message_id: Uuid,
    pub session_id: String,

    #[validate(range(min = 0, message = "Offset start must be non-negative"))]
    pub offset_start: usize,

    #[validate(range(min = 0, message = "Offset end must be non-negative"))]
    pub offset_end: usize,

    #[validate(length(min = 1, message = "Kind cannot be empty"))]
    pub kind: String,

    #[validate(length(
        min = 1,
        max = 50000,
        message = "Text must be between 1 and 50000 characters"
    ))]
    pub text: String,

    #[validate(range(min = 0, message = "Tokens must be non-negative"))]
    pub tokens: i32,
}

/// Chunk response
#[derive(Debug, Serialize)]
pub struct ChunkResponse {
    pub chunk: Chunk,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Chunks list query parameters
#[derive(Debug, Deserialize)]
pub struct ChunksQuery {
    pub session_id: Option<String>,
    pub message_id: Option<Uuid>,
    pub kind: Option<String>,
    pub limit: Option<usize>,
}

/// Chunks list response
#[derive(Debug, Serialize)]
pub struct ChunksResponse {
    pub chunks: Vec<Chunk>,
    pub total_count: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Create a new chunk
pub async fn create_chunk(
    State(state): State<AppState>,
    Json(request): Json<CreateChunkRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate request
    request.validate().map_err(ApiError::from)?;

    // Validate that offset_end > offset_start
    if request.offset_end <= request.offset_start {
        return Err(ApiError::validation(
            "offset_end must be greater than offset_start",
        ));
    }

    let chunk = Chunk {
        id: request.id,
        message_id: request.message_id,
        session_id: request.session_id,
        offset_start: request.offset_start,
        offset_end: request.offset_end,
        kind: request.kind,
        text: request.text,
        tokens: request.tokens,
    };

    let created_chunk = state
        .chunk_repository
        .create_chunk(&chunk)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to create chunk: {}", e)))?;

    let response = ChunkResponse {
        chunk: created_chunk,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::CREATED, Json(response)))
}

/// Get a chunk by ID
pub async fn get_chunk(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let chunk = state
        .chunk_repository
        .get_chunk(&id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get chunk: {}", e)))?
        .ok_or_else(|| ApiError::not_found(format!("Chunk with id {}", id)))?;

    let response = ChunkResponse {
        chunk,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Delete a chunk
pub async fn delete_chunk(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let deleted = state
        .chunk_repository
        .delete_chunk(&id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to delete chunk: {}", e)))?;

    if !deleted {
        return Err(ApiError::not_found(format!("Chunk with id {}", id)));
    }

    Ok((StatusCode::NO_CONTENT, ()))
}

/// List chunks
pub async fn list_chunks(
    State(state): State<AppState>,
    params: QueryParams<ChunksQuery>,
) -> ApiResult<impl IntoResponse> {
    let ChunksQuery {
        session_id,
        message_id,
        kind: _,
        limit: _,
    } = params.0;

    let chunks = if let Some(session_id) = session_id {
        state
            .chunk_repository
            .get_chunks_by_session(&session_id)
            .await
            .map_err(|e| ApiError::internal(format!("Failed to get chunks by session: {}", e)))?
    } else if let Some(message_id) = message_id {
        state
            .chunk_repository
            .get_chunks_by_message(&message_id)
            .await
            .map_err(|e| ApiError::internal(format!("Failed to get chunks by message: {}", e)))?
    } else {
        return Err(ApiError::bad_request(
            "Either session_id or message_id parameter is required",
        ));
    };

    let response = ChunksResponse {
        total_count: chunks.len(),
        chunks,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Get chunks by session
pub async fn get_chunks_by_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let chunks = state
        .chunk_repository
        .get_chunks_by_session(&session_id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get chunks by session: {}", e)))?;

    let response = ChunksResponse {
        total_count: chunks.len(),
        chunks,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Get chunks by message
pub async fn get_chunks_by_message(
    State(state): State<AppState>,
    Path(message_id): Path<Uuid>,
) -> ApiResult<impl IntoResponse> {
    let chunks = state
        .chunk_repository
        .get_chunks_by_message(&message_id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get chunks by message: {}", e)))?;

    let response = ChunksResponse {
        total_count: chunks.len(),
        chunks,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Batch create chunks
#[derive(Debug, Deserialize, Serialize, Validate)]
pub struct BatchCreateChunksRequest {
    #[validate(length(
        min = 1,
        max = 1000,
        message = "Must provide between 1 and 1000 chunks"
    ))]
    pub chunks: Vec<CreateChunkRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchCreateChunksResponse {
    pub chunks: Vec<Chunk>,
    pub created_count: usize,
    pub failed_count: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub async fn batch_create_chunks(
    State(state): State<AppState>,
    Json(request): Json<BatchCreateChunksRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate request
    request.validate().map_err(ApiError::from)?;

    let mut chunks_to_create = Vec::new();
    let mut failed_count = 0;

    let chunk_requests = request.chunks;

    for chunk_request in chunk_requests {
        if let Err(e) = chunk_request.validate() {
            tracing::warn!(error = %e, "Invalid chunk in batch request");
            failed_count += 1;
            continue;
        }

        if chunk_request.offset_end <= chunk_request.offset_start {
            tracing::warn!("Invalid offset range in batch chunk request");
            failed_count += 1;
            continue;
        }

        let chunk = Chunk {
            id: chunk_request.id,
            message_id: chunk_request.message_id,
            session_id: chunk_request.session_id,
            offset_start: chunk_request.offset_start,
            offset_end: chunk_request.offset_end,
            kind: chunk_request.kind,
            text: chunk_request.text,
            tokens: chunk_request.tokens,
        };

        chunks_to_create.push(chunk);
    }

    // Batch create chunks
    let created_chunks = if !chunks_to_create.is_empty() {
        state
            .chunk_repository
            .batch_create_chunks(&chunks_to_create)
            .await
            .map_err(|e| ApiError::internal(format!("Failed to batch create chunks: {}", e)))?
    } else {
        Vec::new()
    };

    let response = BatchCreateChunksResponse {
        created_count: created_chunks.len(),
        chunks: created_chunks,
        failed_count,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::CREATED, Json(response)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_chunk_request_validation() {
        let valid_request = CreateChunkRequest {
            id: "chunk-1".to_string(),
            message_id: Uuid::new_v4(),
            session_id: "session-1".to_string(),
            offset_start: 0,
            offset_end: 100,
            kind: "text".to_string(),
            text: "This is a chunk of text.".to_string(),
            tokens: 10,
        };
        assert!(valid_request.validate().is_ok());

        let invalid_request = CreateChunkRequest {
            id: "chunk-1".to_string(),
            message_id: Uuid::new_v4(),
            session_id: "session-1".to_string(),
            offset_start: 100,
            offset_end: 50,       // Invalid: end < start
            kind: "".to_string(), // Empty kind
            text: "".to_string(), // Empty text
            tokens: -1,           // Negative tokens
        };
        assert!(invalid_request.validate().is_err());
    }

    #[test]
    fn test_batch_create_chunks_validation() {
        let valid_batch = BatchCreateChunksRequest {
            chunks: vec![CreateChunkRequest {
                id: "chunk-1".to_string(),
                message_id: Uuid::new_v4(),
                session_id: "session-1".to_string(),
                offset_start: 0,
                offset_end: 100,
                kind: "text".to_string(),
                text: "Chunk 1".to_string(),
                tokens: 5,
            }],
        };
        assert!(valid_batch.validate().is_ok());

        let empty_batch = BatchCreateChunksRequest { chunks: vec![] };
        assert!(empty_batch.validate().is_err());
    }
}
