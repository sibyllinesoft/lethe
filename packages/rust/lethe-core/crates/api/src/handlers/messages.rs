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
use lethe_shared::Message;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

/// Message creation request
#[derive(Debug, Deserialize, Serialize, Validate, Clone)]
pub struct CreateMessageRequest {
    pub session_id: String,

    #[validate(range(min = 0, message = "Turn must be non-negative"))]
    pub turn: i32,

    #[validate(length(min = 1, message = "Role cannot be empty"))]
    pub role: String,

    #[validate(length(
        min = 1,
        max = 10000,
        message = "Text must be between 1 and 10000 characters"
    ))]
    pub text: String,

    pub meta: Option<serde_json::Value>,
}

/// Message update request
#[derive(Debug, Deserialize, Serialize, Validate, Clone)]
pub struct UpdateMessageRequest {
    pub session_id: Option<String>,
    pub turn: Option<i32>,
    pub role: Option<String>,

    #[validate(length(
        min = 1,
        max = 10000,
        message = "Text must be between 1 and 10000 characters"
    ))]
    pub text: Option<String>,

    pub meta: Option<serde_json::Value>,
}

/// Message response
#[derive(Debug, Serialize)]
pub struct MessageResponse {
    pub message: Message,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Messages list query parameters
#[derive(Debug, Deserialize)]
pub struct MessagesQuery {
    pub session_id: Option<String>,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

/// Messages list response
#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub messages: Vec<Message>,
    pub total_count: usize,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Create a new message
pub async fn create_message(
    State(state): State<AppState>,
    Json(request): Json<CreateMessageRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate request
    request.validate().map_err(ApiError::from)?;

    let message = Message {
        id: Uuid::new_v4(),
        session_id: request.session_id,
        turn: request.turn,
        role: request.role,
        text: request.text,
        ts: chrono::Utc::now(),
        meta: request.meta,
    };

    let created_message = state
        .message_repository
        .create_message(&message)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to create message: {}", e)))?;

    let response = MessageResponse {
        message: created_message,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::CREATED, Json(response)))
}

/// Get a message by ID
pub async fn get_message(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<impl IntoResponse> {
    let message = state
        .message_repository
        .get_message(&id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get message: {}", e)))?
        .ok_or_else(|| ApiError::not_found(format!("Message with id {}", id)))?;

    let response = MessageResponse {
        message,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Update a message
pub async fn update_message(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(request): Json<UpdateMessageRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate request
    request.validate().map_err(ApiError::from)?;

    // Get existing message
    let mut existing_message = state
        .message_repository
        .get_message(&id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get message: {}", e)))?
        .ok_or_else(|| ApiError::not_found(format!("Message with id {}", id)))?;

    // Apply updates
    if let Some(session_id) = request.session_id {
        existing_message.session_id = session_id;
    }
    if let Some(turn) = request.turn {
        existing_message.turn = turn;
    }
    if let Some(role) = request.role {
        existing_message.role = role;
    }
    if let Some(text) = request.text {
        existing_message.text = text;
    }
    if let Some(meta) = request.meta {
        existing_message.meta = Some(meta);
    }

    let updated_message = state
        .message_repository
        .update_message(&existing_message)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to update message: {}", e)))?;

    let response = MessageResponse {
        message: updated_message,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Delete a message
pub async fn delete_message(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<impl IntoResponse> {
    let deleted = state
        .message_repository
        .delete_message(&id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to delete message: {}", e)))?;

    if !deleted {
        return Err(ApiError::not_found(format!("Message with id {}", id)));
    }

    Ok((StatusCode::NO_CONTENT, ()))
}

/// List messages
pub async fn list_messages(
    State(state): State<AppState>,
    params: QueryParams<MessagesQuery>,
) -> ApiResult<impl IntoResponse> {
    let MessagesQuery {
        session_id,
        limit,
        offset,
    } = params.0;

    let messages = if let Some(session_id) = session_id {
        state
            .message_repository
            .get_messages_by_session(&session_id, limit)
            .await
            .map_err(|e| ApiError::internal(format!("Failed to get messages by session: {}", e)))?
    } else {
        // For listing all messages, we'd need a different repository method
        // For now, return an error suggesting to provide session_id
        return Err(ApiError::bad_request("session_id parameter is required"));
    };

    let response = MessagesResponse {
        total_count: messages.len(),
        messages,
        limit,
        offset,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Get recent messages for a session
pub async fn get_recent_messages(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    params: QueryParams<serde_json::Value>,
) -> ApiResult<impl IntoResponse> {
    // Parse count parameter
    let count = params
        .0
        .get("count")
        .and_then(|v| v.as_i64())
        .map(|v| v as i32)
        .unwrap_or(10);

    if count < 1 || count > 100 {
        return Err(ApiError::validation("Count must be between 1 and 100"));
    }

    let messages = state
        .message_repository
        .get_recent_messages(&session_id, count)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get recent messages: {}", e)))?;

    let response = MessagesResponse {
        total_count: messages.len(),
        messages,
        limit: Some(count),
        offset: None,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Batch create messages
#[derive(Debug, Deserialize, Serialize, Validate)]
pub struct BatchCreateMessagesRequest {
    #[validate(length(
        min = 1,
        max = 100,
        message = "Must provide between 1 and 100 messages"
    ))]
    pub messages: Vec<CreateMessageRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchCreateMessagesResponse {
    pub messages: Vec<Message>,
    pub created_count: usize,
    pub failed_count: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub async fn batch_create_messages(
    State(state): State<AppState>,
    Json(request): Json<BatchCreateMessagesRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate request
    request.validate().map_err(ApiError::from)?;

    let mut created_messages = Vec::new();
    let mut failed_count = 0;

    let messages = request.messages;

    for msg_request in messages {
        // Validate individual message
        if let Err(e) = msg_request.validate() {
            tracing::warn!(error = %e, "Invalid message in batch request");
            failed_count += 1;
            continue;
        }

        let message = Message {
            id: Uuid::new_v4(),
            session_id: msg_request.session_id,
            turn: msg_request.turn,
            role: msg_request.role,
            text: msg_request.text,
            ts: chrono::Utc::now(),
            meta: msg_request.meta,
        };

        match state.message_repository.create_message(&message).await {
            Ok(created_message) => {
                created_messages.push(created_message);
            }
            Err(e) => {
                tracing::error!(error = %e, "Failed to create message in batch");
                failed_count += 1;
            }
        }
    }

    let response = BatchCreateMessagesResponse {
        created_count: created_messages.len(),
        messages: created_messages,
        failed_count,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::CREATED, Json(response)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_message_request_validation() {
        let valid_request = CreateMessageRequest {
            session_id: "test-session".to_string(),
            turn: 1,
            role: "user".to_string(),
            text: "Hello, world!".to_string(),
            meta: None,
        };
        assert!(valid_request.validate().is_ok());

        let invalid_request = CreateMessageRequest {
            session_id: "test-session".to_string(),
            turn: -1,             // Invalid turn
            role: "".to_string(), // Empty role
            text: "".to_string(), // Empty text
            meta: None,
        };
        assert!(invalid_request.validate().is_err());
    }

    #[test]
    fn test_update_message_request_validation() {
        let valid_request = UpdateMessageRequest {
            session_id: Some("new-session".to_string()),
            turn: Some(2),
            role: Some("assistant".to_string()),
            text: Some("Updated text".to_string()),
            meta: None,
        };
        assert!(valid_request.validate().is_ok());

        let invalid_request = UpdateMessageRequest {
            session_id: None,
            turn: None,
            role: None,
            text: Some("".to_string()), // Empty text
            meta: None,
        };
        assert!(invalid_request.validate().is_err());
    }

    #[test]
    fn test_batch_create_messages_validation() {
        let valid_batch = BatchCreateMessagesRequest {
            messages: vec![CreateMessageRequest {
                session_id: "test-session".to_string(),
                turn: 1,
                role: "user".to_string(),
                text: "Message 1".to_string(),
                meta: None,
            }],
        };
        assert!(valid_batch.validate().is_ok());

        let empty_batch = BatchCreateMessagesRequest { messages: vec![] };
        assert!(empty_batch.validate().is_err());
    }
}
