use axum::{
    extract::{Path, Query as QueryParams, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use lethe_shared::{Session, SessionState};
use serde::{Deserialize, Serialize};
use validator::Validate;
use crate::{error::{ApiError, ApiResult}, state::AppState};

/// Session creation request
#[derive(Debug, Deserialize, Validate)]
pub struct CreateSessionRequest {
    #[validate(length(min = 1, max = 255, message = "Session ID must be between 1 and 255 characters"))]
    pub id: String,
    pub metadata: Option<serde_json::Value>,
}

/// Session update request
#[derive(Debug, Deserialize, Validate)]
pub struct UpdateSessionRequest {
    pub metadata: Option<serde_json::Value>,
}

/// Session response
#[derive(Debug, Serialize)]
pub struct SessionResponse {
    pub session: Session,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Sessions list query parameters
#[derive(Debug, Deserialize)]
pub struct SessionsQuery {
    pub limit: Option<i32>,
    pub offset: Option<i32>,
}

/// Sessions list response
#[derive(Debug, Serialize)]
pub struct SessionsResponse {
    pub sessions: Vec<Session>,
    pub total_count: usize,
    pub limit: Option<i32>,
    pub offset: Option<i32>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Session state response
#[derive(Debug, Serialize)]
pub struct SessionStateResponse {
    pub state: Vec<SessionState>,
    pub session_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Set session state request
#[derive(Debug, Deserialize, Validate)]
pub struct SetSessionStateRequest {
    #[validate(length(min = 1, max = 255, message = "State key must be between 1 and 255 characters"))]
    pub key: String,
    pub value: serde_json::Value,
}

/// Create a new session
pub async fn create_session(
    State(state): State<AppState>,
    Json(request): Json<CreateSessionRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate request
    request.validate().map_err(ApiError::from)?;

    let session = Session {
        id: request.id,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        metadata: request.metadata,
    };

    // TODO: Implement actual session creation when database is available
    #[cfg(feature = "database")]
    let _created_session = state.session_repository
        .create_session(&session)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to create session: {}", e)))?;

    let created_session = session;

    let response = SessionResponse {
        session: created_session,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::CREATED, Json(response)))
}

/// Get a session by ID
pub async fn get_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let session = state.session_repository
        .get_session(&id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get session: {}", e)))?
        .ok_or_else(|| ApiError::not_found(format!("Session with id {}", id)))?;

    let response = SessionResponse {
        session,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Update a session
pub async fn update_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<UpdateSessionRequest>,
) -> ApiResult<impl IntoResponse> {
    // Validate request
    request.validate().map_err(ApiError::from)?;

    // Get existing session
    let mut existing_session = state.session_repository
        .get_session(&id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get session: {}", e)))?
        .ok_or_else(|| ApiError::not_found(format!("Session with id {}", id)))?;

    // Update metadata and timestamp
    existing_session.metadata = request.metadata;
    existing_session.updated_at = chrono::Utc::now();

    let updated_session = state.session_repository
        .update_session(&existing_session)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to update session: {}", e)))?;

    let response = SessionResponse {
        session: updated_session,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Delete a session
pub async fn delete_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let deleted = state.session_repository
        .delete_session(&id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to delete session: {}", e)))?;

    if !deleted {
        return Err(ApiError::not_found(format!("Session with id {}", id)));
    }

    Ok((StatusCode::NO_CONTENT, ()))
}

/// List sessions
pub async fn list_sessions(
    State(state): State<AppState>,
    params: QueryParams<SessionsQuery>,
) -> ApiResult<impl IntoResponse> {
    let SessionsQuery { limit, offset } = params.0;

    let sessions = state.session_repository
        .list_sessions(limit, offset)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to list sessions: {}", e)))?;

    let response = SessionsResponse {
        total_count: sessions.len(),
        sessions,
        limit,
        offset,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Get session state
pub async fn get_session_state(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let session_state = state.session_repository
        .get_all_session_state(&session_id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get session state: {}", e)))?;

    let response = SessionStateResponse {
        state: session_state,
        session_id,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Get specific session state value
pub async fn get_session_state_value(
    State(state): State<AppState>,
    Path((session_id, key)): Path<(String, String)>,
) -> ApiResult<impl IntoResponse> {
    let value = state.session_repository
        .get_session_state(&session_id, &key)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get session state value: {}", e)))?
        .ok_or_else(|| ApiError::not_found(format!("State key '{}' for session '{}'", key, session_id)))?;

    Ok((StatusCode::OK, Json(serde_json::json!({
        "session_id": session_id,
        "key": key,
        "value": value,
        "timestamp": chrono::Utc::now()
    }))))
}

/// Set session state
pub async fn set_session_state(
    State(state): State<AppState>,
    Path((session_id, key)): Path<(String, String)>,
    Json(request): Json<serde_json::Value>,
) -> ApiResult<impl IntoResponse> {
    state.session_repository
        .set_session_state(&session_id, &key, &request)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to set session state: {}", e)))?;

    Ok((StatusCode::OK, Json(serde_json::json!({
        "session_id": session_id,
        "key": key,
        "value": request,
        "timestamp": chrono::Utc::now()
    }))))
}

/// Delete session state value
pub async fn delete_session_state_value(
    State(state): State<AppState>,
    Path((session_id, key)): Path<(String, String)>,
) -> ApiResult<impl IntoResponse> {
    let deleted = state.session_repository
        .delete_session_state(&session_id, &key)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to delete session state: {}", e)))?;

    if !deleted {
        return Err(ApiError::not_found(format!("State key '{}' for session '{}'", key, session_id)));
    }

    Ok((StatusCode::NO_CONTENT, ()))
}

/// Clear all session state
pub async fn clear_session_state(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    state.session_repository
        .clear_session_state(&session_id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to clear session state: {}", e)))?;

    Ok((StatusCode::NO_CONTENT, ()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_session_request_validation() {
        let valid_request = CreateSessionRequest {
            id: "test-session-1".to_string(),
            metadata: Some(serde_json::json!({"user_id": "user123"})),
        };
        assert!(valid_request.validate().is_ok());

        let invalid_request = CreateSessionRequest {
            id: "".to_string(), // Empty ID
            metadata: None,
        };
        assert!(invalid_request.validate().is_err());
    }

    #[test]
    fn test_update_session_request_validation() {
        let valid_request = UpdateSessionRequest {
            metadata: Some(serde_json::json!({"updated": true})),
        };
        assert!(valid_request.validate().is_ok());
    }

    #[test]
    fn test_set_session_state_request_validation() {
        let valid_request = SetSessionStateRequest {
            key: "user_preferences".to_string(),
            value: serde_json::json!({"theme": "dark"}),
        };
        assert!(valid_request.validate().is_ok());

        let invalid_request = SetSessionStateRequest {
            key: "".to_string(), // Empty key
            value: serde_json::json!(null),
        };
        assert!(invalid_request.validate().is_err());
    }
}