use axum::{
    extract::{Path, Query as QueryParams, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use lethe_infrastructure::{Session as InfraSession, SessionState as InfraSessionState};
use lethe_shared::Session;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{
    error::{ApiError, ApiResult},
    state::AppState,
};

/// Session creation request
#[derive(Debug, Deserialize, Validate)]
pub struct CreateSessionRequest {
    #[validate(length(
        min = 1,
        max = 255,
        message = "Session ID must be between 1 and 255 characters"
    ))]
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

/// Flattened representation of session state key/value pairs
#[derive(Debug, Serialize)]
pub struct SessionStateRecord {
    pub session_id: String,
    pub key: String,
    pub value: serde_json::Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Session state response
#[derive(Debug, Serialize)]
pub struct SessionStateResponse {
    pub state: Vec<SessionStateRecord>,
    pub session_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Convert infrastructure session into shared session representation
fn map_session(session: InfraSession) -> Session {
    Session {
        id: session.id,
        created_at: session.created_at,
        updated_at: session.updated_at,
        metadata: session.metadata,
    }
}

/// Convert infrastructure session state into API record
fn map_session_state(state: InfraSessionState) -> SessionStateRecord {
    SessionStateRecord {
        session_id: state.session_id,
        key: state.state_key,
        value: state.state_value,
        created_at: state.created_at,
        updated_at: state.updated_at,
    }
}

/// Create a new session
pub async fn create_session(
    State(state): State<AppState>,
    Json(request): Json<CreateSessionRequest>,
) -> ApiResult<impl IntoResponse> {
    request.validate().map_err(ApiError::from)?;

    let now = chrono::Utc::now();
    let session = InfraSession {
        id: request.id,
        created_at: now,
        updated_at: now,
        metadata: request.metadata,
    };

    let created_session = state
        .session_repository
        .create_session(&session)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to create session: {}", e)))?;

    let response = SessionResponse {
        session: map_session(created_session),
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::CREATED, Json(response)))
}

/// Get a session by ID
pub async fn get_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let session = state
        .session_repository
        .get_session(&id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get session: {}", e)))?
        .ok_or_else(|| ApiError::not_found(format!("Session with id {}", id)))?;

    let response = SessionResponse {
        session: map_session(session),
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
    request.validate().map_err(ApiError::from)?;

    let mut existing_session = state
        .session_repository
        .get_session(&id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get session: {}", e)))?
        .ok_or_else(|| ApiError::not_found(format!("Session with id {}", id)))?;

    existing_session.metadata = request.metadata;
    existing_session.updated_at = chrono::Utc::now();

    let updated_session = state
        .session_repository
        .update_session(&existing_session)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to update session: {}", e)))?;

    let response = SessionResponse {
        session: map_session(updated_session),
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Delete a session
pub async fn delete_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let deleted = state
        .session_repository
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

    let sessions = state
        .session_repository
        .list_sessions(limit, offset)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to list sessions: {}", e)))?;

    let response = SessionsResponse {
        total_count: sessions.len(),
        sessions: sessions.into_iter().map(map_session).collect(),
        limit,
        offset,
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Get all session state as key/value pairs
pub async fn get_session_state(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    let session_state = state
        .session_repository
        .get_all_session_state(&session_id)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get session state: {}", e)))?;

    let response = SessionStateResponse {
        session_id: session_id.clone(),
        state: session_state.into_iter().map(map_session_state).collect(),
        timestamp: chrono::Utc::now(),
    };

    Ok((StatusCode::OK, Json(response)))
}

/// Get specific session state value
pub async fn get_session_state_value(
    State(state): State<AppState>,
    Path((session_id, key)): Path<(String, String)>,
) -> ApiResult<impl IntoResponse> {
    let value = state
        .session_repository
        .get_session_state(&session_id, &key)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to get session state value: {}", e)))?
        .ok_or_else(|| {
            ApiError::not_found(format!("State key '{}' for session '{}'", key, session_id))
        })?;

    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "session_id": session_id,
            "key": key,
            "value": value,
            "timestamp": chrono::Utc::now()
        })),
    ))
}

/// Set session state value
pub async fn set_session_state(
    State(state): State<AppState>,
    Path((session_id, key)): Path<(String, String)>,
    Json(value): Json<serde_json::Value>,
) -> ApiResult<impl IntoResponse> {
    state
        .session_repository
        .set_session_state(&session_id, &key, &value)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to set session state: {}", e)))?;

    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "session_id": session_id,
            "key": key,
            "value": value,
            "timestamp": chrono::Utc::now()
        })),
    ))
}

/// Delete a specific session state key
pub async fn delete_session_state_value(
    State(state): State<AppState>,
    Path((session_id, key)): Path<(String, String)>,
) -> ApiResult<impl IntoResponse> {
    let deleted = state
        .session_repository
        .delete_session_state(&session_id, &key)
        .await
        .map_err(|e| ApiError::internal(format!("Failed to delete session state: {}", e)))?;

    if !deleted {
        return Err(ApiError::not_found(format!(
            "State key '{}' for session '{}'",
            key, session_id
        )));
    }

    Ok((StatusCode::NO_CONTENT, ()))
}

/// Clear all state for a session
pub async fn clear_session_state(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> ApiResult<impl IntoResponse> {
    state
        .session_repository
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
            id: "".to_string(),
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
}
