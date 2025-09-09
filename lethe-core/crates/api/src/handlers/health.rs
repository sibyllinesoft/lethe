use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use crate::{error::ApiResult, state::AppState};

/// Health check endpoint
pub async fn health_check(State(state): State<AppState>) -> ApiResult<impl IntoResponse> {
    let health_status = state.health_check().await?;
    Ok((StatusCode::OK, Json(health_status)))
}

/// Readiness check endpoint
pub async fn readiness_check(State(state): State<AppState>) -> ApiResult<impl IntoResponse> {
    // Check if all critical services are ready
    let health_status = state.health_check().await?;
    
    let is_ready = health_status.components
        .iter()
        .all(|component| matches!(component.status, crate::state::ServiceStatus::Healthy));

    if is_ready {
        Ok((StatusCode::OK, Json(serde_json::json!({
            "status": "ready",
            "timestamp": chrono::Utc::now()
        }))))
    } else {
        Ok((StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
            "status": "not_ready",
            "health": health_status,
            "timestamp": chrono::Utc::now()
        }))))
    }
}

/// Liveness check endpoint
pub async fn liveness_check() -> impl IntoResponse {
    // Simple liveness check - if this endpoint responds, the service is alive
    (StatusCode::OK, Json(serde_json::json!({
        "status": "alive",
        "timestamp": chrono::Utc::now()
    })))
}

/// Application statistics endpoint
pub async fn app_stats(State(state): State<AppState>) -> ApiResult<impl IntoResponse> {
    let stats = state.get_stats().await?;
    Ok((StatusCode::OK, Json(stats)))
}

/// Version information endpoint
pub async fn version_info() -> impl IntoResponse {
    let version_info = serde_json::json!({
        "name": env!("CARGO_PKG_NAME"),
        "version": env!("CARGO_PKG_VERSION"),
        "description": env!("CARGO_PKG_DESCRIPTION"),
        "authors": env!("CARGO_PKG_AUTHORS").split(':').collect::<Vec<_>>(),
        "repository": env!("CARGO_PKG_REPOSITORY"),
        "build_timestamp": chrono::Utc::now(),
        "rust_version": env!("CARGO_PKG_RUST_VERSION")
    });

    (StatusCode::OK, Json(version_info))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_liveness_check() {
        let response = liveness_check().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_version_info() {
        let response = version_info().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }
}