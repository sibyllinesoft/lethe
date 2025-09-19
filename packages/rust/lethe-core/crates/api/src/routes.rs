use crate::{handlers::*, middleware::*, state::AppState};
use axum::{
    middleware,
    routing::{get, post},
    Router,
};

/// Create the main application router with all routes
pub fn create_router(state: AppState) -> Router {
    let security = state.security.clone();

    Router::new()
        // Health and monitoring routes
        .route("/health", get(health_check))
        .route("/health/ready", get(readiness_check))
        .route("/health/live", get(liveness_check))
        .route("/stats", get(app_stats))
        .route("/version", get(version_info))
        // Query routes - core functionality
        .route("/query", post(query_enhanced))
        .route("/query", get(query_simple))
        .route("/query/batch", post(batch_query))
        .route("/sessions/:session_id/query", post(query_by_session))
        // Middleware test endpoint
        .route("/middleware/health", get(middleware_health_check))
        // Apply middleware layers
        .layer(middleware::from_fn(security_headers_middleware))
        .layer(middleware::from_fn(error_handling_middleware))
        .layer(middleware::from_fn(timing_middleware))
        .layer(middleware::from_fn(request_id_middleware))
        .layer(middleware::from_fn_with_state(
            security.clone(),
            rate_limit_middleware,
        ))
        .layer(middleware::from_fn_with_state(security, auth_middleware))
        .layer(create_cors_layer())
        // Add application state
        .with_state(state)
}

/// Create the complete application with all routes
pub fn create_app(state: AppState) -> Router {
    Router::new()
        .nest("/api/v1", create_router(state))
        .fallback(not_found_handler)
}

/// 404 handler
async fn not_found_handler() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({
        "error": "not_found",
        "message": "The requested resource was not found",
        "timestamp": chrono::Utc::now()
    }))
}
