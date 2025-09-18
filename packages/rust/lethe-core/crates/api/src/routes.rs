use crate::{handlers::*, middleware::*, state::AppState};
use axum::{
    middleware,
    routing::{delete, get, post, put},
    Router,
};

/// Create the main application router with all routes
pub fn create_router(state: AppState) -> Router {
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
        // Messages CRUD routes
        .route("/messages", post(create_message))
        .route("/messages", get(list_messages))
        .route("/messages/batch", post(batch_create_messages))
        .route("/messages/:id", get(get_message))
        .route("/messages/:id", put(update_message))
        .route("/messages/:id", delete(delete_message))
        .route(
            "/sessions/:session_id/messages/recent",
            get(get_recent_messages),
        )
        // Chunks CRUD routes
        .route("/chunks", post(create_chunk))
        .route("/chunks", get(list_chunks))
        .route("/chunks/batch", post(batch_create_chunks))
        .route("/chunks/:id", get(get_chunk))
        .route("/chunks/:id", delete(delete_chunk))
        .route("/sessions/:session_id/chunks", get(get_chunks_by_session))
        .route("/messages/:message_id/chunks", get(get_chunks_by_message))
        // Sessions CRUD routes
        .route("/sessions", post(create_session))
        .route("/sessions", get(list_sessions))
        .route("/sessions/:id", get(get_session))
        .route("/sessions/:id", put(update_session))
        .route("/sessions/:id", delete(delete_session))
        // Session state routes
        .route("/sessions/:session_id/state", get(get_session_state))
        .route("/sessions/:session_id/state", delete(clear_session_state))
        .route(
            "/sessions/:session_id/state/:key",
            get(get_session_state_value),
        )
        .route("/sessions/:session_id/state/:key", put(set_session_state))
        .route(
            "/sessions/:session_id/state/:key",
            delete(delete_session_state_value),
        )
        // Embeddings routes
        .route("/embeddings", post(create_embedding))
        .route("/embeddings", get(list_embeddings))
        .route("/embeddings/batch", post(batch_create_embeddings))
        .route("/embeddings/search", post(similarity_search))
        .route("/embeddings/:chunk_id", get(get_embedding))
        .route("/embeddings/:chunk_id", delete(delete_embedding))
        .route(
            "/sessions/:session_id/embeddings",
            get(get_embeddings_by_session),
        )
        // Middleware test endpoint
        .route("/middleware/health", get(middleware_health_check))
        // Apply middleware layers
        .layer(middleware::from_fn(security_headers_middleware))
        .layer(middleware::from_fn(error_handling_middleware))
        .layer(middleware::from_fn(timing_middleware))
        .layer(middleware::from_fn(request_id_middleware))
        .layer(middleware::from_fn(rate_limit_middleware))
        .layer(middleware::from_fn(auth_middleware))
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
