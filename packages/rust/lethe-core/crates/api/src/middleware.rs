use axum::{
    extract::Request,
    http::{HeaderMap, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::time::Instant;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

/// Request ID middleware for tracing
pub async fn request_id_middleware(mut request: Request, next: Next) -> Response {
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .map(|value| value.to_string())
        .unwrap_or_else(|| {
            let id = Uuid::new_v4().to_string();
            if let Ok(header_value) = HeaderValue::from_str(&id) {
                request.headers_mut().insert("x-request-id", header_value);
            }
            id
        });

    let mut response = next.run(request).await;
    if let Ok(header_value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", header_value);
    }

    response
}

/// Request timing middleware
pub async fn timing_middleware(request: Request, next: Next) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();

    let response = next.run(request).await;
    let duration = start.elapsed();

    tracing::info!(
        method = %method,
        uri = %uri,
        status = response.status().as_u16(),
        duration_ms = duration.as_millis(),
        "Request completed"
    );

    response
}

/// Rate limiting middleware (simple implementation)
pub async fn rate_limit_middleware(request: Request, next: Next) -> Result<Response, StatusCode> {
    // Simple rate limiting based on IP address
    // In production, you'd use a more sophisticated rate limiter like Redis
    let client_ip = request
        .headers()
        .get("x-forwarded-for")
        .or_else(|| request.headers().get("x-real-ip"))
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown");

    // For now, just log the client IP and proceed
    tracing::debug!(client_ip = %client_ip, "Rate limit check");

    Ok(next.run(request).await)
}

/// Authentication middleware
pub async fn auth_middleware(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Check for API key or JWT token
    if let Some(auth_header) = headers.get("authorization") {
        if let Ok(auth_value) = auth_header.to_str() {
            if auth_value.starts_with("Bearer ") || auth_value.starts_with("ApiKey ") {
                // In a real implementation, validate the token/key
                tracing::debug!("Authentication header found");
                return Ok(next.run(request).await);
            }
        }
    }

    // For development, we can make auth optional
    // In production, uncomment the line below to enforce authentication
    // return Err(StatusCode::UNAUTHORIZED);

    tracing::debug!("No authentication header found, proceeding without auth");
    Ok(next.run(request).await)
}

/// CORS configuration
pub fn create_cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin([
            "http://localhost:3000".parse().unwrap(),
            "http://localhost:3001".parse().unwrap(),
            "http://127.0.0.1:3000".parse().unwrap(),
            "http://127.0.0.1:3001".parse().unwrap(),
        ])
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::PUT,
            axum::http::Method::DELETE,
            axum::http::Method::OPTIONS,
        ])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
            axum::http::header::ACCEPT,
            axum::http::HeaderName::from_static("x-request-id"),
        ])
        .expose_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderName::from_static("x-request-id"),
        ])
}

/// Security headers middleware
pub async fn security_headers_middleware(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;

    let headers = response.headers_mut();

    // Add security headers
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert("x-frame-options", HeaderValue::from_static("DENY"));
    headers.insert(
        "x-xss-protection",
        HeaderValue::from_static("1; mode=block"),
    );
    headers.insert(
        "strict-transport-security",
        HeaderValue::from_static("max-age=31536000; includeSubDomains"),
    );
    headers.insert(
        "content-security-policy",
        HeaderValue::from_static(
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';",
        ),
    );

    response
}

/// Error handling middleware
pub async fn error_handling_middleware(request: Request, next: Next) -> Response {
    let response = next.run(request).await;

    // Log errors based on status code
    let status = response.status();
    if status.is_server_error() {
        tracing::error!(status = %status, "Server error occurred");
    } else if status.is_client_error() {
        tracing::warn!(status = %status, "Client error occurred");
    }

    response
}

/// Health check response for middleware testing
#[derive(serde::Serialize)]
struct MiddlewareHealthCheck {
    middleware: &'static str,
    status: &'static str,
    timestamp: chrono::DateTime<chrono::Utc>,
}

/// Test endpoint for middleware functionality
pub async fn middleware_health_check() -> impl IntoResponse {
    axum::Json(MiddlewareHealthCheck {
        middleware: "all",
        status: "operational",
        timestamp: chrono::Utc::now(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Method, Request as HttpRequest},
    };

    #[tokio::test]
    async fn test_cors_layer_creation() {
        let cors = create_cors_layer();
        // CORS layer creation should not panic
        assert!(true);
    }

    #[test]
    fn test_middleware_health_response() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let response = middleware_health_check().await.into_response();
            assert_eq!(response.status(), StatusCode::OK);
        });
    }
}
