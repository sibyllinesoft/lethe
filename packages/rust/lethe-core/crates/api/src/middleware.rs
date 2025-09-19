use std::{sync::Arc, time::Instant};

use crate::{
    error::ApiError,
    security::{AuthenticatedIdentity, RateLimitOutcome, SecurityContext},
};
use axum::{
    extract::{Request, State},
    http::{header, HeaderMap, HeaderName, HeaderValue},
    middleware::Next,
    response::{IntoResponse, Response},
};
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

/// Rate limiting middleware using the configured security context.
pub async fn rate_limit_middleware(
    State(security): State<Arc<SecurityContext>>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    if let Some(limiter) = security.rate_limiter() {
        let client_key = security.extract_client_identifier(request.headers());

        match limiter.check(&client_key) {
            RateLimitOutcome::Allow => {
                tracing::trace!(client = %client_key, "rate limit passed");
            }
            RateLimitOutcome::Deny { retry_after } => {
                tracing::warn!(client = %client_key, ?retry_after, "rate limit exceeded");
                let mut response = ApiError::RateLimit.into_response();
                let retry_after_value = retry_after.as_secs().max(1);
                if let Ok(header_value) = HeaderValue::from_str(&retry_after_value.to_string()) {
                    response
                        .headers_mut()
                        .insert(header::RETRY_AFTER, header_value);
                }
                return Ok(response);
            }
        }
    }

    Ok(next.run(request).await)
}

/// Enforce authentication based on security configuration.
pub async fn auth_middleware(
    State(security): State<Arc<SecurityContext>>,
    mut request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let identity = authenticate_request(&security, request.headers())?;

    if let Some(identity) = identity.clone() {
        request.extensions_mut().insert(identity);
    }

    if security.authentication_required() && identity.is_none() {
        tracing::warn!("Authentication failed for incoming request");
        return Err(ApiError::Authentication);
    }

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

fn authenticate_request(
    security: &SecurityContext,
    headers: &HeaderMap,
) -> Result<Option<AuthenticatedIdentity>, ApiError> {
    if let Some(custom_header) = security.api_key_header() {
        if let Some(value) = headers.get(custom_header) {
            if let Ok(token) = value.to_str() {
                if let Some(identity) = security.try_api_key(token.trim()) {
                    return Ok(Some(identity));
                }
            } else {
                tracing::warn!(header = %custom_header, "Invalid characters in configured API key header");
            }
        }
    }

    if let Some(value) = headers.get(HeaderName::from_static("x-api-key")) {
        if let Ok(token) = value.to_str() {
            if let Some(identity) = security.try_api_key(token.trim()) {
                return Ok(Some(identity));
            }
        }
    }

    if let Some(value) = headers.get(header::AUTHORIZATION) {
        match value.to_str() {
            Ok(raw) => {
                let mut parts = raw.trim().splitn(2, |c: char| c.is_whitespace());
                let scheme = parts
                    .next()
                    .map(|s| s.to_ascii_lowercase())
                    .unwrap_or_default();
                let credential = parts.next().unwrap_or("").trim();

                match scheme.as_str() {
                    "apikey" => {
                        if credential.is_empty() {
                            tracing::warn!("ApiKey authorization header missing credential");
                        } else if let Some(identity) = security.try_api_key(credential) {
                            return Ok(Some(identity));
                        }
                    }
                    "bearer" => {
                        if credential.is_empty() {
                            tracing::warn!("Bearer token missing in Authorization header");
                        } else if let Some(identity) = security.try_jwt(credential)? {
                            return Ok(Some(identity));
                        }
                    }
                    _ => {
                        if !scheme.is_empty() {
                            tracing::debug!(scheme, "Unsupported authorization scheme");
                        }
                    }
                }
            }
            Err(err) => {
                tracing::warn!(error = %err, "Authorization header is not valid UTF-8");
            }
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cors_layer_creation() {
        let _ = create_cors_layer();
    }

    #[test]
    fn test_middleware_health_response() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let response = middleware_health_check().await.into_response();
            assert_eq!(response.status(), axum::http::StatusCode::OK);
        });
    }
}
