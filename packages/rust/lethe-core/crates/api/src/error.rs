use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use lethe_shared::LetheError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

/// API-specific errors
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Domain error: {0}")]
    Domain(#[from] LetheError),

    #[error("Validation error: {message}")]
    Validation { message: String },

    #[error("Authentication required")]
    Authentication,

    #[error("Access forbidden")]
    Forbidden,

    #[error("Resource not found: {resource}")]
    NotFound { resource: String },

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Internal server error: {message}")]
    Internal { message: String },

    #[error("Bad request: {message}")]
    BadRequest { message: String },

    #[error("Service unavailable: {message}")]
    ServiceUnavailable { message: String },
}

/// Standard API error response format
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub request_id: Option<String>,
}

impl ApiError {
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    pub fn not_found(resource: impl Into<String>) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest {
            message: message.into(),
        }
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self::ServiceUnavailable {
            message: message.into(),
        }
    }

    /// Get HTTP status code for this error
    pub fn status_code(&self) -> StatusCode {
        match self {
            ApiError::Domain(e) => match e {
                LetheError::Validation { .. } => StatusCode::BAD_REQUEST,
                LetheError::NotFound { .. } => StatusCode::NOT_FOUND,
                LetheError::Database { .. } => StatusCode::INTERNAL_SERVER_ERROR,
                LetheError::Authentication { .. } => StatusCode::UNAUTHORIZED,
                LetheError::Authorization { .. } => StatusCode::FORBIDDEN,
                LetheError::Timeout { .. } => StatusCode::GATEWAY_TIMEOUT,
                LetheError::ExternalService { .. } => StatusCode::BAD_GATEWAY,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            },
            ApiError::Validation { .. } => StatusCode::BAD_REQUEST,
            ApiError::Authentication => StatusCode::UNAUTHORIZED,
            ApiError::Forbidden => StatusCode::FORBIDDEN,
            ApiError::NotFound { .. } => StatusCode::NOT_FOUND,
            ApiError::RateLimit => StatusCode::TOO_MANY_REQUESTS,
            ApiError::BadRequest { .. } => StatusCode::BAD_REQUEST,
            ApiError::ServiceUnavailable { .. } => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::Internal { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Get error type string
    pub fn error_type(&self) -> &'static str {
        match self {
            ApiError::Domain(_) => "domain_error",
            ApiError::Validation { .. } => "validation_error",
            ApiError::Authentication => "authentication_error",
            ApiError::Forbidden => "forbidden_error",
            ApiError::NotFound { .. } => "not_found_error",
            ApiError::RateLimit => "rate_limit_error",
            ApiError::BadRequest { .. } => "bad_request_error",
            ApiError::ServiceUnavailable { .. } => "service_unavailable_error",
            ApiError::Internal { .. } => "internal_error",
        }
    }

    fn details_value(&self) -> Option<Value> {
        match self {
            ApiError::Domain(domain_error) => match domain_error {
                LetheError::Validation { field, reason } => {
                    Some(json!({"field": field, "reason": reason}))
                }
                LetheError::NotFound { resource_type, id } => {
                    Some(json!({"resource_type": resource_type, "id": id}))
                }
                LetheError::ExternalService { service, message } => {
                    Some(json!({"service": service, "message": message}))
                }
                LetheError::Timeout {
                    operation,
                    timeout_ms,
                } => Some(json!({"operation": operation, "timeout_ms": timeout_ms})),
                LetheError::Authentication { message }
                | LetheError::Authorization { message }
                | LetheError::Embedding { message }
                | LetheError::Config { message }
                | LetheError::Vector { message }
                | LetheError::MathOptimization { message }
                | LetheError::Internal { message }
                | LetheError::Pipeline { message, .. }
                | LetheError::Database { message } => Some(json!({"message": message})),
                _ => None,
            },
            ApiError::Validation { message }
            | ApiError::BadRequest { message }
            | ApiError::Internal { message }
            | ApiError::ServiceUnavailable { message } => Some(json!({"message": message})),
            ApiError::NotFound { resource } => Some(json!({"resource": resource})),
            ApiError::Authentication => Some(json!({"reason": "authentication_required"})),
            ApiError::Forbidden => Some(json!({"reason": "forbidden"})),
            _ => None,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_response = ErrorResponse {
            error: self.error_type().to_string(),
            message: self.to_string(),
            details: self.details_value(),
            timestamp: chrono::Utc::now(),
            request_id: None, // Could be populated by middleware
        };

        // Log the error
        match status {
            StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE => {
                tracing::error!(error = %self, "API error occurred");
            }
            StatusCode::BAD_REQUEST
            | StatusCode::UNAUTHORIZED
            | StatusCode::FORBIDDEN
            | StatusCode::NOT_FOUND => {
                tracing::warn!(error = %self, "Client error occurred");
            }
            _ => {
                tracing::info!(error = %self, "API error occurred");
            }
        }

        (status, Json(error_response)).into_response()
    }
}

impl From<validator::ValidationErrors> for ApiError {
    fn from(errors: validator::ValidationErrors) -> Self {
        let message = errors
            .field_errors()
            .into_iter()
            .map(|(field, errors)| {
                let field_errors: Vec<String> = errors
                    .iter()
                    .map(|error| {
                        error
                            .message
                            .as_ref()
                            .map(|m| m.to_string())
                            .unwrap_or_else(|| format!("Invalid value for field '{}'", field))
                    })
                    .collect();
                format!("{}: {}", field, field_errors.join(", "))
            })
            .collect::<Vec<_>>()
            .join("; ");

        ApiError::validation(message)
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(err: serde_json::Error) -> Self {
        ApiError::bad_request(format!("Invalid JSON: {}", err))
    }
}

/// Result type alias for API operations
pub type ApiResult<T> = Result<T, ApiError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_status_codes() {
        assert_eq!(
            ApiError::validation("test".to_string()).status_code(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            ApiError::Authentication.status_code(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(ApiError::Forbidden.status_code(), StatusCode::FORBIDDEN);
        assert_eq!(
            ApiError::not_found("resource".to_string()).status_code(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            ApiError::RateLimit.status_code(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            ApiError::internal("test".to_string()).status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn test_error_types() {
        assert_eq!(
            ApiError::validation("test".to_string()).error_type(),
            "validation_error"
        );
        assert_eq!(
            ApiError::Authentication.error_type(),
            "authentication_error"
        );
        assert_eq!(
            ApiError::not_found("resource".to_string()).error_type(),
            "not_found_error"
        );
    }
}
