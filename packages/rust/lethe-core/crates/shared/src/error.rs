use thiserror::Error;

/// Main error type for the Lethe system
#[derive(Error, Debug)]
pub enum LetheError {
    /// Database-related errors
    #[error("Database error: {message}")]
    Database { message: String },

    /// Embedding service errors
    #[error("Embedding error: {message}")]
    Embedding { message: String },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Config { message: String },

    /// Validation errors
    #[error("Validation error in {field}: {reason}")]
    Validation { field: String, reason: String },

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// HTTP client errors
    #[error("HTTP client error: {0}")]
    Http(#[from] reqwest::Error),

    /// Timeout errors
    #[error("Operation timed out: {operation} after {timeout_ms}ms")]
    Timeout { operation: String, timeout_ms: u64 },

    /// Resource not found
    #[error("Resource not found: {resource_type} with id {id}")]
    NotFound { resource_type: String, id: String },

    /// Authentication errors
    #[error("Authentication failed: {message}")]
    Authentication { message: String },

    /// Authorization errors
    #[error("Authorization failed: {message}")]
    Authorization { message: String },

    /// External service errors
    #[error("External service error: {service} - {message}")]
    ExternalService { service: String, message: String },

    /// Processing pipeline errors
    #[error("Pipeline error: {stage} - {message}")]
    Pipeline { stage: String, message: String },

    /// Vector operations errors
    #[error("Vector operation error: {message}")]
    Vector { message: String },

    /// Mathematical optimization errors
    #[error("Mathematical optimization error: {message}")]
    MathOptimization { message: String },

    /// Generic internal errors
    #[error("Internal error: {message}")]
    Internal { message: String },
}

/// Result type alias for Lethe operations
pub type Result<T> = std::result::Result<T, LetheError>;

impl LetheError {
    /// Create a database error
    pub fn database(message: impl Into<String>) -> Self {
        Self::Database {
            message: message.into(),
        }
    }

    /// Create an embedding error
    pub fn embedding(message: impl Into<String>) -> Self {
        Self::Embedding {
            message: message.into(),
        }
    }

    /// Create a configuration error
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// Create a validation error
    pub fn validation(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Validation {
            field: field.into(),
            reason: reason.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout(operation: impl Into<String>, timeout_ms: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            timeout_ms,
        }
    }

    /// Create a not found error
    pub fn not_found(resource_type: impl Into<String>, id: impl Into<String>) -> Self {
        Self::NotFound {
            resource_type: resource_type.into(),
            id: id.into(),
        }
    }

    /// Create an external service error
    pub fn external_service(service: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ExternalService {
            service: service.into(),
            message: message.into(),
        }
    }

    /// Create a pipeline error
    pub fn pipeline(stage: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Pipeline {
            stage: stage.into(),
            message: message.into(),
        }
    }

    /// Create a vector operation error
    pub fn vector(message: impl Into<String>) -> Self {
        Self::Vector {
            message: message.into(),
        }
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
}

impl From<sqlx::Error> for LetheError {
    fn from(err: sqlx::Error) -> Self {
        Self::database(err.to_string())
    }
}

impl From<validator::ValidationErrors> for LetheError {
    fn from(err: validator::ValidationErrors) -> Self {
        Self::validation("validation", err.to_string())
    }
}
