use crate::types::{DeterminismConfig, ValidationError};
use serde::{Deserialize, Serialize};
use std::env;

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub determinism: DeterminismConfig,
    pub database: DatabaseConfig,
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connect_timeout_seconds: u64,
    pub idle_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub metrics_port: u16,
    pub dashboard_update_interval_seconds: u64,
    pub alert_cooldown_seconds: u64,
    pub max_alert_history: usize,
}

impl Config {
    pub fn from_env() -> Result<Self, ValidationError> {
        Ok(Config {
            server: ServerConfig {
                host: env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: env::var("SERVER_PORT")
                    .unwrap_or_else(|_| "3001".to_string())
                    .parse()
                    .map_err(|e| ValidationError::SerializationError(
                        serde_json::Error::io(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!("Invalid SERVER_PORT: {}", e)
                        ))
                    ))?,
                workers: env::var("SERVER_WORKERS")
                    .ok()
                    .and_then(|w| w.parse().ok()),
            },
            determinism: DeterminismConfig {
                replay_interval_seconds: env::var("REPLAY_INTERVAL_SECONDS")
                    .unwrap_or_else(|_| "3600".to_string())
                    .parse()
                    .unwrap_or(3600),
                tolerance_ms: env::var("TOLERANCE_MS")
                    .unwrap_or_else(|_| "1".to_string())
                    .parse()
                    .unwrap_or(1),
                performance_budget_percent: env::var("PERFORMANCE_BUDGET_PERCENT")
                    .unwrap_or_else(|_| "2.0".to_string())
                    .parse()
                    .unwrap_or(2.0),
                max_concurrent_replays: env::var("MAX_CONCURRENT_REPLAYS")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .unwrap_or(10),
                clock_skew_test_interval_seconds: env::var("CLOCK_SKEW_TEST_INTERVAL_SECONDS")
                    .unwrap_or_else(|_| "900".to_string())
                    .parse()
                    .unwrap_or(900),
            },
            database: DatabaseConfig {
                url: env::var("DATABASE_URL")
                    .unwrap_or_else(|_| "postgresql://localhost:5432/determinism".to_string()),
                max_connections: env::var("DB_MAX_CONNECTIONS")
                    .unwrap_or_else(|_| "20".to_string())
                    .parse()
                    .unwrap_or(20),
                min_connections: env::var("DB_MIN_CONNECTIONS")
                    .unwrap_or_else(|_| "5".to_string())
                    .parse()
                    .unwrap_or(5),
                connect_timeout_seconds: env::var("DB_CONNECT_TIMEOUT_SECONDS")
                    .unwrap_or_else(|_| "30".to_string())
                    .parse()
                    .unwrap_or(30),
                idle_timeout_seconds: env::var("DB_IDLE_TIMEOUT_SECONDS")
                    .unwrap_or_else(|_| "600".to_string())
                    .parse()
                    .unwrap_or(600),
            },
            monitoring: MonitoringConfig {
                metrics_port: env::var("METRICS_PORT")
                    .unwrap_or_else(|_| "9090".to_string())
                    .parse()
                    .unwrap_or(9090),
                dashboard_update_interval_seconds: env::var("DASHBOARD_UPDATE_INTERVAL_SECONDS")
                    .unwrap_or_else(|_| "60".to_string())
                    .parse()
                    .unwrap_or(60),
                alert_cooldown_seconds: env::var("ALERT_COOLDOWN_SECONDS")
                    .unwrap_or_else(|_| "300".to_string())
                    .parse()
                    .unwrap_or(300),
                max_alert_history: env::var("MAX_ALERT_HISTORY")
                    .unwrap_or_else(|_| "1000".to_string())
                    .parse()
                    .unwrap_or(1000),
            },
        })
    }

    pub fn validate(&self) -> Result<(), ValidationError> {
        // Validate server configuration
        if self.server.port == 0 {
            return Err(ValidationError::SerializationError(
                serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Server port cannot be 0"
                ))
            ));
        }

        // Validate determinism configuration
        if self.determinism.tolerance_ms == 0 {
            return Err(ValidationError::SerializationError(
                serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Tolerance cannot be 0"
                ))
            ));
        }

        if self.determinism.performance_budget_percent <= 0.0 {
            return Err(ValidationError::SerializationError(
                serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Performance budget must be positive"
                ))
            ));
        }

        // Validate database configuration
        if self.database.max_connections == 0 {
            return Err(ValidationError::SerializationError(
                serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Max connections cannot be 0"
                ))
            ));
        }

        if self.database.min_connections > self.database.max_connections {
            return Err(ValidationError::SerializationError(
                serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Min connections cannot exceed max connections"
                ))
            ));
        }

        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 3001,
                workers: None,
            },
            determinism: DeterminismConfig::default(),
            database: DatabaseConfig {
                url: "postgresql://localhost:5432/determinism".to_string(),
                max_connections: 20,
                min_connections: 5,
                connect_timeout_seconds: 30,
                idle_timeout_seconds: 600,
            },
            monitoring: MonitoringConfig {
                metrics_port: 9090,
                dashboard_update_interval_seconds: 60,
                alert_cooldown_seconds: 300,
                max_alert_history: 1000,
            },
        }
    }
}