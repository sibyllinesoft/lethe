use crate::error::{LetheError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Newtype for alpha values ensuring 0.0 <= alpha <= 1.0
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Alpha(f64);

impl Alpha {
    pub fn new(value: f64) -> Result<Self> {
        if value < 0.0 || value > 1.0 {
            Err(LetheError::validation("alpha", "Must be between 0 and 1"))
        } else {
            Ok(Alpha(value))
        }
    }

    pub fn value(self) -> f64 {
        self.0
    }
}

impl Default for Alpha {
    fn default() -> Self {
        Alpha(0.7) // Safe default
    }
}

/// Newtype for beta values ensuring 0.0 <= beta <= 1.0
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Beta(f64);

impl Beta {
    pub fn new(value: f64) -> Result<Self> {
        if value < 0.0 || value > 1.0 {
            Err(LetheError::validation("beta", "Must be between 0 and 1"))
        } else {
            Ok(Beta(value))
        }
    }

    pub fn value(self) -> f64 {
        self.0
    }
}

impl Default for Beta {
    fn default() -> Self {
        Beta(0.5) // Safe default
    }
}

/// Newtype for positive token counts
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PositiveTokens(i32);

impl PositiveTokens {
    pub fn new(value: i32) -> Result<Self> {
        if value <= 0 {
            Err(LetheError::validation("tokens", "Must be positive"))
        } else {
            Ok(PositiveTokens(value))
        }
    }

    pub fn value(self) -> i32 {
        self.0
    }
}

impl Default for PositiveTokens {
    fn default() -> Self {
        PositiveTokens(320) // Safe default
    }
}

/// Newtype for timeout values in milliseconds
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimeoutMs(u64);

impl TimeoutMs {
    pub fn new(value: u64) -> Result<Self> {
        if value == 0 {
            Err(LetheError::validation("timeout", "Must be positive"))
        } else {
            Ok(TimeoutMs(value))
        }
    }

    pub fn value(self) -> u64 {
        self.0
    }
}

impl Default for TimeoutMs {
    fn default() -> Self {
        TimeoutMs(10000) // Safe default
    }
}

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LetheConfig {
    pub version: String,
    pub description: Option<String>,
    pub retrieval: RetrievalConfig,
    pub chunking: ChunkingConfig,
    pub timeouts: TimeoutsConfig,
    pub features: Option<FeaturesConfig>,
    #[serde(default = "default_llm_config_option")]
    pub llm: Option<LlmConfig>,
    pub query_understanding: Option<QueryUnderstandingConfig>,
    pub ml: Option<MlConfig>,
    pub development: Option<DevelopmentConfig>,
    #[serde(default)]
    pub security: SecurityConfig,
    pub lens: Option<LensConfig>,
    pub storage: StorageConfig,
    pub embedding: EmbeddingConfig,
    pub repository_preloading: Option<RepositoryPreloadingConfig>,
}

/// Retrieval algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub alpha: Alpha,
    pub beta: Beta,
    #[serde(default = "default_gamma_kind_boost")]
    pub gamma_kind_boost: HashMap<String, f64>,
    #[serde(default)]
    pub fusion: Option<FusionConfig>,
    #[serde(default)]
    pub llm_rerank: Option<LlmRerankConfig>,
}

fn default_gamma_kind_boost() -> HashMap<String, f64> {
    let mut map = HashMap::new();
    map.insert("code".to_string(), 0.1);
    map.insert("text".to_string(), 0.0);
    map
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            alpha: Alpha::default(),
            beta: Beta::default(),
            gamma_kind_boost: default_gamma_kind_boost(),
            fusion: Some(FusionConfig::default()),
            llm_rerank: Some(LlmRerankConfig::default()),
        }
    }
}

fn default_llm_config_option() -> Option<LlmConfig> {
    Some(LlmConfig::default())
}

/// Fusion configuration for dynamic parameter adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    #[serde(default)]
    pub dynamic: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self { dynamic: false }
    }
}

/// LLM reranking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRerankConfig {
    #[serde(default)]
    pub use_llm: bool,
    #[serde(default = "default_llm_budget")]
    pub llm_budget_ms: u64,
    #[serde(default = "default_llm_model")]
    pub llm_model: String,
    #[serde(default)]
    pub contradiction_enabled: bool,
    #[serde(default = "default_contradiction_penalty")]
    pub contradiction_penalty: f64,
}

fn default_llm_budget() -> u64 {
    1200
}
fn default_llm_model() -> String {
    "llama3.2:1b".to_string()
}
fn default_contradiction_penalty() -> f64 {
    0.15
}

impl Default for LlmRerankConfig {
    fn default() -> Self {
        Self {
            use_llm: false,
            llm_budget_ms: default_llm_budget(),
            llm_model: default_llm_model(),
            contradiction_enabled: false,
            contradiction_penalty: default_contradiction_penalty(),
        }
    }
}

/// Text chunking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    pub target_tokens: PositiveTokens,
    pub overlap: i32, // Can be 0, validated relative to target_tokens
    #[serde(default = "default_chunking_method")]
    pub method: String,
}

fn default_chunking_method() -> String {
    "semantic".to_string()
}

impl ChunkingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.overlap < 0 || self.overlap >= self.target_tokens.value() {
            return Err(LetheError::validation(
                "chunking.overlap",
                "Must be non-negative and less than target_tokens",
            ));
        }
        Ok(())
    }
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            target_tokens: PositiveTokens::default(),
            overlap: 64,
            method: default_chunking_method(),
        }
    }
}

/// Operation timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutsConfig {
    #[serde(default)]
    pub hyde_ms: TimeoutMs,
    #[serde(default)]
    pub summarize_ms: TimeoutMs,
    #[serde(default = "default_connect_timeout")]
    pub ollama_connect_ms: TimeoutMs,
    pub ml_prediction_ms: Option<TimeoutMs>,
}

fn default_connect_timeout() -> TimeoutMs {
    TimeoutMs::new(500).unwrap()
}

impl Default for TimeoutsConfig {
    fn default() -> Self {
        Self {
            hyde_ms: TimeoutMs::default(),
            summarize_ms: TimeoutMs::default(),
            ollama_connect_ms: default_connect_timeout(),
            ml_prediction_ms: Some(TimeoutMs::new(2000).unwrap()),
        }
    }
}

/// Feature toggles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturesConfig {
    #[serde(default = "default_true")]
    pub enable_hyde: bool,
    #[serde(default = "default_true")]
    pub enable_summarization: bool,
    #[serde(default = "default_true")]
    pub enable_plan_selection: bool,
    #[serde(default = "default_true")]
    pub enable_query_understanding: bool,
    #[serde(default)]
    pub enable_ml_prediction: bool,
    #[serde(default = "default_true")]
    pub enable_state_tracking: bool,
}

fn default_true() -> bool {
    true
}

impl Default for FeaturesConfig {
    fn default() -> Self {
        Self {
            enable_hyde: true,
            enable_summarization: true,
            enable_plan_selection: true,
            enable_query_understanding: true,
            enable_ml_prediction: false,
            enable_state_tracking: true,
        }
    }
}

/// Query understanding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryUnderstandingConfig {
    #[serde(default = "default_true")]
    pub rewrite_enabled: bool,
    #[serde(default = "default_true")]
    pub decompose_enabled: bool,
    #[serde(default = "default_max_subqueries")]
    pub max_subqueries: i32,
    #[serde(default = "default_llm_model")]
    pub llm_model: String,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
}

fn default_max_subqueries() -> i32 {
    3
}
fn default_temperature() -> f64 {
    0.1
}

impl Default for QueryUnderstandingConfig {
    fn default() -> Self {
        Self {
            rewrite_enabled: true,
            decompose_enabled: true,
            max_subqueries: default_max_subqueries(),
            llm_model: default_llm_model(),
            temperature: default_temperature(),
        }
    }
}

/// Large language model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    #[serde(default)]
    pub provider: LlmProvider,
    #[serde(default = "default_llm_temperature")]
    pub temperature: f32,
    #[serde(default = "default_llm_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_llm_timeout")]
    pub timeout_ms: u64,
}

fn default_llm_temperature() -> f32 {
    0.7
}

fn default_llm_max_tokens() -> usize {
    512
}

fn default_llm_timeout() -> u64 {
    15000
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: LlmProvider::default(),
            temperature: default_llm_temperature(),
            max_tokens: default_llm_max_tokens(),
            timeout_ms: default_llm_timeout(),
        }
    }
}

/// Supported LLM providers
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LlmProvider {
    Ollama { base_url: String, model: String },
}

impl Default for LlmProvider {
    fn default() -> Self {
        Self::Ollama {
            base_url: "http://localhost:11434".to_string(),
            model: default_llm_model(),
        }
    }
}

/// Machine learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlConfig {
    #[serde(default)]
    pub prediction_service: Option<PredictionServiceConfig>,
    #[serde(default)]
    pub models: Option<ModelsConfig>,
}

impl Default for MlConfig {
    fn default() -> Self {
        Self {
            prediction_service: Some(PredictionServiceConfig::default()),
            models: Some(ModelsConfig::default()),
        }
    }
}

/// ML prediction service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionServiceConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_service_timeout")]
    pub timeout_ms: u64,
    #[serde(default = "default_true")]
    pub fallback_to_static: bool,
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}
fn default_port() -> u16 {
    8080
}
fn default_service_timeout() -> u64 {
    2000
}

impl PredictionServiceConfig {
    pub fn validate(&self) -> Result<()> {
        if self.enabled {
            if self.port == 0 {
                return Err(LetheError::validation(
                    "ml.prediction_service.port",
                    "Must be a valid port number",
                ));
            }
            if self.timeout_ms == 0 {
                return Err(LetheError::validation(
                    "ml.prediction_service.timeout_ms",
                    "Must be positive",
                ));
            }
        }
        Ok(())
    }
}

impl Default for PredictionServiceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            host: default_host(),
            port: default_port(),
            timeout_ms: default_service_timeout(),
            fallback_to_static: true,
        }
    }
}

/// ML models configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    #[serde(default = "default_plan_selector")]
    pub plan_selector: Option<String>,
    #[serde(default = "default_fusion_weights")]
    pub fusion_weights: Option<String>,
    #[serde(default = "default_feature_extractor")]
    pub feature_extractor: Option<String>,
}

fn default_plan_selector() -> Option<String> {
    Some("learned_plan_selector.joblib".to_string())
}
fn default_fusion_weights() -> Option<String> {
    Some("dynamic_fusion_model.joblib".to_string())
}
fn default_feature_extractor() -> Option<String> {
    Some("feature_extractor.json".to_string())
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            plan_selector: default_plan_selector(),
            fusion_weights: default_fusion_weights(),
            feature_extractor: default_feature_extractor(),
        }
    }
}

/// Development-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevelopmentConfig {
    #[serde(default)]
    pub debug_enabled: bool,
    #[serde(default)]
    pub profiling_enabled: bool,
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

fn default_log_level() -> String {
    "info".to_string()
}

impl Default for DevelopmentConfig {
    fn default() -> Self {
        Self {
            debug_enabled: false,
            profiling_enabled: false,
            log_level: default_log_level(),
        }
    }
}

/// API security configuration (authentication & rate limiting)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Require clients to authenticate before accessing the API
    #[serde(default)]
    pub require_authentication: bool,
    /// List of pre-shared API keys that are allowed to access the service
    #[serde(default)]
    pub api_keys: Vec<String>,
    /// Optional name of a header that carries the API key (defaults to `authorization`)
    #[serde(default)]
    pub api_key_header: Option<String>,
    /// Configuration for validating bearer JWTs
    #[serde(default)]
    pub jwt: Option<JwtConfig>,
    /// Optional rate-limiting configuration
    #[serde(default)]
    pub rate_limit: Option<RateLimitConfig>,
    /// Optional header that contains the real client IP when behind a proxy
    #[serde(default)]
    pub client_ip_header: Option<String>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            require_authentication: false,
            api_keys: Vec::new(),
            api_key_header: None,
            jwt: None,
            rate_limit: Some(RateLimitConfig::default()),
            client_ip_header: None,
        }
    }
}

/// JWT authentication options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtConfig {
    /// HMAC secret used to validate tokens
    pub secret: String,
    /// Expected issuer (optional)
    #[serde(default)]
    pub issuer: Option<String>,
    /// Expected audience (optional)
    #[serde(default)]
    pub audience: Option<String>,
    /// Allowed leeway when validating expiry (seconds)
    #[serde(default = "default_jwt_leeway")]
    pub leeway_seconds: u64,
}

fn default_jwt_leeway() -> u64 {
    60
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum number of requests per minute
    pub requests_per_minute: u32,
    /// Maximum burst size allowed before throttling
    #[serde(default = "default_rate_limit_burst")]
    pub burst: u32,
    /// Header to trust for client identity (defaults to IP detection)
    #[serde(default)]
    pub identifier_header: Option<String>,
}

fn default_rate_limit_burst() -> u32 {
    60
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 120,
            burst: default_rate_limit_burst(),
            identifier_header: None,
        }
    }
}

/// Lens integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_lens_base_url")]
    pub base_url: String,
    #[serde(default = "default_lens_connect_timeout")]
    pub connect_timeout_ms: u64,
    #[serde(default = "default_lens_request_timeout")]
    pub request_timeout_ms: u64,
    #[serde(default = "default_lens_request_timeout")]
    pub sla_recall_ms: u64,
    #[serde(default = "default_topic_fanout_k")]
    pub topic_fanout_k: i32,
    #[serde(default = "default_weight_cap")]
    pub weight_cap: f64,
    #[serde(default = "default_max_tokens_per_response")]
    pub max_tokens_per_response: i32,
    #[serde(default = "default_lens_mode")]
    pub mode: String,
    #[serde(default = "default_dpp_rank")]
    pub dpp_rank: i32,
    #[serde(default = "default_true")]
    pub enable_facility_location: bool,
    #[serde(default = "default_true")]
    pub enable_log_det_dpp: bool,
    #[serde(default = "default_lambda_multiplier")]
    pub lambda_multiplier: f64,
    #[serde(default = "default_mu_multiplier")]
    pub mu_multiplier: f64,
    #[serde(default = "default_max_tokens_per_response")]
    pub lens_tokens_cap: i32,
}

fn default_lens_base_url() -> String {
    "http://localhost:8081".to_string()
}
fn default_lens_connect_timeout() -> u64 {
    500
}
fn default_lens_request_timeout() -> u64 {
    150
}
fn default_topic_fanout_k() -> i32 {
    240
}
fn default_weight_cap() -> f64 {
    0.4
}
fn default_max_tokens_per_response() -> i32 {
    4000
}
fn default_lens_mode() -> String {
    "auto".to_string()
}
fn default_dpp_rank() -> i32 {
    14
}
fn default_lambda_multiplier() -> f64 {
    1.2
}
fn default_mu_multiplier() -> f64 {
    1.0
}

impl LensConfig {
    pub fn validate(&self) -> Result<()> {
        if self.enabled {
            if self.sla_recall_ms == 0 || self.sla_recall_ms > 1000 {
                return Err(LetheError::validation(
                    "lens.sla_recall_ms",
                    "Must be between 0 and 1000",
                ));
            }
            if self.topic_fanout_k <= 0 || self.topic_fanout_k > 1000 {
                return Err(LetheError::validation(
                    "lens.topic_fanout_k",
                    "Must be between 0 and 1000",
                ));
            }
            if self.weight_cap <= 0.0 || self.weight_cap > 1.0 {
                return Err(LetheError::validation(
                    "lens.weight_cap",
                    "Must be between 0 and 1.0",
                ));
            }
            if !self.base_url.starts_with("http") {
                return Err(LetheError::validation(
                    "lens.base_url",
                    "Must be a valid HTTP URL",
                ));
            }
        }
        Ok(())
    }
}

impl Default for LensConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: default_lens_base_url(),
            connect_timeout_ms: default_lens_connect_timeout(),
            request_timeout_ms: default_lens_request_timeout(),
            sla_recall_ms: default_lens_request_timeout(),
            topic_fanout_k: default_topic_fanout_k(),
            weight_cap: default_weight_cap(),
            max_tokens_per_response: default_max_tokens_per_response(),
            mode: default_lens_mode(),
            dpp_rank: default_dpp_rank(),
            enable_facility_location: true,
            enable_log_det_dpp: true,
            lambda_multiplier: default_lambda_multiplier(),
            mu_multiplier: default_mu_multiplier(),
            lens_tokens_cap: default_max_tokens_per_response(),
        }
    }
}

/// Storage configuration for file-backed repositories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    #[serde(default = "default_storage_index_root")]
    pub index_root: String,
    #[serde(default)]
    pub persist_embeddings: bool,
}

fn default_storage_index_root() -> String {
    "./data/index".to_string()
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            index_root: default_storage_index_root(),
            persist_embeddings: true,
        }
    }
}

/// Embedding service provider
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum EmbeddingProvider {
    Ollama { base_url: String, model: String },
    Fallback,
}

impl Default for EmbeddingProvider {
    fn default() -> Self {
        Self::Ollama {
            base_url: "http://localhost:11434".to_string(),
            model: "nomic-embed-text".to_string(),
        }
    }
}

/// Embedding service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    #[serde(default)]
    pub provider: EmbeddingProvider,
    #[serde(default = "default_embedding_dimension")]
    pub dimension: usize,
    #[serde(default = "default_embedding_timeout")]
    pub timeout_ms: u64,
}

fn default_embedding_dimension() -> usize {
    768
}
fn default_embedding_timeout() -> u64 {
    10000
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::default(),
            dimension: default_embedding_dimension(),
            timeout_ms: default_embedding_timeout(),
        }
    }
}

/// Repository preloading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryPreloadingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub repositories: Vec<RepositoryConfig>,
    #[serde(default = "default_max_concurrent_repos")]
    pub max_concurrent_repos: usize,
    #[serde(default = "default_true")]
    pub fail_on_error: bool,
    #[serde(default)]
    pub file_patterns: Vec<String>,
    #[serde(default)]
    pub exclude_patterns: Vec<String>,
}

fn default_max_concurrent_repos() -> usize {
    4
}

impl Default for RepositoryPreloadingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            repositories: Vec::new(),
            max_concurrent_repos: default_max_concurrent_repos(),
            fail_on_error: false,
            file_patterns: vec![
                "*.rs".to_string(),
                "*.py".to_string(),
                "*.js".to_string(),
                "*.ts".to_string(),
                "*.java".to_string(),
                "*.cpp".to_string(),
                "*.c".to_string(),
                "*.h".to_string(),
                "*.md".to_string(),
                "*.txt".to_string(),
            ],
            exclude_patterns: vec![
                "target/**".to_string(),
                "node_modules/**".to_string(),
                ".git/**".to_string(),
                "*.lock".to_string(),
                "*.log".to_string(),
            ],
        }
    }
}

/// Individual repository configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryConfig {
    pub path: String,
    pub name: Option<String>,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub file_patterns: Option<Vec<String>>,
    #[serde(default)]
    pub exclude_patterns: Option<Vec<String>>,
}

impl RepositoryConfig {
    pub fn new<S: Into<String>>(path: S) -> Self {
        Self {
            path: path.into(),
            name: None,
            enabled: true,
            file_patterns: None,
            exclude_patterns: None,
        }
    }

    pub fn with_name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_patterns(mut self, patterns: Vec<String>) -> Self {
        self.file_patterns = Some(patterns);
        self
    }

    pub fn with_excludes(mut self, excludes: Vec<String>) -> Self {
        self.exclude_patterns = Some(excludes);
        self
    }
}

impl Default for LetheConfig {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            description: Some("Default Lethe configuration".to_string()),
            retrieval: RetrievalConfig::default(),
            chunking: ChunkingConfig::default(),
            timeouts: TimeoutsConfig::default(),
            features: Some(FeaturesConfig::default()),
            llm: default_llm_config_option(),
            query_understanding: Some(QueryUnderstandingConfig::default()),
            ml: Some(MlConfig::default()),
            development: Some(DevelopmentConfig::default()),
            security: SecurityConfig::default(),
            lens: Some(LensConfig::default()),
            storage: StorageConfig::default(),
            embedding: EmbeddingConfig::default(),
            repository_preloading: Some(RepositoryPreloadingConfig::default()),
        }
    }
}

impl LetheConfig {
    /// Load configuration from file
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| LetheError::config(format!("Failed to read config file: {}", e)))?;

        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("json");

        let config: Self = match extension.to_lowercase().as_str() {
            "toml" => toml::from_str(&content)
                .map_err(|e| LetheError::config(format!("Failed to parse TOML config: {}", e)))?,
            "yaml" | "yml" => serde_yaml::from_str(&content)
                .map_err(|e| LetheError::config(format!("Failed to parse YAML config: {}", e)))?,
            "json" | _ => serde_json::from_str(&content)
                .map_err(|e| LetheError::config(format!("Failed to parse JSON config: {}", e)))?,
        };

        config.validate()?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn to_file(&self, path: &std::path::Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)
            .map_err(|e| LetheError::config(format!("Failed to write config file: {}", e)))?;
        Ok(())
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<()> {
        // Alpha and Beta are now validated at construction time via newtype wrappers

        // Validate chunking configuration
        self.chunking.validate()?;

        // Timeout validation is now handled by TimeoutMs newtype

        // Validate ML service configuration
        if let Some(ml) = &self.ml {
            if let Some(service) = &ml.prediction_service {
                service.validate()?;
            }
        }

        // Validate Lens configuration
        if let Some(lens) = &self.lens {
            lens.validate()?;
        }

        Ok(())
    }

    /// Merge with another configuration, preferring other's values
    pub fn merge_with(&mut self, other: &Self) {
        self.version = other.version.clone();

        // Use Option::or to prefer other's value when it exists
        if other.description.is_some() {
            self.description = other.description.clone();
        }

        // Always merge core configs (they should always exist)
        self.retrieval = other.retrieval.clone();
        self.chunking = other.chunking.clone();
        self.timeouts = other.timeouts.clone();
        self.storage = other.storage.clone();
        self.embedding = other.embedding.clone();

        // Use or_else for optional configs to maintain existing values when other is None
        self.features = other.features.clone().or_else(|| self.features.clone());
        self.llm = other.llm.clone().or_else(|| self.llm.clone());
        self.query_understanding = other
            .query_understanding
            .clone()
            .or_else(|| self.query_understanding.clone());
        self.ml = other.ml.clone().or_else(|| self.ml.clone());
        self.development = other
            .development
            .clone()
            .or_else(|| self.development.clone());
        self.lens = other.lens.clone().or_else(|| self.lens.clone());
        self.repository_preloading = other
            .repository_preloading
            .clone()
            .or_else(|| self.repository_preloading.clone());
    }

    /// Builder pattern for creating configurations
    pub fn builder() -> LetheConfigBuilder {
        LetheConfigBuilder::default()
    }
}

/// Builder for LetheConfig to make complex configurations easier
#[derive(Debug, Default)]
pub struct LetheConfigBuilder {
    config: LetheConfig,
}

impl LetheConfigBuilder {
    pub fn version<S: Into<String>>(mut self, version: S) -> Self {
        self.config.version = version.into();
        self
    }

    pub fn description<S: Into<String>>(mut self, description: S) -> Self {
        self.config.description = Some(description.into());
        self
    }

    pub fn retrieval(mut self, retrieval: RetrievalConfig) -> Self {
        self.config.retrieval = retrieval;
        self
    }

    pub fn chunking(mut self, chunking: ChunkingConfig) -> Self {
        self.config.chunking = chunking;
        self
    }

    pub fn features(mut self, features: FeaturesConfig) -> Self {
        self.config.features = Some(features);
        self
    }

    pub fn llm(mut self, llm: Option<LlmConfig>) -> Self {
        self.config.llm = llm;
        self
    }

    pub fn build(self) -> Result<LetheConfig> {
        let config = self.config;
        config.validate()?;
        Ok(config)
    }
}
