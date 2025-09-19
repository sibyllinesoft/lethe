use crate::hyde::{HydeConfig, LlmService};
use async_trait::async_trait;
use lethe_shared::{
    LetheError, LlmConfig as SharedLlmConfig, LlmProvider as SharedLlmProvider, Result,
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

/// Domain-level configuration for constructing an LLM service
#[derive(Debug, Clone)]
pub struct LlmServiceConfig {
    pub provider: LlmProviderConfig,
    pub max_tokens: usize,
    pub temperature: f32,
    pub timeout_ms: u64,
}

#[async_trait]
pub trait LlmProviderBuilder: Send + Sync {
    fn id(&self) -> &'static str;
    fn matches(&self, provider: &LlmProviderConfig) -> bool;
    async fn build(
        &self,
        config: &LlmServiceConfig,
        provider: &LlmProviderConfig,
    ) -> Result<Arc<dyn LlmService>>;
}

fn llm_provider_registry() -> &'static RwLock<HashMap<&'static str, Arc<dyn LlmProviderBuilder>>> {
    static REGISTRY: OnceLock<RwLock<HashMap<&'static str, Arc<dyn LlmProviderBuilder>>>> =
        OnceLock::new();

    REGISTRY.get_or_init(|| {
        let mut providers: HashMap<&'static str, Arc<dyn LlmProviderBuilder>> = HashMap::new();
        providers.insert("ollama", Arc::new(OllamaLlmProvider));
        RwLock::new(providers)
    })
}

pub fn register_llm_provider(builder: Arc<dyn LlmProviderBuilder>) {
    let registry = llm_provider_registry();
    if let Ok(mut guard) = registry.write() {
        guard.insert(builder.id(), builder);
    }
}

/// Supported provider variants for the domain LLM service
#[derive(Debug, Clone)]
pub enum LlmProviderConfig {
    Ollama {
        base_url: String,
        model: String,
    },
    Custom {
        name: String,
        settings: Option<Value>,
    },
}

impl LlmServiceConfig {
    /// Convert a shared configuration into the domain representation
    pub fn from_shared(config: &SharedLlmConfig) -> Self {
        let provider = match &config.provider {
            SharedLlmProvider::Ollama { base_url, model } => LlmProviderConfig::Ollama {
                base_url: base_url.clone(),
                model: model.clone(),
            },
        };

        Self {
            provider,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            timeout_ms: config.timeout_ms,
        }
    }
}

/// Factory that constructs concrete LLM services
pub struct LlmServiceFactory;

impl LlmServiceFactory {
    /// Build an LLM service based on the supplied configuration
    pub async fn create(config: &LlmServiceConfig) -> Result<Arc<dyn LlmService>> {
        if let Some(result) = Self::build_from_registry(config, &config.provider).await {
            return result;
        }

        match &config.provider {
            LlmProviderConfig::Custom { name, .. } => Err(LetheError::config(format!(
                "No LLM provider registered for '{}'",
                name
            ))),
            _ => Err(LetheError::config(
                "No LLM provider available for requested configuration",
            )),
        }
    }

    async fn build_from_registry(
        config: &LlmServiceConfig,
        provider: &LlmProviderConfig,
    ) -> Option<Result<Arc<dyn LlmService>>> {
        let builder = {
            let registry = llm_provider_registry();
            let guard = registry.read().ok()?;
            match provider {
                LlmProviderConfig::Custom { name, .. } => guard.get(name.as_str()).cloned(),
                _ => guard
                    .values()
                    .find(|builder| builder.matches(provider))
                    .cloned(),
            }
        }?;

        Some(builder.build(config, provider).await)
    }
}

/// Ollama-backed LLM implementation
pub struct OllamaLlmService {
    base_url: String,
    model: String,
    client: Client,
    fallback_temperature: f32,
    fallback_max_tokens: usize,
}

impl OllamaLlmService {
    fn new(
        base_url: String,
        model: String,
        timeout_ms: u64,
        fallback_temperature: f32,
        fallback_max_tokens: usize,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_millis(timeout_ms))
            .build()
            .map_err(|e| LetheError::config(format!("Failed to build HTTP client: {e}")))?;

        Ok(Self {
            base_url,
            model,
            client,
            fallback_temperature,
            fallback_max_tokens,
        })
    }

    async fn check_connectivity(&self) -> Result<bool> {
        let url = format!("{}/api/version", self.base_url);
        let response = self.client.get(url).send().await;

        Ok(matches!(response, Ok(ref resp) if resp.status().is_success()))
    }

    fn build_request_body(&self, prompt: &str, config: &HydeConfig) -> serde_json::Value {
        let temperature = if config.temperature > 0.0 {
            config.temperature
        } else {
            self.fallback_temperature
        };

        let max_tokens = if config.max_tokens > 0 {
            config.max_tokens as i64
        } else {
            self.fallback_max_tokens as i64
        };

        serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        })
    }

    fn split_documents(&self, raw: &str, expected: usize) -> Vec<String> {
        if let Some(parsed) = Self::parse_json_documents(raw, expected) {
            return parsed;
        }

        let mut documents = Vec::new();
        let mut current = String::new();

        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let starts_new_section = trimmed.chars().take_while(|c| c.is_ascii_digit()).count() > 0
                && trimmed.contains('.');

            if starts_new_section && !current.is_empty() {
                documents.push(Self::normalise_document(&current));
                current.clear();
            }

            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(trimmed);
        }

        if !current.is_empty() {
            documents.push(Self::normalise_document(&current));
        }

        if documents.is_empty() {
            documents = raw
                .split("\n\n")
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(Self::normalise_document)
                .collect();
        }

        documents.truncate(expected);
        documents
    }

    fn parse_json_documents(raw: &str, expected: usize) -> Option<Vec<String>> {
        let candidate = raw.trim();
        let json_slice = candidate
            .find('{')
            .and_then(|start| candidate.rfind('}').map(|end| &candidate[start..=end]))
            .unwrap_or(candidate);

        if json_slice.is_empty() {
            return None;
        }

        if let Ok(response) = serde_json::from_str::<StructuredDocumentResponse>(json_slice) {
            let mut documents = response
                .documents
                .into_iter()
                .filter_map(|entry| entry.into_text())
                .map(|text| Self::normalise_document(&text))
                .collect::<Vec<_>>();
            if !documents.is_empty() {
                documents.truncate(expected);
                return Some(documents);
            }
        }

        if let Ok(entries) = serde_json::from_str::<Vec<DocumentEntry>>(json_slice) {
            let mut documents = entries
                .into_iter()
                .filter_map(|entry| entry.into_text())
                .map(|text| Self::normalise_document(&text))
                .collect::<Vec<_>>();
            if !documents.is_empty() {
                documents.truncate(expected);
                return Some(documents);
            }
        }

        None
    }

    fn normalise_document(text: &str) -> String {
        let cleaned = text
            .trim()
            .trim_start_matches(|c: char| {
                c.is_ascii_digit() || c == '.' || c == ')' || c == ':' || c.is_whitespace()
            })
            .trim();

        cleaned.to_string()
    }
}

#[async_trait]
impl LlmService for OllamaLlmService {
    async fn generate_text(&self, prompt: &str, config: &HydeConfig) -> Result<Vec<String>> {
        let body = self.build_request_body(prompt, config);
        let url = format!("{}/api/generate", self.base_url);

        let response = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await
            .map_err(|e| LetheError::external_service("ollama", format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(LetheError::external_service(
                "ollama",
                format!("Generation failed with status {}", response.status()),
            ));
        }

        let payload: OllamaResponse = response.json().await.map_err(|e| {
            LetheError::external_service("ollama", format!("Invalid response: {e}"))
        })?;

        let raw = payload.response.trim();
        if raw.is_empty() {
            return Err(LetheError::external_service(
                "ollama",
                "Empty response from LLM",
            ));
        }

        let mut documents = self.split_documents(raw, config.num_documents);
        if documents.is_empty() {
            documents.push(raw.to_string());
        }

        Ok(documents)
    }
}

struct OllamaLlmProvider;

#[async_trait]
impl LlmProviderBuilder for OllamaLlmProvider {
    fn id(&self) -> &'static str {
        "ollama"
    }

    fn matches(&self, provider: &LlmProviderConfig) -> bool {
        matches!(provider, LlmProviderConfig::Ollama { .. })
    }

    async fn build(
        &self,
        config: &LlmServiceConfig,
        provider: &LlmProviderConfig,
    ) -> Result<Arc<dyn LlmService>> {
        if let LlmProviderConfig::Ollama { base_url, model } = provider {
            let service = OllamaLlmService::new(
                base_url.clone(),
                model.clone(),
                config.timeout_ms,
                config.temperature,
                config.max_tokens,
            )?;

            if !service.check_connectivity().await? {
                tracing::warn!(
                    "Ollama service not reachable during initialisation; continuing with lazy attempts"
                );
            }

            Ok(Arc::new(service))
        } else {
            Err(LetheError::config(
                "Ollama LLM provider received incompatible configuration",
            ))
        }
    }
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    response: String,
}

#[derive(Debug, Deserialize)]
struct StructuredDocumentResponse {
    documents: Vec<DocumentEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DocumentEntry {
    Text(String),
    Object { _id: Option<String>, text: String },
}

impl DocumentEntry {
    fn into_text(self) -> Option<String> {
        match self {
            DocumentEntry::Text(text) => Some(text),
            DocumentEntry::Object { text, .. } => Some(text),
        }
    }
}
