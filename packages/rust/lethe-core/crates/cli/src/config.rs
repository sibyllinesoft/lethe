use lethe_shared::{LetheConfig, Result};
use std::path::{Path, PathBuf};

/// Loaded configuration alongside its source path (if any)
pub struct LoadedConfig {
    pub config: LetheConfig,
    pub path: Option<PathBuf>,
}

/// Load configuration from file or use defaults
pub async fn load_config(config_path: Option<&Path>) -> Result<LoadedConfig> {
    let resolved_path = if let Some(path) = config_path {
        Some(path.to_path_buf())
    } else {
        let default_paths = ["lethe.json", "lethe.yaml", "lethe.yml", "lethe.toml"];
        default_paths
            .iter()
            .map(PathBuf::from)
            .find(|candidate| candidate.exists())
    };

    if let Some(path) = resolved_path {
        tracing::info!("Loading configuration from: {}", path.display());

        if !path.exists() {
            return Err(lethe_shared::LetheError::config(format!(
                "Configuration file not found: {}",
                path.display()
            )));
        }

        let content = tokio::fs::read_to_string(&path).await?;

        let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

        let config: LetheConfig = match extension {
            "json" => serde_json::from_str(&content)?,
            "yaml" | "yml" => serde_yaml::from_str(&content).map_err(|e| {
                lethe_shared::LetheError::config(format!(
                    "Failed to parse YAML configuration: {}",
                    e
                ))
            })?,
            "toml" => toml::from_str(&content).map_err(|e| {
                lethe_shared::LetheError::config(format!(
                    "Failed to parse TOML configuration: {}",
                    e
                ))
            })?,
            _ => {
                if content.trim_start().starts_with('{') {
                    serde_json::from_str(&content)?
                } else if content.contains("---") || content.contains(":") {
                    serde_yaml::from_str(&content).map_err(|e| {
                        lethe_shared::LetheError::config(format!(
                            "Failed to parse configuration as YAML: {}",
                            e
                        ))
                    })?
                } else {
                    return Err(lethe_shared::LetheError::config(
                        "Unknown configuration file format. Use .json, .yaml, or .toml",
                    ));
                }
            }
        };

        Ok(LoadedConfig {
            config,
            path: Some(path),
        })
    } else {
        tracing::info!("No configuration file found, using defaults");
        Ok(LoadedConfig {
            config: LetheConfig::default(),
            path: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_load_json_config() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let mut config_struct = LetheConfig::default();
        config_struct.storage.index_root = "./storage-test".to_string();
        config_struct.embedding.provider = lethe_shared::config::EmbeddingProvider::Fallback;
        let json = serde_json::to_string_pretty(&config_struct).unwrap();
        write!(temp_file, "{}", json).unwrap();

        let loaded = load_config(Some(temp_file.path())).await.unwrap();
        assert_eq!(loaded.config.storage.index_root, "./storage-test");
        assert!(loaded.path.is_some());
    }

    #[tokio::test]
    async fn test_load_yaml_config() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let mut config_struct = LetheConfig::default();
        config_struct.storage.index_root = "./storage-test".to_string();
        config_struct.embedding.provider = lethe_shared::config::EmbeddingProvider::Fallback;
        let yaml = serde_yaml::to_string(&config_struct).unwrap();
        write!(temp_file, "{}", yaml).unwrap();

        let loaded = load_config(Some(temp_file.path())).await.unwrap();
        assert_eq!(loaded.config.storage.index_root, "./storage-test");
        assert!(loaded.path.is_some());
    }

    #[tokio::test]
    async fn test_load_default_config() {
        let loaded = load_config(None).await.unwrap();
        assert!(loaded.path.is_none());
        assert!(!loaded.config.storage.index_root.is_empty());
    }

    #[tokio::test]
    async fn test_nonexistent_config() {
        let result = load_config(Some(Path::new("nonexistent.json"))).await;
        assert!(result.is_err());
    }
}
