use lethe_shared::{LetheConfig, Result};
use std::path::Path;

/// Load configuration from file or use defaults
pub async fn load_config(config_path: Option<&Path>) -> Result<LetheConfig> {
    if let Some(path) = config_path {
        tracing::info!("Loading configuration from: {}", path.display());

        if !path.exists() {
            return Err(lethe_shared::LetheError::config(format!(
                "Configuration file not found: {}",
                path.display()
            )));
        }

        let content = tokio::fs::read_to_string(path).await?;

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
                // Try to auto-detect format based on content
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

        Ok(config)
    } else {
        // Check for default configuration files
        let default_paths = ["lethe.json", "lethe.yaml", "lethe.yml", "lethe.toml"];

        for default_path in &default_paths {
            if Path::new(default_path).exists() {
                tracing::info!("Found default configuration: {}", default_path);
                return Box::pin(load_config(Some(Path::new(default_path)))).await;
            }
        }

        tracing::info!("No configuration file found, using defaults");
        Ok(LetheConfig::default())
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
        writeln!(
            temp_file,
            r#"{{
                "database": {{
                    "url": "postgresql://localhost/test"
                }},
                "embedding": {{
                    "provider": "fallback"
                }}
            }}"#
        )
        .unwrap();

        let config = load_config(Some(temp_file.path())).await.unwrap();
        assert_eq!(config.database.url, "postgresql://localhost/test");
    }

    #[tokio::test]
    async fn test_load_yaml_config() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"
database:
  url: "postgresql://localhost/test"
embedding:
  provider: "fallback"
            "#
        )
        .unwrap();

        let config = load_config(Some(temp_file.path())).await.unwrap();
        assert_eq!(config.database.url, "postgresql://localhost/test");
    }

    #[tokio::test]
    async fn test_load_default_config() {
        let config = load_config(None).await.unwrap();
        assert!(!config.database.url.is_empty());
    }

    #[tokio::test]
    async fn test_nonexistent_config() {
        let result = load_config(Some(Path::new("nonexistent.json"))).await;
        assert!(result.is_err());
    }
}
