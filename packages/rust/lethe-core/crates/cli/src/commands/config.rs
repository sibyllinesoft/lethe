use super::Command;
use crate::utils::{AppContext, OutputFormat};
use async_trait::async_trait;
use clap::{Args, Subcommand};
use lethe_shared::{LetheError, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Args)]
pub struct ConfigCommand {
    #[command(subcommand)]
    action: ConfigAction,
}

#[derive(Debug, Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Validate configuration integrity
    Validate,
    /// Set configuration value
    Set {
        /// Configuration key (dot notation)
        key: String,
        /// Configuration value (parsed as JSON when possible)
        value: String,
    },
    /// Get configuration value
    Get {
        /// Configuration key (dot notation)
        key: String,
    },
}

#[async_trait]
impl Command for ConfigCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        match &self.action {
            ConfigAction::Show => self.show(context),
            ConfigAction::Validate => self.validate(context),
            ConfigAction::Set { key, value } => self.set(context, key, value).await,
            ConfigAction::Get { key } => self.get(context, key),
        }
    }
}

impl ConfigCommand {
    fn show(&self, context: &AppContext) -> Result<()> {
        match context.output_format {
            OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&context.config)?);
            }
            OutputFormat::Yaml => {
                let yaml = serde_yaml::to_string(&context.config).map_err(|e| {
                    LetheError::config(format!("Failed to serialize configuration: {}", e))
                })?;
                println!("{}", yaml);
            }
            OutputFormat::Pretty | OutputFormat::Table => {
                println!("⚙️  Configuration Summary:");
                println!("   Storage root: {}", context.storage_root.display());
                println!(
                    "   Embedding Provider: {:?}",
                    context.config.embedding.provider
                );
                if let Some(features) = &context.config.features {
                    println!("   Features:");
                    println!("     HyDE: {}", features.enable_hyde);
                    println!(
                        "     Query understanding: {}",
                        features.enable_query_understanding
                    );
                    println!("     ML prediction: {}", features.enable_ml_prediction);
                    println!("     State tracking: {}", features.enable_state_tracking);
                }
                println!(
                    "   Retrieval alpha: {:.2}",
                    context.config.retrieval.alpha.value()
                );
                println!(
                    "   Retrieval beta: {:.2}",
                    context.config.retrieval.beta.value()
                );
                println!(
                    "   Chunk target tokens: {}",
                    context.config.chunking.target_tokens.value()
                );
                println!("   Chunk overlap: {}", context.config.chunking.overlap);
            }
        }

        Ok(())
    }

    fn validate(&self, context: &AppContext) -> Result<()> {
        context.config.validate()?;
        if !context.quiet {
            println!("✅ Configuration is valid");
        }
        Ok(())
    }

    async fn set(&self, context: &AppContext, key: &str, value: &str) -> Result<()> {
        let mut json = serde_json::to_value(&context.config)?;
        let new_value = Self::parse_user_value(value);
        Self::set_path_value(&mut json, key, new_value)?;

        let updated: lethe_shared::LetheConfig = serde_json::from_value(json)?;
        updated.validate()?;

        let target_path = context
            .config_path
            .clone()
            .unwrap_or_else(|| PathBuf::from("lethe.json"));

        if let Some(parent) = target_path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }

        let payload = Self::serialize_config(&updated, &target_path)?;
        tokio::fs::write(&target_path, payload).await?;

        if !context.quiet {
            println!(
                "✅ Updated '{}' and saved changes to {}",
                key,
                target_path.display()
            );
        }

        Ok(())
    }

    fn get(&self, context: &AppContext, key: &str) -> Result<()> {
        let json = serde_json::to_value(&context.config)?;
        if let Some(value) = Self::get_path_value(&json, key) {
            match context.output_format {
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(value)?);
                }
                OutputFormat::Yaml => {
                    let yaml = serde_yaml::to_string(value).map_err(|e| {
                        LetheError::config(format!("Failed to render value as YAML: {}", e))
                    })?;
                    println!("{}", yaml);
                }
                OutputFormat::Pretty | OutputFormat::Table => {
                    println!("{}", Self::value_to_string(value));
                }
            }
            Ok(())
        } else {
            Err(LetheError::config(format!(
                "Configuration key '{}' not found",
                key
            )))
        }
    }

    fn parse_user_value(input: &str) -> Value {
        serde_json::from_str(input).unwrap_or_else(|_| Value::String(input.to_string()))
    }

    fn set_path_value(target: &mut Value, path: &str, new_value: Value) -> Result<()> {
        let parts: Vec<&str> = path.split('.').collect();
        if parts.is_empty() {
            return Err(LetheError::config("Configuration key cannot be empty"));
        }

        let mut current = target;
        for part in &parts[..parts.len() - 1] {
            current = current.get_mut(*part).ok_or_else(|| {
                LetheError::config(format!("Configuration path '{}' not found", path))
            })?;
            if !current.is_object() {
                return Err(LetheError::config(format!(
                    "Configuration path '{}' is not an object",
                    path
                )));
            }
        }

        let last = parts.last().unwrap();
        let obj = current.as_object_mut().ok_or_else(|| {
            LetheError::config(format!("Configuration path '{}' is not an object", path))
        })?;
        obj.insert(last.to_string(), new_value);
        Ok(())
    }

    fn get_path_value<'a>(target: &'a Value, path: &str) -> Option<&'a Value> {
        let mut current = target;
        for part in path.split('.') {
            current = current.get(part)?;
        }
        Some(current)
    }

    fn serialize_config(config: &lethe_shared::LetheConfig, path: &Path) -> Result<String> {
        match path.extension().and_then(|ext| ext.to_str()).unwrap_or("") {
            "yaml" | "yml" => serde_yaml::to_string(config)
                .map_err(|e| LetheError::config(format!("Failed to serialize YAML: {}", e))),
            "toml" => toml::to_string_pretty(config)
                .map_err(|e| LetheError::config(format!("Failed to serialize TOML: {}", e))),
            _ => Ok(serde_json::to_string_pretty(config)?),
        }
    }

    fn value_to_string(value: &Value) -> String {
        match value {
            Value::Null => "null".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) => n.to_string(),
            Value::String(s) => s.clone(),
            _ => serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string()),
        }
    }
}
