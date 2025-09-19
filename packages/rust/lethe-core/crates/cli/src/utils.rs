use lethe_shared::LetheConfig;
use std::path::PathBuf;

/// Application context shared across all CLI commands
#[derive(Debug, Clone)]
pub struct AppContext {
    /// Loaded configuration
    pub config: LetheConfig,
    /// Path to the configuration file (if it exists)
    pub config_path: Option<PathBuf>,
    /// Root directory for storage/index assets
    pub storage_root: PathBuf,
    /// Output format for command results
    pub output_format: OutputFormat,
    /// Whether to suppress non-essential output
    pub quiet: bool,
}

/// Output format options for CLI commands
#[derive(Debug, Clone)]
pub enum OutputFormat {
    /// Tabular format for human reading
    Table,
    /// JSON format for programmatic use
    Json,
    /// YAML format for configuration
    Yaml,
    /// Pretty-printed format with colors and emojis
    Pretty,
}

impl From<crate::OutputFormat> for OutputFormat {
    fn from(format: crate::OutputFormat) -> Self {
        match format {
            crate::OutputFormat::Table => OutputFormat::Table,
            crate::OutputFormat::Json => OutputFormat::Json,
            crate::OutputFormat::Yaml => OutputFormat::Yaml,
            crate::OutputFormat::Pretty => OutputFormat::Pretty,
        }
    }
}

/// Progress indicator for long-running operations
#[allow(dead_code)]
pub struct ProgressIndicator {
    pb: Option<indicatif::ProgressBar>,
    quiet: bool,
}

#[allow(dead_code)]
impl ProgressIndicator {
    pub fn new(total: u64, message: &str, quiet: bool) -> Self {
        let pb = if quiet {
            None
        } else {
            let pb = indicatif::ProgressBar::new(total);
            pb.set_style(
                indicatif::ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:.cyan/blue}] {pos:>7}/{len:7} {msg}")
                    .expect("Failed to set progress bar template")
                    .progress_chars("#>-"),
            );
            pb.set_message(message.to_string());
            Some(pb)
        };

        Self { pb, quiet }
    }

    pub fn inc(&self, delta: u64) {
        if let Some(ref pb) = self.pb {
            pb.inc(delta);
        }
    }

    pub fn set_position(&self, pos: u64) {
        if let Some(ref pb) = self.pb {
            pb.set_position(pos);
        }
    }

    pub fn set_message(&self, message: &str) {
        if let Some(ref pb) = self.pb {
            pb.set_message(message.to_string());
        }
    }

    pub fn finish_with_message(&self, message: &str) {
        if let Some(ref pb) = self.pb {
            pb.finish_with_message(message.to_string());
        } else if !self.quiet {
            println!("{}", message);
        }
    }
}

/// Utility functions for CLI operations
pub mod helpers {

    /// Format duration in a human-readable way
    #[allow(dead_code)]
    pub fn format_duration(duration: std::time::Duration) -> String {
        let total_secs = duration.as_secs();
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;
        let millis = duration.subsec_millis();

        if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else if seconds > 0 {
            format!("{}.{}s", seconds, millis / 100)
        } else {
            format!("{}ms", millis)
        }
    }

    /// Format file size in human-readable way
    #[allow(dead_code)]
    pub fn format_file_size(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.1} {}", size, UNITS[unit_index])
        }
    }

    /// Truncate text to a maximum length with ellipsis
    pub fn truncate_text(text: &str, max_len: usize) -> String {
        if text.len() <= max_len {
            text.to_string()
        } else {
            format!("{}...", &text[..max_len.saturating_sub(3)])
        }
    }

    /// Validate UUID string format
    #[allow(dead_code)]
    pub fn validate_uuid(uuid_str: &str) -> Result<uuid::Uuid, String> {
        uuid::Uuid::parse_str(uuid_str).map_err(|e| format!("Invalid UUID '{}': {}", uuid_str, e))
    }

    /// Get terminal width for formatting
    #[allow(dead_code)]
    pub fn terminal_width() -> usize {
        terminal_size::terminal_size()
            .map(|(w, _)| w.0 as usize)
            .unwrap_or(80)
    }
}

#[cfg(test)]
mod tests {
    use super::helpers::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(
            format_duration(std::time::Duration::from_millis(500)),
            "500ms"
        );
        assert_eq!(format_duration(std::time::Duration::from_secs(1)), "1.0s");
        assert_eq!(format_duration(std::time::Duration::from_secs(65)), "1m 5s");
        assert_eq!(
            format_duration(std::time::Duration::from_secs(3665)),
            "1h 1m 5s"
        );
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(512), "512 B");
        assert_eq!(format_file_size(1024), "1.0 KB");
        assert_eq!(format_file_size(1536), "1.5 KB");
        assert_eq!(format_file_size(1024 * 1024), "1.0 MB");
    }

    #[test]
    fn test_truncate_text() {
        assert_eq!(truncate_text("hello", 10), "hello");
        assert_eq!(truncate_text("hello world", 8), "hello...");
        assert_eq!(truncate_text("hi", 8), "hi");
    }

    #[test]
    fn test_validate_uuid() {
        assert!(validate_uuid("550e8400-e29b-41d4-a716-446655440000").is_ok());
        assert!(validate_uuid("invalid-uuid").is_err());
        assert!(validate_uuid("").is_err());
    }
}
