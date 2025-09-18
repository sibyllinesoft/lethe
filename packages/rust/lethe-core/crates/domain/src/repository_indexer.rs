use async_trait::async_trait;
use chrono::Utc;
use lethe_shared::utils::{HashUtils, TextProcessor, TokenCounter};
use lethe_shared::{
    Chunk, LetheError, Message, RepositoryConfig, RepositoryPreloadingConfig, Result,
};
use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use uuid::Uuid;
use walkdir::WalkDir;

/// Simple chunking configuration for repository indexing
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    pub target_tokens: i32,
    pub overlap: i32,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            target_tokens: 320,
            overlap: 64,
        }
    }
}

/// Repository indexing results
#[derive(Debug, Clone)]
pub struct IndexingResult {
    pub repository_path: String,
    pub total_files: usize,
    pub indexed_files: usize,
    pub total_chunks: usize,
    pub skipped_files: usize,
    pub errors: Vec<IndexingError>,
    pub duration_ms: u64,
}

/// Repository indexing error
#[derive(Debug, Clone)]
pub struct IndexingError {
    pub file_path: String,
    pub error: String,
}

/// Repository trait for chunk persistence
#[async_trait]
pub trait ChunkRepository: Send + Sync {
    async fn create_chunk(&self, chunk: &Chunk) -> Result<Chunk>;
    async fn batch_create_chunks(&self, chunks: &[Chunk]) -> Result<Vec<Chunk>>;
    async fn get_chunks_by_session(&self, session_id: &str) -> Result<Vec<Chunk>>;
    async fn delete_chunks_by_session(&self, session_id: &str) -> Result<bool>;
}

/// Repository trait for message persistence
#[async_trait]
pub trait MessageRepository: Send + Sync {
    async fn create_message(&self, message: &Message) -> Result<Message>;
    async fn get_message(&self, id: &Uuid) -> Result<Option<Message>>;
    async fn delete_message(&self, id: &Uuid) -> Result<bool>;
}

/// Service for indexing repositories and creating chunks
pub struct RepositoryIndexer {
    chunking_config: ChunkingConfig,
    file_patterns: Vec<Regex>,
    exclude_patterns: Vec<Regex>,
}

impl RepositoryIndexer {
    /// Create a new repository indexer
    pub fn new(
        chunking_config: ChunkingConfig,
        file_patterns: Vec<String>,
        exclude_patterns: Vec<String>,
    ) -> Result<Self> {
        let mut file_regex_patterns = Vec::new();
        for pattern in file_patterns {
            let regex = Self::glob_to_regex(&pattern).map_err(|e| {
                LetheError::config(format!("Invalid file pattern '{}': {}", pattern, e))
            })?;
            file_regex_patterns.push(regex);
        }

        let mut exclude_regex_patterns = Vec::new();
        for pattern in exclude_patterns {
            let regex = Self::glob_to_regex(&pattern).map_err(|e| {
                LetheError::config(format!("Invalid exclude pattern '{}': {}", pattern, e))
            })?;
            exclude_regex_patterns.push(regex);
        }

        Ok(Self {
            chunking_config,
            file_patterns: file_regex_patterns,
            exclude_patterns: exclude_regex_patterns,
        })
    }

    /// Index a single repository
    pub async fn index_repository<CR, MR>(
        &self,
        repo_config: &RepositoryConfig,
        chunk_repo: &CR,
        message_repo: &MR,
    ) -> Result<IndexingResult>
    where
        CR: ChunkRepository,
        MR: MessageRepository,
    {
        let start_time = std::time::Instant::now();
        let repo_path = Path::new(&repo_config.path);

        tracing::info!(
            path = %repo_config.path,
            name = ?repo_config.name,
            "Starting repository indexing"
        );

        if !repo_path.exists() {
            return Err(LetheError::config(format!(
                "Repository path does not exist: {}",
                repo_config.path
            )));
        }

        // Generate session ID for this repository
        let session_id = format!(
            "repo_{}_{}",
            repo_config.name.as_deref().unwrap_or("unnamed"),
            HashUtils::short_hash(&repo_config.path)
        );

        // Clean up any existing chunks for this repository
        let _ = chunk_repo.delete_chunks_by_session(&session_id).await;

        let mut result = IndexingResult {
            repository_path: repo_config.path.clone(),
            total_files: 0,
            indexed_files: 0,
            total_chunks: 0,
            skipped_files: 0,
            errors: Vec::new(),
            duration_ms: 0,
        };

        // Get file patterns for this repository
        let empty_patterns = vec![];
        let file_patterns = repo_config
            .file_patterns
            .as_ref()
            .unwrap_or(&empty_patterns);
        let exclude_patterns = repo_config
            .exclude_patterns
            .as_ref()
            .unwrap_or(&empty_patterns);

        // Collect all files to process
        let files: Vec<PathBuf> = WalkDir::new(repo_path)
            .into_iter()
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();

                if !path.is_file() {
                    return None;
                }

                result.total_files += 1;

                // Check exclusion patterns first
                if self.should_exclude_file(path, exclude_patterns) {
                    result.skipped_files += 1;
                    return None;
                }

                // Check inclusion patterns
                if !self.should_include_file(path, file_patterns) {
                    result.skipped_files += 1;
                    return None;
                }

                Some(path.to_path_buf())
            })
            .collect();

        tracing::info!(
            total_files = result.total_files,
            files_to_index = files.len(),
            skipped_files = result.skipped_files,
            "Files collected for indexing"
        );

        // Process files and create chunks
        for file_path in files {
            match self
                .process_file(&file_path, &session_id, chunk_repo, message_repo)
                .await
            {
                Ok(chunk_count) => {
                    result.indexed_files += 1;
                    result.total_chunks += chunk_count;
                }
                Err(e) => {
                    let error = IndexingError {
                        file_path: file_path.display().to_string(),
                        error: e.to_string(),
                    };
                    result.errors.push(error);
                    tracing::warn!(
                        file = %file_path.display(),
                        error = %e,
                        "Failed to process file"
                    );
                }
            }
        }

        result.duration_ms = start_time.elapsed().as_millis() as u64;

        tracing::info!(
            indexed_files = result.indexed_files,
            total_chunks = result.total_chunks,
            errors = result.errors.len(),
            duration_ms = result.duration_ms,
            "Repository indexing completed"
        );

        Ok(result)
    }

    /// Process multiple repositories in parallel
    pub async fn index_repositories<CR, MR>(
        &self,
        config: &RepositoryPreloadingConfig,
        chunk_repo: Arc<CR>,
        message_repo: Arc<MR>,
    ) -> Result<Vec<IndexingResult>>
    where
        CR: ChunkRepository + 'static,
        MR: MessageRepository + 'static,
    {
        if !config.enabled {
            return Ok(Vec::new());
        }

        let enabled_repos: Vec<&RepositoryConfig> = config
            .repositories
            .iter()
            .filter(|repo| repo.enabled)
            .collect();

        if enabled_repos.is_empty() {
            tracing::info!("No enabled repositories to index");
            return Ok(Vec::new());
        }

        tracing::info!(
            total_repos = enabled_repos.len(),
            max_concurrent = config.max_concurrent_repos,
            "Starting parallel repository indexing"
        );

        let semaphore = Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_repos));
        let mut handles = Vec::new();

        for repo_config in enabled_repos {
            let permit =
                semaphore.clone().acquire_owned().await.map_err(|e| {
                    LetheError::internal(format!("Failed to acquire semaphore: {}", e))
                })?;

            let repo_config = repo_config.clone();
            let chunk_repo = chunk_repo.clone();
            let message_repo = message_repo.clone();
            let indexer = self.clone();

            let handle = tokio::spawn(async move {
                let _permit = permit; // Hold permit until task completes
                indexer
                    .index_repository(&repo_config, chunk_repo.as_ref(), message_repo.as_ref())
                    .await
            });

            handles.push(handle);
        }

        let mut results = Vec::new();
        let mut has_errors = false;

        for handle in handles {
            match handle.await {
                Ok(Ok(result)) => {
                    if !result.errors.is_empty() {
                        has_errors = true;
                    }
                    results.push(result);
                }
                Ok(Err(e)) => {
                    has_errors = true;
                    tracing::error!(error = %e, "Repository indexing failed");
                    if config.fail_on_error {
                        return Err(e);
                    }
                }
                Err(e) => {
                    has_errors = true;
                    tracing::error!(error = %e, "Repository indexing task panicked");
                    if config.fail_on_error {
                        return Err(LetheError::internal(format!("Indexing task failed: {}", e)));
                    }
                }
            }
        }

        let total_files: usize = results.iter().map(|r| r.indexed_files).sum();
        let total_chunks: usize = results.iter().map(|r| r.total_chunks).sum();
        let total_errors: usize = results.iter().map(|r| r.errors.len()).sum();

        tracing::info!(
            repositories = results.len(),
            total_files,
            total_chunks,
            total_errors,
            "Parallel repository indexing completed"
        );

        if config.fail_on_error && has_errors {
            return Err(LetheError::internal(
                "Repository indexing completed with errors",
            ));
        }

        Ok(results)
    }

    /// Process a single file and create chunks
    async fn process_file<CR, MR>(
        &self,
        file_path: &Path,
        session_id: &str,
        chunk_repo: &CR,
        message_repo: &MR,
    ) -> Result<usize>
    where
        CR: ChunkRepository,
        MR: MessageRepository,
    {
        // Read file content
        let content = fs::read_to_string(file_path).await.map_err(|e| {
            LetheError::internal(format!(
                "Failed to read file {}: {}",
                file_path.display(),
                e
            ))
        })?;

        if content.trim().is_empty() {
            return Ok(0);
        }

        // Determine file kind from extension
        let kind = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| match ext.to_lowercase().as_str() {
                "rs" => "rust",
                "py" => "python",
                "js" => "javascript",
                "ts" => "typescript",
                "java" => "java",
                "cpp" | "cc" | "cxx" => "cpp",
                "c" => "c",
                "h" | "hpp" => "header",
                "md" => "markdown",
                "txt" => "text",
                _ => "text",
            })
            .unwrap_or("text");

        // Create a message for this file
        let message_id = Uuid::new_v4();
        let file_info = serde_json::json!({
            "file_name": file_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown"),
            "file_path": file_path.display().to_string(),
            "kind": kind
        });

        let message = Message {
            id: message_id,
            session_id: session_id.to_string(),
            turn: 0,
            role: "system".to_string(),
            text: content.clone(),
            ts: Utc::now(),
            meta: Some(file_info),
        };

        // Store the message
        let stored_message = message_repo.create_message(&message).await?;

        // Create chunks from the content
        let chunks = self.create_chunks_from_content(&stored_message, &content, kind)?;

        if !chunks.is_empty() {
            chunk_repo.batch_create_chunks(&chunks).await?;
        }

        Ok(chunks.len())
    }

    /// Create chunks from file content
    fn create_chunks_from_content(
        &self,
        message: &Message,
        content: &str,
        kind: &str,
    ) -> Result<Vec<Chunk>> {
        let normalized_text = TextProcessor::normalize_text(content);
        let tokens = TokenCounter::count_tokens(&normalized_text);

        if tokens as i32 <= self.chunking_config.target_tokens {
            // Small file, create single chunk
            let chunk = Chunk {
                id: format!("{}_{}", message.id, 0),
                message_id: message.id,
                session_id: message.session_id.clone(),
                offset_start: 0,
                offset_end: content.len(),
                kind: kind.to_string(),
                text: normalized_text,
                tokens: tokens as i32,
            };
            return Ok(vec![chunk]);
        }

        // Large file, split into chunks
        let target_tokens = self.chunking_config.target_tokens as usize;
        let overlap_tokens = self.chunking_config.overlap as usize;

        let mut chunks = Vec::new();
        let mut start_offset = 0;
        let mut chunk_index = 0;

        while start_offset < content.len() {
            let (chunk_text, end_offset) =
                self.extract_chunk_text(content, start_offset, target_tokens);

            let tokens = TokenCounter::count_tokens(&chunk_text) as i32;
            let chunk = Chunk {
                id: format!("{}_{}", message.id, chunk_index),
                message_id: message.id,
                session_id: message.session_id.clone(),
                offset_start: start_offset,
                offset_end: end_offset,
                kind: kind.to_string(),
                text: chunk_text,
                tokens,
            };

            chunks.push(chunk);

            // Calculate next start position with overlap
            let overlap_chars =
                self.calculate_overlap_chars(content, start_offset, end_offset, overlap_tokens);
            start_offset = end_offset.saturating_sub(overlap_chars);

            if start_offset >= end_offset {
                break;
            }

            chunk_index += 1;
        }

        Ok(chunks)
    }

    /// Extract chunk text with proper token limits
    fn extract_chunk_text(
        &self,
        content: &str,
        start: usize,
        target_tokens: usize,
    ) -> (String, usize) {
        let remaining = &content[start..];
        let estimated_chars = target_tokens * 4; // Rough estimate: 4 chars per token

        if remaining.len() <= estimated_chars {
            return (remaining.to_string(), content.len());
        }

        // Find a good break point (end of line preferred)
        let max_chars = estimated_chars.min(remaining.len());
        let potential_text = &remaining[..max_chars];

        if let Some(last_newline) = potential_text.rfind('\n') {
            let break_point = start + last_newline + 1;
            (content[start..break_point].to_string(), break_point)
        } else {
            let break_point = start + max_chars;
            (content[start..break_point].to_string(), break_point)
        }
    }

    /// Calculate overlap characters
    fn calculate_overlap_chars(
        &self,
        _content: &str,
        start: usize,
        end: usize,
        overlap_tokens: usize,
    ) -> usize {
        let overlap_chars = overlap_tokens * 4; // Rough estimate
        let chunk_size = end - start;
        overlap_chars.min(chunk_size / 2).min(end)
    }

    /// Check if file should be included based on patterns
    fn should_include_file(&self, path: &Path, file_patterns: &[String]) -> bool {
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // If no specific patterns for this repo, use global patterns
        let patterns = if file_patterns.is_empty() {
            &self.file_patterns
        } else {
            // Convert repo-specific patterns to regex if needed
            &self.file_patterns // For now, use global patterns
        };

        if patterns.is_empty() {
            return true; // Include all files if no patterns
        }

        patterns.iter().any(|pattern| pattern.is_match(file_name))
    }

    /// Check if file should be excluded based on patterns
    fn should_exclude_file(&self, path: &Path, exclude_patterns: &[String]) -> bool {
        let path_str = path.to_string_lossy();

        // Use repo-specific exclude patterns if available, otherwise global
        let patterns = if exclude_patterns.is_empty() {
            &self.exclude_patterns
        } else {
            &self.exclude_patterns // For now, use global patterns
        };

        patterns.iter().any(|pattern| pattern.is_match(&path_str))
    }

    /// Convert glob pattern to regex
    fn glob_to_regex(pattern: &str) -> std::result::Result<Regex, regex::Error> {
        let mut regex_pattern = String::new();
        let mut chars = pattern.chars().peekable();

        while let Some(ch) = chars.next() {
            match ch {
                '*' => {
                    if chars.peek() == Some(&'*') {
                        chars.next(); // consume second *
                        if chars.peek() == Some(&'/') {
                            chars.next(); // consume /
                            regex_pattern.push_str(".*"); // **/ matches any path
                        } else {
                            regex_pattern.push_str(".*"); // ** at end
                        }
                    } else {
                        regex_pattern.push_str("[^/]*"); // * matches anything except /
                    }
                }
                '?' => regex_pattern.push('.'),
                '.' => regex_pattern.push_str("\\."),
                '+' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\' => {
                    regex_pattern.push('\\');
                    regex_pattern.push(ch);
                }
                _ => regex_pattern.push(ch),
            }
        }

        Regex::new(&format!("^{}$", regex_pattern))
    }
}

impl Clone for RepositoryIndexer {
    fn clone(&self) -> Self {
        Self {
            chunking_config: self.chunking_config.clone(),
            file_patterns: self.file_patterns.clone(),
            exclude_patterns: self.exclude_patterns.clone(),
        }
    }
}

/// Factory for creating repository indexers
pub struct RepositoryIndexerFactory;

impl RepositoryIndexerFactory {
    /// Create a repository indexer from configuration
    pub fn create_indexer(
        chunking_config: ChunkingConfig,
        preloading_config: &RepositoryPreloadingConfig,
    ) -> Result<RepositoryIndexer> {
        RepositoryIndexer::new(
            chunking_config,
            preloading_config.file_patterns.clone(),
            preloading_config.exclude_patterns.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_to_regex() {
        let regex = RepositoryIndexer::glob_to_regex("*.rs").unwrap();
        assert!(regex.is_match("main.rs"));
        assert!(regex.is_match("lib.rs"));
        assert!(!regex.is_match("main.py"));
        assert!(!regex.is_match("src/main.rs"));

        let regex = RepositoryIndexer::glob_to_regex("**/*.rs").unwrap();
        assert!(regex.is_match("main.rs"));
        assert!(regex.is_match("src/main.rs"));
        assert!(regex.is_match("src/lib/test.rs"));
        assert!(!regex.is_match("main.py"));
    }

    #[test]
    fn test_chunk_creation() {
        let config = ChunkingConfig {
            target_tokens: 100,
            overlap: 20,
        };

        let indexer = RepositoryIndexer::new(
            config,
            vec!["*.rs".to_string()],
            vec!["target/**".to_string()],
        )
        .unwrap();

        let message = Message {
            id: Uuid::new_v4(),
            session_id: "test_session".to_string(),
            turn: 0,
            role: "system".to_string(),
            text: "This is a test content.".to_string(),
            ts: Utc::now(),
            meta: Some(serde_json::json!({
                "file_name": "test.rs",
                "file_path": "/test.rs"
            })),
        };

        let chunks = indexer
            .create_chunks_from_content(&message, "This is a test content.", "rust")
            .unwrap();

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].kind, "rust");
        assert_eq!(chunks[0].message_id, message.id);
    }
}
