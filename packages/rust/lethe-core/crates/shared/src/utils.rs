use regex::Regex;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::sync::OnceLock;

/// Pre-compiled regexes for performance
struct CompiledRegexes {
    alphanumeric: Regex,
    punctuation: Regex,
    sentence_split: Regex,
    code_fence: Regex,
    word_boundary: Regex,
    code_symbol: Regex,
    error_token: Regex,
    path_file: Regex,
    numeric_id: Regex,
}

impl CompiledRegexes {
    fn new() -> Self {
        Self {
            alphanumeric: Regex::new(r"[a-zA-Z0-9]+").unwrap(),
            punctuation: Regex::new(r"[^\w\s]").unwrap(),
            sentence_split: Regex::new(r"[.!?]\s+").unwrap(),
            code_fence: Regex::new(r"```[\s\S]*?```").unwrap(),
            word_boundary: Regex::new(r"\b\w+\b").unwrap(),
            code_symbol: Regex::new(r"[_a-zA-Z][\w]*\(|\b[A-Z][A-Za-z0-9]+::[A-Za-z0-9]+\b")
                .unwrap(),
            error_token: Regex::new(r"(?i)(Exception|Error|stack trace|errno|\bE\d{2,}\b)")
                .unwrap(),
            path_file: Regex::new(r"/[^\s]+\.[a-zA-Z0-9]+|[A-Za-z]:\\[^\s]+\.[a-zA-Z0-9]+")
                .unwrap(),
            numeric_id: Regex::new(r"\b\d{3,}\b").unwrap(),
        }
    }
}

/// Global regex cache to avoid repeated compilation
static REGEX_CACHE: OnceLock<CompiledRegexes> = OnceLock::new();

fn get_regex_cache() -> &'static CompiledRegexes {
    REGEX_CACHE.get_or_init(CompiledRegexes::new)
}

/// Token counting utilities
pub struct TokenCounter;

impl TokenCounter {
    /// Count tokens in text using GPT-style approximation
    /// This provides a rough estimate - for actual tokenization, use a proper tokenizer
    pub fn count_tokens(text: &str) -> i32 {
        if text.is_empty() {
            return 0;
        }

        Self::count_tokens_detailed(text).total_tokens
    }

    /// Count tokens with detailed breakdown for debugging
    pub fn count_tokens_detailed(text: &str) -> TokenCounts {
        if text.is_empty() {
            return TokenCounts::default();
        }

        let regex_cache = get_regex_cache();
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return TokenCounts::default();
        }

        let mut alphanumeric_tokens = 0;
        let mut punctuation_tokens = 0;

        for word in &words {
            // Count alphanumeric sequences
            alphanumeric_tokens += regex_cache.alphanumeric.find_iter(word).count() as i32;

            // Count punctuation separately
            punctuation_tokens += regex_cache.punctuation.find_iter(word).count() as i32;
        }

        // Count whitespace between words (words.len() - 1 spaces)
        let whitespace_tokens = if words.len() > 1 {
            (words.len() - 1) as i32
        } else {
            0
        };

        // Total approximation: alphanumeric + punctuation/2 + whitespace
        let total_tokens = alphanumeric_tokens + (punctuation_tokens + 1) / 2 + whitespace_tokens;

        TokenCounts {
            alphanumeric_tokens,
            punctuation_tokens,
            whitespace_tokens,
            total_tokens: std::cmp::max(1, total_tokens),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct TokenCounts {
    pub alphanumeric_tokens: i32,
    pub punctuation_tokens: i32,
    pub whitespace_tokens: i32,
    pub total_tokens: i32,
}

/// Configuration options for sentence splitting
#[derive(Debug, Clone)]
pub struct SentenceSplitOptions {
    pub min_sentence_length: usize,
    pub min_word_length: usize,
    pub fallback_to_words: bool,
}

impl Default for SentenceSplitOptions {
    fn default() -> Self {
        Self {
            min_sentence_length: 1,
            min_word_length: 1,
            fallback_to_words: false,
        }
    }
}

/// Configuration options for code fence extraction
#[derive(Debug, Clone)]
pub struct CodeFenceOptions {
    pub skip_empty_text: bool,
    pub min_code_length: usize,
}

impl Default for CodeFenceOptions {
    fn default() -> Self {
        Self {
            skip_empty_text: true,
            min_code_length: 6, // Minimum "```x```" length
        }
    }
}

/// Configuration options for tokenization
#[derive(Debug, Clone)]
pub struct TokenizeOptions {
    pub min_word_length: usize,
    pub to_lowercase: bool,
}

impl Default for TokenizeOptions {
    fn default() -> Self {
        Self {
            min_word_length: 2,
            to_lowercase: true,
        }
    }
}

/// Text processing utilities
pub struct TextProcessor;

impl TextProcessor {
    /// Split text into sentences with fallback to words
    pub fn split_sentences(text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        Self::split_sentences_advanced(text, SentenceSplitOptions::default())
    }

    /// Split sentences with configurable options
    pub fn split_sentences_advanced(text: &str, options: SentenceSplitOptions) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        let regex_cache = get_regex_cache();
        let mut sentences = Vec::new();
        let mut current_start = 0;

        for mat in regex_cache.sentence_split.find_iter(text) {
            let end = mat.start() + 1; // Include the punctuation
            let sentence = text[current_start..end].trim();
            if !sentence.is_empty() && sentence.len() >= options.min_sentence_length {
                sentences.push(sentence.to_string());
            }
            current_start = mat.end();
        }

        // Add the remaining text if any
        if current_start < text.len() {
            let sentence = text[current_start..].trim();
            if !sentence.is_empty() && sentence.len() >= options.min_sentence_length {
                sentences.push(sentence.to_string());
            }
        }

        // Fallback to word splitting if no sentences or if explicitly requested
        if (sentences.len() <= 1 && !text.contains(['.', '!', '?'])) || options.fallback_to_words {
            return text
                .split_whitespace()
                .map(|w| w.to_string())
                .filter(|w| !w.is_empty() && w.len() >= options.min_word_length)
                .collect();
        }

        sentences
    }

    /// Extract code fences and text parts with better error handling
    pub fn extract_code_fences(text: &str) -> Vec<TextPart> {
        if text.is_empty() {
            return vec![TextPart {
                kind: TextPartKind::Text,
                content: String::new(),
                start: 0,
                end: 0,
            }];
        }

        Self::extract_code_fences_with_options(text, CodeFenceOptions::default())
    }

    /// Extract code fences with configurable options
    pub fn extract_code_fences_with_options(
        text: &str,
        options: CodeFenceOptions,
    ) -> Vec<TextPart> {
        let mut parts = Vec::new();
        let regex_cache = get_regex_cache();
        let mut last_end = 0;

        for mat in regex_cache.code_fence.find_iter(text) {
            // Add text before code block
            if mat.start() > last_end {
                let text_content = &text[last_end..mat.start()];
                if !text_content.trim().is_empty() || !options.skip_empty_text {
                    parts.push(TextPart {
                        kind: TextPartKind::Text,
                        content: text_content.to_string(),
                        start: last_end,
                        end: mat.start(),
                    });
                }
            }

            // Add code block
            let code_content = mat.as_str();
            if code_content.len() >= options.min_code_length {
                parts.push(TextPart {
                    kind: TextPartKind::Code,
                    content: code_content.to_string(),
                    start: mat.start(),
                    end: mat.end(),
                });
            }

            last_end = mat.end();
        }

        // Add remaining text
        if last_end < text.len() {
            let text_content = &text[last_end..];
            if !text_content.trim().is_empty() || !options.skip_empty_text {
                parts.push(TextPart {
                    kind: TextPartKind::Text,
                    content: text_content.to_string(),
                    start: last_end,
                    end: text.len(),
                });
            }
        }

        // If no parts found, treat as single text part
        if parts.is_empty() {
            parts.push(TextPart {
                kind: TextPartKind::Text,
                content: text.to_string(),
                start: 0,
                end: text.len(),
            });
        }

        parts
    }

    /// Normalize text to NFC form
    pub fn normalize_text(text: &str) -> String {
        // Rust's String is already UTF-8, but we can apply basic normalization
        text.chars().collect::<String>()
    }

    /// Tokenize text for search (similar to TF-IDF processing) with better performance
    pub fn tokenize(text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        Self::tokenize_with_options(text, TokenizeOptions::default())
    }

    /// Tokenize with configurable options
    pub fn tokenize_with_options(text: &str, options: TokenizeOptions) -> Vec<String> {
        let regex_cache = get_regex_cache();
        let text_to_process = if options.to_lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        regex_cache
            .word_boundary
            .find_iter(&text_to_process)
            .map(|mat| mat.as_str().to_string())
            .filter(|word| word.len() >= options.min_word_length)
            .collect()
    }
}

/// Hash utilities
pub struct HashUtils;

impl HashUtils {
    /// Generate SHA-256 hash of input
    pub fn sha256_hash(input: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Generate short hash (16 chars) for IDs
    pub fn short_hash(input: &str) -> String {
        Self::sha256_hash(input)[..16].to_string()
    }
}

/// Query feature detection
pub struct QueryFeatures;

impl QueryFeatures {
    /// Extract features from query text using cached regexes for better performance
    pub fn extract_features(query: &str) -> QueryFeatureFlags {
        if query.is_empty() {
            return QueryFeatureFlags::default();
        }

        let regex_cache = get_regex_cache();

        QueryFeatureFlags {
            has_code_symbol: regex_cache.code_symbol.is_match(query),
            has_error_token: regex_cache.error_token.is_match(query),
            has_path_or_file: regex_cache.path_file.is_match(query),
            has_numeric_id: regex_cache.numeric_id.is_match(query),
        }
    }

    /// Calculate gamma boost based on query features and content kind
    pub fn gamma_boost(kind: &str, features: &QueryFeatureFlags) -> f64 {
        let mut boost = 0.0;

        if features.has_code_symbol && (kind == "code" || kind == "user_code") {
            boost += 0.10;
        }

        if features.has_error_token && kind == "tool_result" {
            boost += 0.08;
        }

        if features.has_path_or_file && kind == "code" {
            boost += 0.04;
        }

        boost
    }
}

/// Overlap calculation utilities
pub struct OverlapUtils;

impl OverlapUtils {
    /// Calculate overlap ratio between two sets of document IDs
    pub fn calculate_overlap_ratio(set1: &[String], set2: &[String]) -> f64 {
        if set1.is_empty() || set2.is_empty() {
            return 0.0;
        }

        let ids1: HashSet<_> = set1.iter().collect();
        let ids2: HashSet<_> = set2.iter().collect();

        let intersection_size = ids1.intersection(&ids2).count();
        let union_size = ids1.union(&ids2).count();

        if union_size == 0 {
            0.0
        } else {
            intersection_size as f64 / union_size as f64
        }
    }
}

/// Text part from code fence extraction
#[derive(Debug, Clone)]
pub struct TextPart {
    pub kind: TextPartKind,
    pub content: String,
    pub start: usize,
    pub end: usize,
}

/// Kind of text part
#[derive(Debug, Clone, PartialEq)]
pub enum TextPartKind {
    Text,
    Code,
}

/// Query feature flags
#[derive(Debug, Clone, Default)]
pub struct QueryFeatureFlags {
    pub has_code_symbol: bool,
    pub has_error_token: bool,
    pub has_path_or_file: bool,
    pub has_numeric_id: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_counting() {
        assert_eq!(TokenCounter::count_tokens(""), 0);
        assert_eq!(TokenCounter::count_tokens("hello"), 1);
        assert_eq!(TokenCounter::count_tokens("hello world"), 3); // "hello" + "world" + whitespace = 3
        assert_eq!(TokenCounter::count_tokens("function_name()"), 3); // function_name + () = 3

        // Test the detailed counting for debugging
        let detailed = TokenCounter::count_tokens_detailed("hello world");
        assert_eq!(detailed.alphanumeric_tokens, 2); // "hello", "world"
        assert_eq!(detailed.whitespace_tokens, 1); // one space
        assert_eq!(detailed.total_tokens, 3); // 2 + 0 + 1 = 3
    }

    #[test]
    fn test_sentence_splitting() {
        let sentences = TextProcessor::split_sentences("Hello world. How are you? Fine thanks!");
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "How are you?");
        assert_eq!(sentences[2], "Fine thanks!");
    }

    #[test]
    fn test_code_fence_extraction() {
        let text = "Some text\n```rust\nfn main() {}\n```\nMore text";
        let parts = TextProcessor::extract_code_fences(text);
        assert_eq!(parts.len(), 3);
        assert!(matches!(parts[0].kind, TextPartKind::Text));
        assert!(matches!(parts[1].kind, TextPartKind::Code));
        assert!(matches!(parts[2].kind, TextPartKind::Text));
    }

    #[test]
    fn test_query_features() {
        let features = QueryFeatures::extract_features("function_name() error in /path/file.rs");
        assert!(features.has_code_symbol);
        assert!(features.has_error_token);
        assert!(features.has_path_or_file);
    }

    #[test]
    fn test_overlap_calculation() {
        let set1 = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let set2 = vec!["b".to_string(), "c".to_string(), "d".to_string()];
        let ratio = OverlapUtils::calculate_overlap_ratio(&set1, &set2);
        assert!((ratio - 0.5).abs() < f64::EPSILON); // 2 intersection / 4 union = 0.5
    }

    #[test]
    fn test_hash_generation() {
        let hash = HashUtils::short_hash("test input");
        assert_eq!(hash.len(), 16);

        // Same input should produce same hash
        let hash2 = HashUtils::short_hash("test input");
        assert_eq!(hash, hash2);

        // Different input should produce different hash
        let hash3 = HashUtils::short_hash("different input");
        assert_ne!(hash, hash3);
    }
}
