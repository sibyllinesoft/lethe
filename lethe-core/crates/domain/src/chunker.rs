use lethe_shared::{Chunk, Message, Result};
use lethe_shared::utils::{HashUtils, TextProcessor, TokenCounter, TextPart, TextPartKind};
use uuid::Uuid;

/// Configuration for the chunking service
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

/// Service for chunking messages into smaller text segments
pub struct ChunkingService {
    config: ChunkingConfig,
}

impl ChunkingService {
    /// Create a new chunking service with configuration
    pub fn new(config: ChunkingConfig) -> Self {
        Self { config }
    }

    /// Chunk a message into text segments
    pub fn chunk_message(&self, message: &Message) -> Result<Vec<Chunk>> {
        // Normalize text to NFC
        let normalized_text = TextProcessor::normalize_text(&message.text);
        
        // Extract code fences and text parts
        let parts = TextProcessor::extract_code_fences(&normalized_text);
        
        // Create chunks from parts
        let mut chunks = Vec::new();
        for part in parts {
            let part_chunks = self.create_chunks_from_part(
                &message.id,
                &message.session_id,
                &part,
            )?;
            chunks.extend(part_chunks);
        }

        Ok(chunks)
    }

    /// Create chunks from a text part
    fn create_chunks_from_part(
        &self,
        message_id: &Uuid,
        session_id: &str,
        part: &TextPart,
    ) -> Result<Vec<Chunk>> {
        let tokens = TokenCounter::count_tokens(&part.content);
        let mut chunks = Vec::new();

        if tokens <= self.config.target_tokens {
            // Part fits in one chunk
            let chunk_id = HashUtils::short_hash(&format!("{}-{}-{}", message_id, part.start, part.end));
            
            chunks.push(Chunk {
                id: chunk_id,
                message_id: *message_id,
                session_id: session_id.to_string(),
                offset_start: part.start,
                offset_end: part.end,
                kind: match part.kind {
                    TextPartKind::Text => "text".to_string(),
                    TextPartKind::Code => "code".to_string(),
                },
                text: part.content.clone(),
                tokens,
            });
        } else {
            // Need to split the part
            match part.kind {
                TextPartKind::Text => {
                    chunks.extend(self.split_text_part(message_id, session_id, part)?);
                }
                TextPartKind::Code => {
                    chunks.extend(self.split_code_part(message_id, session_id, part)?);
                }
            }
        }

        Ok(chunks)
    }

    /// Split a text part into multiple chunks
    fn split_text_part(
        &self,
        message_id: &Uuid,
        session_id: &str,
        part: &TextPart,
    ) -> Result<Vec<Chunk>> {
        let sentences = TextProcessor::split_sentences(&part.content);
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_start = part.start;
        let mut current_tokens = 0;

        for sentence in sentences {
            let sentence_tokens = TokenCounter::count_tokens(&sentence);
            
            if current_tokens + sentence_tokens > self.config.target_tokens && !current_chunk.is_empty() {
                // Create chunk
                let chunk_end = current_start + current_chunk.len();
                let chunk_id = HashUtils::short_hash(&format!("{}-{}-{}", message_id, current_start, chunk_end));

                chunks.push(Chunk {
                    id: chunk_id,
                    message_id: *message_id,
                    session_id: session_id.to_string(),
                    offset_start: current_start,
                    offset_end: chunk_end,
                    kind: "text".to_string(),
                    text: current_chunk.trim().to_string(),
                    tokens: current_tokens,
                });

                // Start new chunk with overlap
                let overlap_text = if current_chunk.len() > self.config.overlap as usize {
                    current_chunk[current_chunk.len() - self.config.overlap as usize..].to_string()
                } else {
                    current_chunk.clone()
                };
                
                current_chunk = format!("{} {}", overlap_text, sentence);
                current_start = chunk_end - overlap_text.len();
                current_tokens = TokenCounter::count_tokens(&current_chunk);
            } else {
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(&sentence);
                current_tokens += sentence_tokens;
            }
        }

        // Add final chunk
        if !current_chunk.trim().is_empty() {
            let chunk_end = current_start + current_chunk.len();
            let chunk_id = HashUtils::short_hash(&format!("{}-{}-{}", message_id, current_start, chunk_end));

            chunks.push(Chunk {
                id: chunk_id,
                message_id: *message_id,
                session_id: session_id.to_string(),
                offset_start: current_start,
                offset_end: chunk_end,
                kind: "text".to_string(),
                text: current_chunk.trim().to_string(),
                tokens: current_tokens,
            });
        }

        Ok(chunks)
    }

    /// Split a code part into multiple chunks
    fn split_code_part(
        &self,
        message_id: &Uuid,
        session_id: &str,
        part: &TextPart,
    ) -> Result<Vec<Chunk>> {
        let lines: Vec<&str> = part.content.split('\n').collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_start = part.start;
        let mut current_tokens = 0;
        let mut line_offset = 0;

        for (i, line) in lines.iter().enumerate() {
            let line_with_newline = if i < lines.len() - 1 {
                format!("{}\n", line)
            } else {
                line.to_string()
            };
            let line_tokens = TokenCounter::count_tokens(&line_with_newline);
            
            if current_tokens + line_tokens > self.config.target_tokens && !current_chunk.is_empty() {
                // Create chunk
                let chunk_end = current_start + current_chunk.len();
                let chunk_id = HashUtils::short_hash(&format!("{}-{}-{}", message_id, current_start, chunk_end));

                chunks.push(Chunk {
                    id: chunk_id,
                    message_id: *message_id,
                    session_id: session_id.to_string(),
                    offset_start: current_start,
                    offset_end: chunk_end,
                    kind: "code".to_string(),
                    text: current_chunk.clone(),
                    tokens: current_tokens,
                });

                // Start new chunk with overlap (few lines)
                let overlap_lines = std::cmp::min(3, self.config.overlap / 20);
                let start_idx = std::cmp::max(0, i as i32 - overlap_lines) as usize;
                let overlap_text = lines[start_idx..i].join("\n");
                
                let line_len = line_with_newline.len(); // Store length before move
                
                current_chunk = if overlap_text.is_empty() {
                    line_with_newline
                } else {
                    format!("{}\n{}", overlap_text, line_with_newline)
                };
                
                current_start = part.start + line_offset - overlap_text.len();
                current_tokens = TokenCounter::count_tokens(&current_chunk);
                line_offset += line_len;
            } else {
                line_offset += line_with_newline.len();
                current_chunk.push_str(&line_with_newline);
                current_tokens += line_tokens;
            }
        }

        // Add final chunk
        if !current_chunk.trim().is_empty() {
            let chunk_end = current_start + current_chunk.len();
            let chunk_id = HashUtils::short_hash(&format!("{}-{}-{}", message_id, current_start, chunk_end));

            chunks.push(Chunk {
                id: chunk_id,
                message_id: *message_id,
                session_id: session_id.to_string(),
                offset_start: current_start,
                offset_end: chunk_end,
                kind: "code".to_string(),
                text: current_chunk,
                tokens: current_tokens,
            });
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_message(text: &str) -> Message {
        Message {
            id: Uuid::new_v4(),
            session_id: "test-session".to_string(),
            turn: 1,
            role: "user".to_string(),
            text: text.to_string(),
            ts: Utc::now(),
            meta: None,
        }
    }

    #[test]
    fn test_simple_chunking() {
        let config = ChunkingConfig::default();
        let service = ChunkingService::new(config);
        
        let message = create_test_message("This is a simple test message.");
        let chunks = service.chunk_message(&message).unwrap();
        
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, "text");
        assert_eq!(chunks[0].text, "This is a simple test message.");
    }

    #[test]
    fn test_code_fence_detection() {
        let config = ChunkingConfig::default();
        let service = ChunkingService::new(config);
        
        let message = create_test_message("Here's some code:\n```rust\nfn main() {\n    println!(\"Hello\");\n}\n```\nThat was the code.");
        let chunks = service.chunk_message(&message).unwrap();
        
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].kind, "text");
        assert_eq!(chunks[1].kind, "code");
        assert_eq!(chunks[2].kind, "text");
    }

    #[test]
    fn test_long_text_splitting() {
        let config = ChunkingConfig {
            target_tokens: 10, // Very small for testing
            overlap: 2,
        };
        let service = ChunkingService::new(config);
        
        let long_text = "This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth sentence.";
        let message = create_test_message(long_text);
        let chunks = service.chunk_message(&message).unwrap();
        
        // Should split into multiple chunks due to small target_tokens
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| c.kind == "text"));
    }

    #[test]
    fn test_token_counting_accuracy() {
        let short_text = "hello";
        let medium_text = "hello world";
        let long_text = "This is a longer text with multiple words and punctuation!";
        
        assert_eq!(TokenCounter::count_tokens(short_text), 1);
        assert!(TokenCounter::count_tokens(medium_text) >= 2);
        assert!(TokenCounter::count_tokens(long_text) > TokenCounter::count_tokens(medium_text));
    }

    #[test]
    fn test_chunking_configuration() {
        let small_config = ChunkingConfig {
            target_tokens: 5,
            overlap: 1,
        };
        
        let large_config = ChunkingConfig {
            target_tokens: 100,
            overlap: 10,
        };
        
        let small_service = ChunkingService::new(small_config);
        let large_service = ChunkingService::new(large_config);
        
        let text = "This is a test message with several words that should be chunked differently based on configuration.";
        let message = create_test_message(text);
        
        let small_chunks = small_service.chunk_message(&message).unwrap();
        let large_chunks = large_service.chunk_message(&message).unwrap();
        
        // Small config should create more chunks
        assert!(small_chunks.len() >= large_chunks.len());
        
        // All chunks should have proper metadata
        for chunk in &small_chunks {
            assert!(!chunk.id.is_empty());
            assert_eq!(chunk.message_id, message.id);
            assert_eq!(chunk.session_id, message.session_id);
            assert!(chunk.tokens > 0);
        }
    }

    #[test]
    fn test_chunking_overlap_behavior() {
        let config = ChunkingConfig {
            target_tokens: 10,
            overlap: 3,
        };
        let service = ChunkingService::new(config);
        
        let text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here.";
        let message = create_test_message(text);
        let chunks = service.chunk_message(&message).unwrap();
        
        if chunks.len() > 1 {
            // Check that subsequent chunks have some overlapping content
            // This is hard to test precisely due to sentence splitting, but we can verify structure
            for i in 1..chunks.len() {
                assert!(chunks[i].offset_start < chunks[i].offset_end);
                assert!(chunks[i-1].offset_end > chunks[i].offset_start); // Some overlap expected
            }
        }
    }

    #[test]
    fn test_chunking_edge_cases() {
        let service = ChunkingService::new(ChunkingConfig::default());
        
        // Test empty message
        let empty_message = create_test_message("");
        let empty_chunks = service.chunk_message(&empty_message).unwrap();
        assert_eq!(empty_chunks.len(), 1); // Even empty creates one chunk
        
        // Test whitespace only
        let whitespace_message = create_test_message("   \n\t  ");
        let whitespace_chunks = service.chunk_message(&whitespace_message).unwrap();
        assert_eq!(whitespace_chunks.len(), 1); // Whitespace creates a chunk too
        
        // Test single word
        let single_word_message = create_test_message("hello");
        let single_word_chunks = service.chunk_message(&single_word_message).unwrap();
        assert_eq!(single_word_chunks.len(), 1);
        assert_eq!(single_word_chunks[0].text, "hello");
        
        // Test very long word
        let long_word = "a".repeat(1000);
        let long_word_message = create_test_message(&long_word);
        let long_word_chunks = service.chunk_message(&long_word_message).unwrap();
        assert!(!long_word_chunks.is_empty());
        assert!(long_word_chunks[0].text.len() <= 1000);
    }

    #[test]
    fn test_mixed_content_chunking() {
        let service = ChunkingService::new(ChunkingConfig::default());
        
        let mixed_content = r#"
This is regular text content.

```python
def hello_world():
    print("Hello, World!")
    return "success"
```

And this is more text after the code block.

```javascript  
function greet(name) {
    return `Hello, ${name}!`;
}
```

Final text content here.
        "#;
        
        let message = create_test_message(mixed_content);
        let chunks = service.chunk_message(&message).unwrap();
        
        assert!(!chunks.is_empty());
        
        // Should have different kinds of chunks
        let kinds: Vec<String> = chunks.iter().map(|c| c.kind.clone()).collect();
        let unique_kinds: std::collections::HashSet<String> = kinds.into_iter().collect();
        
        // Should have at least text chunks, possibly code chunks too
        assert!(unique_kinds.contains("text"));
        
        // All chunks should have valid offsets
        for chunk in &chunks {
            assert!(chunk.offset_start < chunk.offset_end);
            assert!(chunk.offset_end <= mixed_content.len());
        }
    }

    #[test]
    fn test_token_counter_edge_cases() {
        // Test empty string
        assert_eq!(TokenCounter::count_tokens(""), 0);
        
        // Test whitespace only  
        assert_eq!(TokenCounter::count_tokens("   "), 0);
        assert_eq!(TokenCounter::count_tokens("\n\t"), 0);
        
        // Test punctuation only
        assert!(TokenCounter::count_tokens("!!!") > 0);
        assert!(TokenCounter::count_tokens("...") > 0);
        
        // Test numbers
        assert_eq!(TokenCounter::count_tokens("123"), 1);
        assert_eq!(TokenCounter::count_tokens("123 456"), 3); // 2 alphanumeric + 1 whitespace
        
        // Test mixed alphanumeric
        assert_eq!(TokenCounter::count_tokens("abc123"), 1);
        assert_eq!(TokenCounter::count_tokens("test123 demo456"), 3); // 2 alphanumeric + 1 whitespace
        
        // Test special characters
        assert!(TokenCounter::count_tokens("@#$%") > 0);
        assert!(TokenCounter::count_tokens("email@domain.com") > 0);
        
        // Test unicode
        assert_eq!(TokenCounter::count_tokens("hello"), TokenCounter::count_tokens("hello"));
        assert!(TokenCounter::count_tokens("测试") > 0);
        assert!(TokenCounter::count_tokens("🌍🚀") > 0);
    }

    #[test]
    fn test_chunk_validation() {
        let service = ChunkingService::new(ChunkingConfig::default());
        let message = create_test_message("Test message with multiple sentences. Each should be properly chunked.");
        let chunks = service.chunk_message(&message).unwrap();
        
        for chunk in &chunks {
            // Validate chunk structure
            assert!(!chunk.id.is_empty());
            assert_eq!(chunk.message_id, message.id);
            assert_eq!(chunk.session_id, message.session_id);
            assert!(!chunk.text.is_empty());
            assert!(chunk.tokens > 0);
            assert!(chunk.offset_start < chunk.offset_end);
            
            // Validate that chunk text matches the message text at the specified offsets
            let expected_text = message.text[chunk.offset_start..chunk.offset_end].trim();
            assert!(!expected_text.is_empty());
        }
    }

    #[test]  
    fn test_chunking_service_consistency() {
        let service = ChunkingService::new(ChunkingConfig::default());
        let text = "Consistent test message for chunking.";
        let message = create_test_message(text);
        
        // Chunk the same message multiple times
        let chunks1 = service.chunk_message(&message).unwrap();
        let chunks2 = service.chunk_message(&message).unwrap();
        
        // Results should be identical
        assert_eq!(chunks1.len(), chunks2.len());
        
        for (c1, c2) in chunks1.iter().zip(chunks2.iter()) {
            assert_eq!(c1.text, c2.text);
            assert_eq!(c1.kind, c2.kind);
            assert_eq!(c1.offset_start, c2.offset_start);
            assert_eq!(c1.offset_end, c2.offset_end);
            assert_eq!(c1.tokens, c2.tokens);
        }
    }

    #[test]
    fn test_chunking_config_clone_and_debug() {
        let config = ChunkingConfig {
            target_tokens: 50,
            overlap: 5,
        };
        
        // Test Clone trait
        let cloned_config = config.clone();
        assert_eq!(config.target_tokens, cloned_config.target_tokens);
        assert_eq!(config.overlap, cloned_config.overlap);
        
        // Test Debug trait
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("ChunkingConfig"));
        assert!(debug_str.contains("target_tokens"));
        assert!(debug_str.contains("overlap"));
    }
}