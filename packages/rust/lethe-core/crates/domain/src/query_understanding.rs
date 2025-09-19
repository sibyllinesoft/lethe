use lethe_shared::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Pre-compiled regex patterns for query analysis
struct QueryRegexes {
    code_function_call: Regex,
    code_method_access: Regex,
    code_punctuation: Regex,
    code_keywords: Regex,
    complexity_complex: Regex,
    complexity_simple: Regex,
    year_pattern: Regex,
    date_pattern: Regex,
    month_pattern: Regex,
}

impl QueryRegexes {
    fn new() -> Self {
        Self {
            code_function_call: Regex::new(r"\w+\(\)").unwrap(),
            code_method_access: Regex::new(r"\w+\.\w+").unwrap(),
            code_punctuation: Regex::new(r"[{}:;\[\]]").unwrap(),
            code_keywords: Regex::new(r"(?i)\b(def|class|import|function|const|let|var)\b").unwrap(),
            complexity_complex: Regex::new(r"(?i)\b(complex|advanced|sophisticated|intricate)\b").unwrap(),
            complexity_simple: Regex::new(r"(?i)\b(simple|basic|easy|straightforward)\b").unwrap(),
            year_pattern: Regex::new(r"\b\d{4}\b").unwrap(),
            date_pattern: Regex::new(r"\b\d{1,2}/\d{1,2}/\d{4}\b").unwrap(),
            month_pattern: Regex::new(r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\b").unwrap(),
        }
    }
}

static QUERY_REGEXES: OnceLock<QueryRegexes> = OnceLock::new();

fn get_query_regexes() -> &'static QueryRegexes {
    QUERY_REGEXES.get_or_init(QueryRegexes::new)
}

/// Static classification patterns to replace hardcoded logic
#[allow(dead_code)]
static QUERY_TYPE_PATTERNS: &[(QueryType, &[&str])] = &[
    (
        QueryType::Definitional,
        &["what is", "define", "definition of", "meaning of"],
    ),
    (
        QueryType::Procedural,
        &["how to", "steps to", "process of", "method to"],
    ),
    (
        QueryType::Comparative,
        &[
            "compare",
            "difference between",
            "vs",
            "versus",
            "better than",
        ],
    ),
    (
        QueryType::Enumerative,
        &["list of", "examples of", "types of", "kinds of"],
    ),
    (
        QueryType::Analytical,
        &["why", "analyze", "explain", "reason"],
    ),
    (
        QueryType::Subjective,
        &["opinion", "think", "feel", "recommend", "suggest"],
    ),
];

#[allow(dead_code)]
static QUERY_INTENT_PATTERNS: &[(QueryIntent, &[&str])] = &[
    (
        QueryIntent::Debug,
        &["error", "debug", "fix", "problem", "issue", "bug"],
    ),
    (
        QueryIntent::Code,
        &["code", "implement", "function", "class", "method"],
    ),
    (
        QueryIntent::Compare,
        &["compare", "difference", "vs", "versus"],
    ),
    (
        QueryIntent::Guide,
        &["steps", "guide", "tutorial", "instructions"],
    ),
    (
        QueryIntent::Explain,
        &["explain", "understand", "what", "clarify"],
    ),
    (QueryIntent::Assist, &["help", "assist", "how to", "need"]),
    (QueryIntent::Chat, &["hello", "hi", "thanks", "thank you"]),
];

static TECHNICAL_DOMAINS: &[(&str, &[&str])] = &[
    (
        "programming",
        &[
            "code",
            "function",
            "variable",
            "algorithm",
            "programming",
            "software",
            "debug",
            "api",
            "library",
            "javascript",
            "python",
            "java",
            "rust",
            "typescript",
        ],
    ),
    (
        "machine_learning",
        &[
            "machine learning",
            "neural network",
            "model",
            "training",
            "dataset",
            "prediction",
            "classification",
            "ai",
            "artificial intelligence",
        ],
    ),
    (
        "web_development",
        &[
            "html",
            "css",
            "javascript",
            "react",
            "vue",
            "angular",
            "frontend",
            "backend",
            "web",
            "http",
            "api",
            "rest",
        ],
    ),
    (
        "database",
        &[
            "database", "sql", "query", "table", "index", "schema", "postgres", "mysql", "mongodb",
            "nosql",
        ],
    ),
];

static QUESTION_WORDS: &[&str] = &[
    "what", "how", "why", "when", "where", "who", "which", "whose", "can", "could", "should",
    "would", "will", "do", "does", "did", "is", "are", "was", "were", "have", "has", "had",
];

/// Query classification types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QueryType {
    /// Simple factual question
    Factual,
    /// Complex analytical question requiring reasoning
    Analytical,
    /// Question asking for a comparison
    Comparative,
    /// Question asking for a list or enumeration
    Enumerative,
    /// Question asking for a definition
    Definitional,
    /// Question asking for procedural steps
    Procedural,
    /// Question asking for code or technical implementation
    Technical,
    /// Question asking for opinion or subjective analysis
    Subjective,
    /// General conversational query
    Conversational,
}

/// Intent classification for the query
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QueryIntent {
    /// User wants to find specific information
    Search,
    /// User wants an explanation or understanding
    Explain,
    /// User wants help with a task
    Assist,
    /// User wants to compare options
    Compare,
    /// User wants step-by-step instructions
    Guide,
    /// User wants code or technical solution
    Code,
    /// User wants to troubleshoot an issue
    Debug,
    /// User is having a conversation
    Chat,
}

/// Complexity level of the query
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QueryComplexity {
    Simple,
    Medium,
    Complex,
    VeryComplex,
}

/// Domain classification for the query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryDomain {
    pub primary_domain: String,
    pub secondary_domains: Vec<String>,
    pub confidence: f32,
}

/// Extracted entities from the query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryEntity {
    pub text: String,
    pub entity_type: String,
    pub start_pos: usize,
    pub end_pos: usize,
    pub confidence: f32,
}

/// Features extracted from the query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeatures {
    pub word_count: usize,
    pub sentence_count: usize,
    pub question_words: Vec<String>,
    pub technical_terms: Vec<String>,
    pub has_code: bool,
    pub has_numbers: bool,
    pub has_dates: bool,
    pub language: String,
}

/// Comprehensive query understanding result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryUnderstanding {
    pub original_query: String,
    pub query_type: QueryType,
    pub intent: QueryIntent,
    pub complexity: QueryComplexity,
    pub domain: QueryDomain,
    pub entities: Vec<QueryEntity>,
    pub features: QueryFeatures,
    pub keywords: Vec<String>,
    pub confidence: f32,
}

/// Helper struct for analyzing query complexity metrics
#[derive(Debug)]
#[allow(dead_code)]
struct QueryComplexityMetrics {
    word_count: usize,
    sentence_count: usize,
    has_technical_terms: bool,
    has_multiple_questions: bool,
}

#[allow(dead_code)]
impl QueryComplexityMetrics {
    fn analyze(query: &str) -> Self {
        let word_count = query.split_whitespace().count();
        let sentence_count = query.split('.').count();
        let has_technical_terms = QueryUnderstandingService::has_technical_terms(query);
        let has_multiple_questions = query.matches('?').count() > 1;

        Self {
            word_count,
            sentence_count,
            has_technical_terms,
            has_multiple_questions,
        }
    }
}

/// Query understanding service with optimized pattern matching
pub struct QueryUnderstandingService {
    // Using static data instead of instance data for better performance
}

impl QueryUnderstandingService {
    pub fn new() -> Self {
        Self {}
    }

    /// Analyze a query and return comprehensive understanding
    pub fn understand_query(&self, query: &str) -> Result<QueryUnderstanding> {
        let normalized_query = query.to_lowercase().trim().to_string();

        let query_type = self.classify_query_type(&normalized_query);
        let intent = self.classify_intent(&normalized_query);
        let complexity = self.classify_complexity(&normalized_query);
        let domain = self.classify_domain(&normalized_query);
        let entities = self.extract_entities(&normalized_query);
        let features = self.extract_features(&normalized_query);
        let keywords = self.extract_keywords(&normalized_query);
        let confidence = self.calculate_confidence(&normalized_query, &query_type, &intent);

        Ok(QueryUnderstanding {
            original_query: query.to_string(),
            query_type,
            intent,
            complexity,
            domain,
            entities,
            features,
            keywords,
            confidence,
        })
    }

    /// Classify the type of query
    fn classify_query_type(&self, query: &str) -> QueryType {
        // Check for definitional queries
        if query.contains("what is") || query.contains("define") || query.contains("definition") {
            return QueryType::Definitional;
        }

        // Check for procedural queries
        if query.contains("how to") || query.contains("steps") || query.contains("process") {
            return QueryType::Procedural;
        }

        // Check for comparative queries
        if query.contains("compare")
            || query.contains("difference")
            || query.contains("vs")
            || query.contains("versus")
            || query.contains("better")
        {
            return QueryType::Comparative;
        }

        // Check for enumerative queries
        if query.contains("list") || query.contains("examples") || query.contains("types of") {
            return QueryType::Enumerative;
        }

        // Check for technical queries
        if self.has_code_patterns(query) || Self::has_technical_terms(query) {
            return QueryType::Technical;
        }

        // Check for analytical queries
        if query.contains("why") || query.contains("analyze") || query.contains("explain") {
            return QueryType::Analytical;
        }

        // Check for subjective queries
        if query.contains("opinion")
            || query.contains("think")
            || query.contains("feel")
            || query.contains("recommend")
        {
            return QueryType::Subjective;
        }

        // Default to factual for simple questions
        QueryType::Factual
    }

    /// Classify the intent of the query
    fn classify_intent(&self, query: &str) -> QueryIntent {
        // Check more specific intents first before general ones
        if query.contains("error")
            || query.contains("debug")
            || query.contains("fix")
            || query.contains("problem")
        {
            return QueryIntent::Debug;
        }

        if self.has_code_patterns(query) || query.contains("code") || query.contains("implement") {
            return QueryIntent::Code;
        }

        if query.contains("compare") || query.contains("difference") || query.contains("vs") {
            return QueryIntent::Compare;
        }

        if query.contains("steps") || query.contains("guide") || query.contains("tutorial") {
            return QueryIntent::Guide;
        }

        if query.contains("explain") || query.contains("understand") || query.contains("what") {
            return QueryIntent::Explain;
        }

        if query.contains("help") || query.contains("assist") || query.contains("how to") {
            return QueryIntent::Assist;
        }

        if query.contains("hello") || query.contains("thanks") || query.len() < 20 {
            return QueryIntent::Chat;
        }

        QueryIntent::Search
    }

    /// Classify the complexity of the query
    fn classify_complexity(&self, query: &str) -> QueryComplexity {
        let regexes = get_query_regexes();

        // Check against predefined complexity patterns
        if regexes.complexity_complex.is_match(query) {
            return QueryComplexity::Complex;
        }
        if regexes.complexity_simple.is_match(query) {
            return QueryComplexity::Simple;
        }

        let word_count = query.split_whitespace().count();
        let sentence_count = query.split('.').count();
        let has_technical = Self::has_technical_terms(query);
        let has_multiple_questions = query.matches('?').count() > 1;

        match (
            word_count,
            sentence_count,
            has_technical,
            has_multiple_questions,
        ) {
            (w, s, true, true) if w > 30 && s > 3 => QueryComplexity::VeryComplex,
            (w, s, _, true) if w > 20 && s > 2 => QueryComplexity::Complex,
            (w, _, true, _) if w > 15 => QueryComplexity::Complex,
            (w, _, _, _) if w > 10 => QueryComplexity::Medium,
            _ => QueryComplexity::Simple,
        }
    }

    /// Classify the domain of the query
    fn classify_domain(&self, query: &str) -> QueryDomain {
        let mut domain_scores: HashMap<String, f32> = HashMap::new();

        // Check each technical domain
        for (domain, keywords) in TECHNICAL_DOMAINS {
            let mut score = 0.0;
            for keyword in *keywords {
                if query.contains(keyword) {
                    score += 1.0;
                }
            }
            if score > 0.0 {
                domain_scores.insert(domain.to_string(), score / keywords.len() as f32);
            }
        }

        // Find the best matching domain
        if let Some((primary_domain, confidence)) = domain_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let mut secondary_domains: Vec<String> = domain_scores
                .iter()
                .filter(|(domain, score)| *domain != primary_domain && **score > 0.3)
                .map(|(domain, _)| domain.clone())
                .collect();
            secondary_domains.sort();

            QueryDomain {
                primary_domain: primary_domain.clone(),
                secondary_domains,
                confidence: *confidence,
            }
        } else {
            QueryDomain {
                primary_domain: "general".to_string(),
                secondary_domains: Vec::new(),
                confidence: 0.5,
            }
        }
    }

    /// Extract named entities from the query
    fn extract_entities(&self, query: &str) -> Vec<QueryEntity> {
        let mut entities = Vec::new();

        // Simple entity extraction patterns
        let patterns = vec![
            (r"\b\d{4}\b", "year"),
            (r"\b\d+\.\d+\.\d+\b", "version"),
            (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", "proper_noun"),
            (r"\b\w+\(\)", "function"),
            (r"\b\w+\.\w+\b", "method_or_attribute"),
        ];

        for (pattern, entity_type) in patterns {
            if let Ok(regex) = Regex::new(pattern) {
                for mat in regex.find_iter(query) {
                    entities.push(QueryEntity {
                        text: mat.as_str().to_string(),
                        entity_type: entity_type.to_string(),
                        start_pos: mat.start(),
                        end_pos: mat.end(),
                        confidence: 0.8,
                    });
                }
            }
        }

        entities
    }

    /// Extract features from the query
    fn extract_features(&self, query: &str) -> QueryFeatures {
        let words: Vec<&str> = query.split_whitespace().collect();
        let sentences: Vec<&str> = query.split('.').collect();

        let question_words = words
            .iter()
            .filter(|word| QUESTION_WORDS.contains(&word.to_lowercase().as_str()))
            .map(|word| word.to_string())
            .collect();

        let technical_terms = self.extract_technical_terms(query);

        QueryFeatures {
            word_count: words.len(),
            sentence_count: sentences.len(),
            question_words,
            technical_terms,
            has_code: self.has_code_patterns(query),
            has_numbers: query.chars().any(|c| c.is_ascii_digit()),
            has_dates: self.has_date_patterns(query),
            language: "en".to_string(), // Simple language detection
        }
    }

    /// Extract keywords from the query
    fn extract_keywords(&self, query: &str) -> Vec<String> {
        let stop_words = vec![
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
            "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
            "the", "this", "but", "they", "have", "had", "what", "said", "each", "which", "she",
            "do", "how",
        ];

        query
            .split_whitespace()
            .filter(|word| {
                let word = word.to_lowercase();
                word.len() > 2 && !stop_words.contains(&word.as_str())
            })
            .map(|word| word.to_lowercase())
            .collect()
    }

    /// Calculate confidence in the query understanding
    fn calculate_confidence(
        &self,
        query: &str,
        query_type: &QueryType,
        _intent: &QueryIntent,
    ) -> f32 {
        let mut confidence: f32 = 0.5; // Base confidence

        // Boost confidence for clear patterns
        if self.has_clear_question_words(query) {
            confidence += 0.2;
        }

        if Self::has_technical_terms(query) && matches!(query_type, QueryType::Technical) {
            confidence += 0.2;
        }

        if query.ends_with('?') {
            confidence += 0.1;
        }

        // Reduce confidence for very short or very long queries
        let word_count = query.split_whitespace().count();
        if word_count < 3 || word_count > 50 {
            confidence -= 0.1;
        }

        confidence.min(1.0_f32).max(0.0_f32)
    }

    /// Check if query has code patterns
    fn has_code_patterns(&self, query: &str) -> bool {
        let regexes = get_query_regexes();
        regexes.code_function_call.is_match(query)
            || regexes.code_method_access.is_match(query)
            || regexes.code_punctuation.is_match(query)
            || regexes.code_keywords.is_match(query)
    }

    /// Check if query has technical terms
    fn has_technical_terms(query: &str) -> bool {
        TECHNICAL_DOMAINS
            .iter()
            .any(|(_, terms)| terms.iter().any(|term| query.contains(term)))
    }

    /// Check if query has clear question words
    fn has_clear_question_words(&self, query: &str) -> bool {
        QUESTION_WORDS.iter().any(|word| query.contains(word))
    }

    /// Check if query has date patterns
    fn has_date_patterns(&self, query: &str) -> bool {
        let regexes = get_query_regexes();
        regexes.year_pattern.is_match(query)
            || regexes.date_pattern.is_match(query)
            || regexes.month_pattern.is_match(query)
    }

    /// Extract technical terms from query
    fn extract_technical_terms(&self, query: &str) -> Vec<String> {
        let mut terms = Vec::new();

        for (_, domain_terms) in TECHNICAL_DOMAINS {
            for term in *domain_terms {
                if query.contains(term) {
                    terms.push(term.to_string());
                }
            }
        }

        terms
    }
}

impl Default for QueryUnderstandingService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_type_classification() {
        let service = QueryUnderstandingService::new();

        let understanding = service
            .understand_query("What is machine learning?")
            .unwrap();
        assert_eq!(understanding.query_type, QueryType::Definitional);

        let understanding = service
            .understand_query("How to implement a neural network?")
            .unwrap();
        assert_eq!(understanding.query_type, QueryType::Procedural);

        let understanding = service.understand_query("Compare React vs Vue").unwrap();
        assert_eq!(understanding.query_type, QueryType::Comparative);
    }

    #[test]
    fn test_intent_classification() {
        let service = QueryUnderstandingService::new();

        let understanding = service
            .understand_query("Explain how neural networks work")
            .unwrap();
        assert_eq!(understanding.intent, QueryIntent::Explain);

        let understanding = service.understand_query("Help me debug this code").unwrap();
        assert_eq!(understanding.intent, QueryIntent::Debug);

        let understanding = service
            .understand_query("Show me the steps to install Python")
            .unwrap();
        assert_eq!(understanding.intent, QueryIntent::Guide);
    }

    #[test]
    fn test_complexity_classification() {
        let service = QueryUnderstandingService::new();

        let understanding = service.understand_query("Hi").unwrap();
        assert_eq!(understanding.complexity, QueryComplexity::Simple);

        let understanding = service
            .understand_query(
                "How do I implement a complex distributed system with microservices architecture?",
            )
            .unwrap();
        assert_eq!(understanding.complexity, QueryComplexity::Complex);
    }

    #[test]
    fn test_domain_classification() {
        let service = QueryUnderstandingService::new();

        let understanding = service
            .understand_query("How to train a machine learning model?")
            .unwrap();
        assert_eq!(understanding.domain.primary_domain, "machine_learning");

        let understanding = service
            .understand_query("Write a JavaScript function")
            .unwrap();
        assert_eq!(understanding.domain.primary_domain, "programming");
    }

    #[test]
    fn test_feature_extraction() {
        let service = QueryUnderstandingService::new();

        let understanding = service
            .understand_query("What is the function setTimeout() in JavaScript?")
            .unwrap();
        assert!(understanding.features.word_count > 0);
        assert!(understanding.features.has_code);
        assert!(!understanding.features.question_words.is_empty());
    }

    #[test]
    fn test_keyword_extraction() {
        let service = QueryUnderstandingService::new();

        let understanding = service
            .understand_query("How to implement machine learning algorithms")
            .unwrap();
        assert!(understanding.keywords.contains(&"implement".to_string()));
        assert!(understanding.keywords.contains(&"machine".to_string()));
        assert!(understanding.keywords.contains(&"learning".to_string()));
        assert!(understanding.keywords.contains(&"algorithms".to_string()));
    }

    #[test]
    fn test_confidence_calculation() {
        let service = QueryUnderstandingService::new();

        let understanding = service
            .understand_query("What is machine learning?")
            .unwrap();
        assert!(understanding.confidence > 0.5);

        let understanding = service.understand_query("a").unwrap();
        assert!(understanding.confidence < 0.5);
    }
}
