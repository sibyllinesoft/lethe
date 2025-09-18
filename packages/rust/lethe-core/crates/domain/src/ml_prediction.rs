use crate::query_understanding::{QueryComplexity, QueryIntent, QueryType, QueryUnderstanding};
use lethe_shared::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Static feature weight configurations to avoid HashMap initialization
static FEATURE_WEIGHTS: &[(&str, f32)] = &[
    ("query_length", 0.15),
    ("complexity", 0.25),
    ("technical_terms", 0.20),
    ("domain_specificity", 0.15),
    ("semantic_complexity", 0.25),
];

/// Static strategy weight configurations
static STRATEGY_WEIGHTS: &[(RetrievalStrategy, f32)] = &[
    (RetrievalStrategy::BM25Only, 1.0),
    (RetrievalStrategy::VectorOnly, 1.0),
    (RetrievalStrategy::Hybrid, 1.2),
    (RetrievalStrategy::HydeEnhanced, 0.8),
    (RetrievalStrategy::MultiStep, 0.9),
    (RetrievalStrategy::Adaptive, 1.1),
];

/// Static feature scoring rules to replace complex if-statements
struct FeatureScoringRule {
    condition: fn(&MLFeatures) -> bool,
    strategy: RetrievalStrategy,
    score: f32,
}

static FEATURE_SCORING_RULES: &[FeatureScoringRule] = &[
    FeatureScoringRule {
        condition: |f| f.semantic_complexity > 0.7,
        strategy: RetrievalStrategy::VectorOnly,
        score: 0.3,
    },
    FeatureScoringRule {
        condition: |f| f.semantic_complexity > 0.7,
        strategy: RetrievalStrategy::HydeEnhanced,
        score: 0.2,
    },
    FeatureScoringRule {
        condition: |f| f.technical_term_count > 0.5 || f.has_code > 0.5,
        strategy: RetrievalStrategy::BM25Only,
        score: 0.3,
    },
    FeatureScoringRule {
        condition: |f| f.query_complexity_score > 0.6,
        strategy: RetrievalStrategy::Hybrid,
        score: 0.4,
    },
    FeatureScoringRule {
        condition: |f| f.query_complexity_score > 0.6,
        strategy: RetrievalStrategy::MultiStep,
        score: 0.2,
    },
    FeatureScoringRule {
        condition: |f| f.domain_specificity < 0.5,
        strategy: RetrievalStrategy::Adaptive,
        score: 0.2,
    },
];

/// Static feature names to avoid vector allocation
static FEATURE_NAMES: &[&str] = &[
    "query_length",
    "query_complexity_score",
    "technical_term_count",
    "question_word_presence",
    "domain_specificity",
    "has_code",
    "has_numbers",
    "intent_score",
    "semantic_complexity",
];

/// Static strategy name mappings
static STRATEGY_NAMES: &[(RetrievalStrategy, &str)] = &[
    (RetrievalStrategy::BM25Only, "BM25-only"),
    (RetrievalStrategy::VectorOnly, "Vector-only"),
    (RetrievalStrategy::Hybrid, "Hybrid"),
    (RetrievalStrategy::HydeEnhanced, "HyDE-enhanced"),
    (RetrievalStrategy::MultiStep, "Multi-step"),
    (RetrievalStrategy::Adaptive, "Adaptive"),
];

/// Static complexity scoring patterns
static COMPLEXITY_SCORES: &[(QueryComplexity, f32)] = &[
    (QueryComplexity::Simple, 0.2),
    (QueryComplexity::Medium, 0.5),
    (QueryComplexity::Complex, 0.8),
    (QueryComplexity::VeryComplex, 1.0),
];

/// Static intent scoring patterns
static INTENT_SCORES: &[(QueryIntent, f32)] = &[
    (QueryIntent::Search, 0.8),
    (QueryIntent::Explain, 0.6),
    (QueryIntent::Code, 1.0),
    (QueryIntent::Debug, 0.9),
    (QueryIntent::Compare, 0.7),
    (QueryIntent::Guide, 0.5),
    (QueryIntent::Assist, 0.4),
    (QueryIntent::Chat, 0.2),
];

/// ML model prediction for retrieval strategy selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalStrategyPrediction {
    pub strategy: RetrievalStrategy,
    pub confidence: f32,
    pub features_used: Vec<String>,
    pub alternatives: Vec<(RetrievalStrategy, f32)>,
}

/// Available retrieval strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RetrievalStrategy {
    /// Pure BM25 lexical search
    BM25Only,
    /// Pure vector similarity search
    VectorOnly,
    /// Hybrid BM25 + vector search
    Hybrid,
    /// HyDE-enhanced vector search
    HydeEnhanced,
    /// Multi-step retrieval with reranking
    MultiStep,
    /// Adaptive strategy based on query
    Adaptive,
}

/// Feature vector for ML prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLFeatures {
    pub query_length: f32,
    pub query_complexity_score: f32,
    pub technical_term_count: f32,
    pub question_word_presence: f32,
    pub domain_specificity: f32,
    pub has_code: f32,
    pub has_numbers: f32,
    pub intent_score: f32,
    pub semantic_complexity: f32,
}

/// ML prediction result with explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPredictionResult {
    pub prediction: RetrievalStrategyPrediction,
    pub explanation: String,
    pub feature_importance: HashMap<String, f32>,
    pub model_confidence: f32,
}

/// Configuration for ML prediction service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPredictionConfig {
    pub enable_hybrid_fallback: bool,
    pub confidence_threshold: f32,
    pub feature_weights: HashMap<String, f32>,
    pub strategy_weights: HashMap<RetrievalStrategy, f32>,
}

impl Default for MLPredictionConfig {
    fn default() -> Self {
        let feature_weights = FEATURE_WEIGHTS
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();

        let strategy_weights = STRATEGY_WEIGHTS
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        Self {
            enable_hybrid_fallback: true,
            confidence_threshold: 0.7,
            feature_weights,
            strategy_weights,
        }
    }
}

/// ML prediction service for retrieval strategy selection
pub struct MLPredictionService {
    _config: MLPredictionConfig,
    strategy_rules: Vec<Box<dyn StrategyRule>>,
}

impl MLPredictionService {
    pub fn new(config: MLPredictionConfig) -> Self {
        let mut service = Self {
            _config: config,
            strategy_rules: Vec::new(),
        };

        service.initialize_rules();
        service
    }

    /// Predict the best retrieval strategy for a given query understanding
    pub fn predict_strategy(
        &self,
        understanding: &QueryUnderstanding,
    ) -> Result<MLPredictionResult> {
        let features = self.extract_features(understanding);
        let (strategy_scores, explanations) =
            self.collect_strategy_scores(understanding, &features);
        let prediction = self.create_prediction_from_scores(strategy_scores, &features);
        let explanation = self.generate_explanation(&prediction, understanding, &explanations);
        let feature_importance = self.calculate_feature_importance(&features);
        let confidence = prediction.confidence;

        Ok(MLPredictionResult {
            prediction,
            explanation,
            feature_importance,
            model_confidence: confidence,
        })
    }

    /// Collect strategy scores from rules and features
    fn collect_strategy_scores(
        &self,
        understanding: &QueryUnderstanding,
        features: &MLFeatures,
    ) -> (HashMap<RetrievalStrategy, f32>, Vec<String>) {
        let mut strategy_scores: HashMap<RetrievalStrategy, f32> = HashMap::new();
        let mut explanations = Vec::new();

        // Apply rule-based predictions
        for rule in &self.strategy_rules {
            if let Some(prediction) = rule.evaluate(understanding, features) {
                *strategy_scores
                    .entry(prediction.strategy.clone())
                    .or_insert(0.0) += prediction.confidence;
                explanations.push(prediction.explanation);
            }
        }

        // Apply feature-based scoring
        self.apply_feature_scoring(features, &mut strategy_scores);

        (strategy_scores, explanations)
    }

    /// Create prediction from strategy scores
    fn create_prediction_from_scores(
        &self,
        strategy_scores: HashMap<RetrievalStrategy, f32>,
        features: &MLFeatures,
    ) -> RetrievalStrategyPrediction {
        let (best_strategy, best_score) = self.select_best_strategy(&strategy_scores);
        let total_score: f32 = strategy_scores.values().sum();
        let alternatives = self.create_alternatives(strategy_scores, &best_strategy, total_score);

        RetrievalStrategyPrediction {
            strategy: best_strategy,
            confidence: (best_score / total_score).min(1.0),
            features_used: features.get_feature_names(),
            alternatives,
        }
    }

    /// Select the best strategy from scores
    fn select_best_strategy(
        &self,
        strategy_scores: &HashMap<RetrievalStrategy, f32>,
    ) -> (RetrievalStrategy, f32) {
        strategy_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(s, score)| (s.clone(), *score))
            .unwrap_or((RetrievalStrategy::Hybrid, 0.5))
    }

    /// Create alternative strategies list
    fn create_alternatives(
        &self,
        strategy_scores: HashMap<RetrievalStrategy, f32>,
        best_strategy: &RetrievalStrategy,
        total_score: f32,
    ) -> Vec<(RetrievalStrategy, f32)> {
        let mut alternatives: Vec<(RetrievalStrategy, f32)> = strategy_scores
            .into_iter()
            .filter(|(s, _)| s != best_strategy)
            .map(|(s, score)| (s, score / total_score))
            .collect();

        alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        alternatives
    }

    /// Extract ML features from query understanding
    fn extract_features(&self, understanding: &QueryUnderstanding) -> MLFeatures {
        let query_length = (understanding.original_query.len() as f32 / 100.0).min(2.0);

        let query_complexity_score = COMPLEXITY_SCORES
            .iter()
            .find(|(complexity, _)| *complexity == understanding.complexity)
            .map(|(_, score)| *score)
            .unwrap_or(0.5);

        let technical_term_count =
            (understanding.features.technical_terms.len() as f32 / 10.0).min(1.0);

        let question_word_presence = if understanding.features.question_words.is_empty() {
            0.0
        } else {
            (understanding.features.question_words.len() as f32 / 5.0).min(1.0)
        };

        let domain_specificity = understanding.domain.confidence;

        let has_code = if understanding.features.has_code {
            1.0
        } else {
            0.0
        };
        let has_numbers = if understanding.features.has_numbers {
            1.0
        } else {
            0.0
        };

        let intent_score = INTENT_SCORES
            .iter()
            .find(|(intent, _)| *intent == understanding.intent)
            .map(|(_, score)| *score)
            .unwrap_or(0.5);

        let semantic_complexity = self.calculate_semantic_complexity(understanding);

        MLFeatures {
            query_length,
            query_complexity_score,
            technical_term_count,
            question_word_presence,
            domain_specificity,
            has_code,
            has_numbers,
            intent_score,
            semantic_complexity,
        }
    }

    /// Apply feature-based scoring to strategy predictions using static rules
    fn apply_feature_scoring(
        &self,
        features: &MLFeatures,
        strategy_scores: &mut HashMap<RetrievalStrategy, f32>,
    ) {
        for rule in FEATURE_SCORING_RULES {
            if (rule.condition)(features) {
                *strategy_scores.entry(rule.strategy.clone()).or_insert(0.0) += rule.score;
            }
        }
    }

    /// Calculate semantic complexity of the query
    fn calculate_semantic_complexity(&self, understanding: &QueryUnderstanding) -> f32 {
        let mut complexity = 0.0;

        // Abstract concepts increase semantic complexity
        if understanding.query_type == QueryType::Analytical
            || understanding.query_type == QueryType::Subjective
        {
            complexity += 0.3;
        }

        // Multiple entities increase complexity
        complexity += (understanding.entities.len() as f32 / 10.0).min(0.3);

        // Long queries with few technical terms are more semantic
        if understanding.features.word_count > 10
            && understanding.features.technical_terms.len() < 3
        {
            complexity += 0.4;
        }

        complexity.min(1.0)
    }

    /// Generate human-readable explanation for the prediction
    fn generate_explanation(
        &self,
        prediction: &RetrievalStrategyPrediction,
        understanding: &QueryUnderstanding,
        _rule_explanations: &[String],
    ) -> String {
        let mut explanation = format!(
            "Selected {} strategy with {:.1}% confidence. ",
            strategy_to_string(&prediction.strategy),
            prediction.confidence * 100.0
        );

        // Add reasoning based on query characteristics
        match prediction.strategy {
            RetrievalStrategy::BM25Only => {
                explanation.push_str("This strategy was chosen because the query contains specific technical terms or keywords that benefit from exact matching.");
            }
            RetrievalStrategy::VectorOnly => {
                explanation.push_str("This strategy was chosen because the query is conceptual and would benefit from semantic similarity matching.");
            }
            RetrievalStrategy::Hybrid => {
                explanation.push_str("This strategy combines both keyword matching and semantic similarity for comprehensive results.");
            }
            RetrievalStrategy::HydeEnhanced => {
                explanation.push_str("This strategy uses hypothetical document generation to improve semantic matching for complex queries.");
            }
            RetrievalStrategy::MultiStep => {
                explanation.push_str("This strategy uses multiple retrieval phases with reranking for high-precision results.");
            }
            RetrievalStrategy::Adaptive => {
                explanation.push_str(
                    "This strategy dynamically adjusts based on initial results quality.",
                );
            }
        }

        // Add specific insights
        if understanding.features.has_code {
            explanation.push_str(" Code-related queries detected.");
        }
        if understanding.complexity == QueryComplexity::VeryComplex {
            explanation.push_str(" High query complexity requires sophisticated retrieval.");
        }

        explanation
    }

    /// Calculate feature importance scores
    fn calculate_feature_importance(&self, features: &MLFeatures) -> HashMap<String, f32> {
        let mut importance = HashMap::new();

        importance.insert("query_length".to_string(), features.query_length * 0.15);
        importance.insert(
            "complexity".to_string(),
            features.query_complexity_score * 0.25,
        );
        importance.insert(
            "technical_terms".to_string(),
            features.technical_term_count * 0.20,
        );
        importance.insert(
            "domain_specificity".to_string(),
            features.domain_specificity * 0.15,
        );
        importance.insert(
            "semantic_complexity".to_string(),
            features.semantic_complexity * 0.25,
        );

        importance
    }

    /// Initialize strategy selection rules
    fn initialize_rules(&mut self) {
        self.strategy_rules.push(Box::new(TechnicalQueryRule));
        self.strategy_rules.push(Box::new(SemanticQueryRule));
        self.strategy_rules.push(Box::new(ComplexQueryRule));
        self.strategy_rules.push(Box::new(CodeQueryRule));
        self.strategy_rules.push(Box::new(ComparisonQueryRule));
    }
}

impl Default for MLPredictionService {
    fn default() -> Self {
        Self::new(MLPredictionConfig::default())
    }
}

/// Rule-based prediction for strategy selection
trait StrategyRule: Send + Sync {
    fn evaluate(
        &self,
        understanding: &QueryUnderstanding,
        features: &MLFeatures,
    ) -> Option<RulePrediction>;
}

/// Individual rule prediction
struct RulePrediction {
    strategy: RetrievalStrategy,
    confidence: f32,
    explanation: String,
}

/// Rule for technical queries
struct TechnicalQueryRule;

impl StrategyRule for TechnicalQueryRule {
    fn evaluate(
        &self,
        _understanding: &QueryUnderstanding,
        features: &MLFeatures,
    ) -> Option<RulePrediction> {
        if features.technical_term_count > 0.6 || features.has_code > 0.5 {
            Some(RulePrediction {
                strategy: RetrievalStrategy::BM25Only,
                confidence: 0.8,
                explanation: "Technical terms favor keyword-based search".to_string(),
            })
        } else {
            None
        }
    }
}

/// Rule for semantic queries
struct SemanticQueryRule;

impl StrategyRule for SemanticQueryRule {
    fn evaluate(
        &self,
        _understanding: &QueryUnderstanding,
        features: &MLFeatures,
    ) -> Option<RulePrediction> {
        if features.semantic_complexity > 0.7 && features.technical_term_count < 0.3 {
            Some(RulePrediction {
                strategy: RetrievalStrategy::VectorOnly,
                confidence: 0.7,
                explanation: "High semantic complexity favors vector search".to_string(),
            })
        } else {
            None
        }
    }
}

/// Rule for complex queries
struct ComplexQueryRule;

impl StrategyRule for ComplexQueryRule {
    fn evaluate(
        &self,
        understanding: &QueryUnderstanding,
        _features: &MLFeatures,
    ) -> Option<RulePrediction> {
        if understanding.complexity == QueryComplexity::VeryComplex {
            Some(RulePrediction {
                strategy: RetrievalStrategy::MultiStep,
                confidence: 0.6,
                explanation: "Very complex queries benefit from multi-step retrieval".to_string(),
            })
        } else if understanding.complexity == QueryComplexity::Complex {
            Some(RulePrediction {
                strategy: RetrievalStrategy::Hybrid,
                confidence: 0.7,
                explanation: "Complex queries benefit from hybrid approach".to_string(),
            })
        } else {
            None
        }
    }
}

/// Rule for code-related queries
struct CodeQueryRule;

impl StrategyRule for CodeQueryRule {
    fn evaluate(
        &self,
        understanding: &QueryUnderstanding,
        _features: &MLFeatures,
    ) -> Option<RulePrediction> {
        if understanding.query_type == QueryType::Technical
            && understanding.intent == QueryIntent::Code
        {
            Some(RulePrediction {
                strategy: RetrievalStrategy::BM25Only,
                confidence: 0.9,
                explanation: "Code queries require exact matching".to_string(),
            })
        } else {
            None
        }
    }
}

/// Rule for comparison queries
struct ComparisonQueryRule;

impl StrategyRule for ComparisonQueryRule {
    fn evaluate(
        &self,
        understanding: &QueryUnderstanding,
        _features: &MLFeatures,
    ) -> Option<RulePrediction> {
        if understanding.query_type == QueryType::Comparative {
            Some(RulePrediction {
                strategy: RetrievalStrategy::HydeEnhanced,
                confidence: 0.6,
                explanation: "Comparison queries benefit from hypothetical document expansion"
                    .to_string(),
            })
        } else {
            None
        }
    }
}

impl MLFeatures {
    fn get_feature_names(&self) -> Vec<String> {
        FEATURE_NAMES.iter().map(|s| s.to_string()).collect()
    }
}

fn strategy_to_string(strategy: &RetrievalStrategy) -> &'static str {
    STRATEGY_NAMES
        .iter()
        .find(|(s, _)| s == strategy)
        .map(|(_, name)| *name)
        .unwrap_or("Unknown")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_understanding::{QueryDomain, QueryFeatures};

    fn create_test_understanding(
        query_type: QueryType,
        intent: QueryIntent,
        complexity: QueryComplexity,
    ) -> QueryUnderstanding {
        let (technical_terms, has_code) = match query_type {
            QueryType::Technical => (vec!["code".to_string(), "api".to_string()], true),
            QueryType::Analytical => (vec![], false),
            _ => (vec!["term".to_string()], false),
        };

        QueryUnderstanding {
            original_query: "test query".to_string(),
            query_type,
            intent,
            complexity,
            domain: QueryDomain {
                primary_domain: "programming".to_string(),
                secondary_domains: vec![],
                confidence: 0.8,
            },
            entities: vec![],
            features: QueryFeatures {
                word_count: 5,
                sentence_count: 1,
                question_words: vec!["what".to_string()],
                technical_terms,
                has_code,
                has_numbers: false,
                has_dates: false,
                language: "en".to_string(),
            },
            keywords: vec!["test".to_string(), "query".to_string()],
            confidence: 0.8,
        }
    }

    #[test]
    fn test_technical_query_prediction() {
        let service = MLPredictionService::default();
        let understanding = create_test_understanding(
            QueryType::Technical,
            QueryIntent::Code,
            QueryComplexity::Medium,
        );

        let result = service.predict_strategy(&understanding).unwrap();
        assert_eq!(result.prediction.strategy, RetrievalStrategy::BM25Only);
        assert!(result.prediction.confidence > 0.5);
    }

    #[test]
    fn test_complex_query_prediction() {
        let service = MLPredictionService::default();
        let understanding = create_test_understanding(
            QueryType::Analytical,
            QueryIntent::Explain,
            QueryComplexity::VeryComplex,
        );

        let result = service.predict_strategy(&understanding).unwrap();
        // Should prefer multi-step or hybrid for very complex queries
        assert!(matches!(
            result.prediction.strategy,
            RetrievalStrategy::MultiStep | RetrievalStrategy::Hybrid
        ));
    }

    #[test]
    fn test_feature_extraction() {
        let service = MLPredictionService::default();
        let understanding = create_test_understanding(
            QueryType::Technical,
            QueryIntent::Code,
            QueryComplexity::Complex,
        );

        let features = service.extract_features(&understanding);
        assert!(features.has_code > 0.0);
        assert!(features.query_complexity_score > 0.5);
        assert!(features.technical_term_count > 0.0);
    }

    #[test]
    fn test_explanation_generation() {
        let service = MLPredictionService::default();
        let understanding = create_test_understanding(
            QueryType::Technical,
            QueryIntent::Code,
            QueryComplexity::Medium,
        );

        let result = service.predict_strategy(&understanding).unwrap();
        assert!(!result.explanation.is_empty());
        assert!(result.explanation.contains("strategy"));
    }

    #[test]
    fn test_feature_importance() {
        let service = MLPredictionService::default();
        let understanding = create_test_understanding(
            QueryType::Technical,
            QueryIntent::Code,
            QueryComplexity::Medium,
        );

        let result = service.predict_strategy(&understanding).unwrap();
        assert!(!result.feature_importance.is_empty());
        assert!(result.feature_importance.contains_key("complexity"));
    }
}
