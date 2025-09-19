use crate::query_understanding::{QueryComplexity, QueryIntent, QueryType, QueryUnderstanding};
use lethe_shared::{LetheError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const BUNDLED_RULES_YAML: &str = include_str!("../../../config/ml_strategy_rules.yaml");

#[derive(Debug, Clone)]
struct MlStaticRules {
    feature_weights: HashMap<String, f32>,
    strategy_weights: HashMap<RetrievalStrategy, f32>,
    feature_names: Vec<String>,
    strategy_labels: HashMap<RetrievalStrategy, String>,
    complexity_scores: HashMap<QueryComplexity, f32>,
    intent_scores: HashMap<QueryIntent, f32>,
    feature_rules: Vec<FeatureRule>,
}

impl MlStaticRules {
    fn from_yaml_str(yaml: &str) -> Result<Self> {
        let raw: MlStaticRulesRaw = serde_yaml::from_str(yaml).map_err(|err| {
            LetheError::config(format!("Failed to parse ML strategy rules: {err}"))
        })?;
        Self::try_from(raw)
    }

    fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = fs::read_to_string(&path).map_err(|err| {
            LetheError::config(format!(
                "Failed to read ML strategy rules from {}: {}",
                path.as_ref().display(),
                err
            ))
        })?;
        Self::from_yaml_str(&contents)
    }

    fn bundled() -> Result<Self> {
        Self::from_yaml_str(BUNDLED_RULES_YAML)
    }
}

impl TryFrom<MlStaticRulesRaw> for MlStaticRules {
    type Error = LetheError;

    fn try_from(value: MlStaticRulesRaw) -> Result<Self> {
        let feature_weights = value.feature_weights;

        let strategy_weights = value
            .strategy_weights
            .into_iter()
            .map(|(name, weight)| {
                parse_strategy(&name)
                    .map(|strategy| (strategy, weight))
                    .map_err(|err| {
                        LetheError::config(format!(
                            "Unknown strategy '{name}' in strategy_weights: {err}"
                        ))
                    })
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let strategy_labels = value
            .strategy_labels
            .into_iter()
            .map(|(name, label)| {
                parse_strategy(&name)
                    .map(|strategy| (strategy, label))
                    .map_err(|err| {
                        LetheError::config(format!(
                            "Unknown strategy '{name}' in strategy_labels: {err}"
                        ))
                    })
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let complexity_scores = value
            .complexity_scores
            .into_iter()
            .map(|(name, score)| {
                parse_complexity(&name)
                    .map(|complexity| (complexity, score))
                    .map_err(|err| {
                        LetheError::config(format!(
                            "Unknown complexity '{name}' in complexity_scores: {err}"
                        ))
                    })
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let intent_scores = value
            .intent_scores
            .into_iter()
            .map(|(name, score)| {
                parse_intent(&name)
                    .map(|intent| (intent, score))
                    .map_err(|err| {
                        LetheError::config(format!(
                            "Unknown intent '{name}' in intent_scores: {err}"
                        ))
                    })
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let feature_rules = value
            .rules
            .into_iter()
            .map(|raw| FeatureRule::try_from(raw))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            feature_weights,
            strategy_weights,
            feature_names: value.feature_names,
            strategy_labels,
            complexity_scores,
            intent_scores,
            feature_rules,
        })
    }
}

#[derive(Debug, Deserialize)]
struct MlStaticRulesRaw {
    feature_weights: HashMap<String, f32>,
    strategy_weights: HashMap<String, f32>,
    feature_names: Vec<String>,
    strategy_labels: HashMap<String, String>,
    complexity_scores: HashMap<String, f32>,
    intent_scores: HashMap<String, f32>,
    rules: Vec<FeatureRuleRaw>,
}

#[derive(Debug, Deserialize)]
struct FeatureRuleRaw {
    #[serde(default)]
    id: Option<String>,
    strategy: String,
    score: f32,
    #[serde(default)]
    mode: ConditionMode,
    conditions: Vec<FeatureConditionRaw>,
}

#[derive(Debug, Deserialize)]
struct FeatureConditionRaw {
    feature: String,
    operator: ComparisonOperator,
    threshold: f32,
}

#[derive(Debug, Clone)]
struct FeatureRule {
    _id: Option<String>,
    strategy: RetrievalStrategy,
    score: f32,
    mode: ConditionMode,
    conditions: Vec<FeatureCondition>,
}

impl TryFrom<FeatureRuleRaw> for FeatureRule {
    type Error = LetheError;

    fn try_from(raw: FeatureRuleRaw) -> Result<Self> {
        let strategy = parse_strategy(&raw.strategy).map_err(|err| {
            LetheError::config(format!(
                "Unknown strategy '{}' in feature rule: {err}",
                raw.strategy
            ))
        })?;

        let conditions = raw
            .conditions
            .into_iter()
            .map(|condition| FeatureCondition::try_from(condition))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            _id: raw.id,
            strategy,
            score: raw.score,
            mode: raw.mode,
            conditions,
        })
    }
}

impl FeatureRule {
    fn matches(&self, features: &MLFeatures) -> bool {
        if self.conditions.is_empty() {
            return false;
        }
        let evaluations = self
            .conditions
            .iter()
            .map(|condition| condition.evaluate(features))
            .collect::<Vec<bool>>();

        match self.mode {
            ConditionMode::All => evaluations.into_iter().all(|result| result),
            ConditionMode::Any => evaluations.into_iter().any(|result| result),
        }
    }
}

#[derive(Debug, Clone)]
struct FeatureCondition {
    feature: String,
    operator: ComparisonOperator,
    threshold: f32,
}

impl FeatureCondition {
    fn evaluate(&self, features: &MLFeatures) -> bool {
        let value = features.value_by_name(&self.feature).unwrap_or(0.0);
        self.operator.evaluate(value, self.threshold)
    }

    fn try_from(raw: FeatureConditionRaw) -> Result<Self> {
        Ok(Self {
            feature: raw.feature,
            operator: raw.operator,
            threshold: raw.threshold,
        })
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ConditionMode {
    Any,
    All,
}

impl Default for ConditionMode {
    fn default() -> Self {
        ConditionMode::All
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
enum ComparisonOperator {
    #[serde(rename = ">")]
    GreaterThan,
    #[serde(rename = ">=")]
    GreaterOrEqual,
    #[serde(rename = "<")]
    LessThan,
    #[serde(rename = "<=")]
    LessOrEqual,
}

impl ComparisonOperator {
    fn evaluate(self, value: f32, threshold: f32) -> bool {
        match self {
            ComparisonOperator::GreaterThan => value > threshold,
            ComparisonOperator::GreaterOrEqual => value >= threshold,
            ComparisonOperator::LessThan => value < threshold,
            ComparisonOperator::LessOrEqual => value <= threshold,
        }
    }
}

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
}

impl Default for MLPredictionConfig {
    fn default() -> Self {
        Self {
            enable_hybrid_fallback: true,
            confidence_threshold: 0.7,
        }
    }
}

/// ML prediction service for retrieval strategy selection
pub struct MLPredictionService {
    _config: MLPredictionConfig,
    strategy_rules: Vec<Box<dyn StrategyRule>>,
    feature_rules: Vec<FeatureRule>,
    feature_weights: HashMap<String, f32>,
    strategy_weights: HashMap<RetrievalStrategy, f32>,
    feature_names: Vec<String>,
    strategy_labels: HashMap<RetrievalStrategy, String>,
    complexity_scores: HashMap<QueryComplexity, f32>,
    intent_scores: HashMap<QueryIntent, f32>,
}

impl MLPredictionService {
    fn with_rules(config: MLPredictionConfig, rules: MlStaticRules) -> Self {
        let mut service = Self {
            _config: config,
            strategy_rules: Vec::new(),
            feature_rules: rules.feature_rules,
            feature_weights: rules.feature_weights,
            strategy_weights: rules.strategy_weights,
            feature_names: rules.feature_names,
            strategy_labels: rules.strategy_labels,
            complexity_scores: rules.complexity_scores,
            intent_scores: rules.intent_scores,
        };

        service.initialize_rules();
        service
    }

    pub fn from_rules_path(config: MLPredictionConfig, path: Option<&Path>) -> Result<Self> {
        let rules = match path {
            Some(p) => MlStaticRules::load_from_path(p)?,
            None => MlStaticRules::bundled()?,
        };
        Ok(Self::with_rules(config, rules))
    }

    /// Predict the best retrieval strategy for a given query understanding
    pub fn predict_strategy(
        &self,
        understanding: &QueryUnderstanding,
    ) -> Result<MLPredictionResult> {
        let features = self.extract_features(understanding);
        let (strategy_scores, explanations) =
            self.collect_strategy_scores(understanding, &features);
        let prediction = self.create_prediction_from_scores(strategy_scores);
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

        for (strategy, weight) in &self.strategy_weights {
            if let Some(score) = strategy_scores.get_mut(strategy) {
                *score *= *weight;
            }
        }

        (strategy_scores, explanations)
    }

    /// Create prediction from strategy scores
    fn create_prediction_from_scores(
        &self,
        strategy_scores: HashMap<RetrievalStrategy, f32>,
    ) -> RetrievalStrategyPrediction {
        let (best_strategy, best_score) = self.select_best_strategy(&strategy_scores);
        let total_score: f32 = strategy_scores.values().sum();
        let alternatives = self.create_alternatives(strategy_scores, &best_strategy, total_score);

        RetrievalStrategyPrediction {
            strategy: best_strategy,
            confidence: (best_score / total_score).min(1.0),
            features_used: self.feature_names.clone(),
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

        let query_complexity_score = self
            .complexity_scores
            .get(&understanding.complexity)
            .copied()
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

        let intent_score = self
            .intent_scores
            .get(&understanding.intent)
            .copied()
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
        for rule in &self.feature_rules {
            if rule.matches(features) {
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

    fn strategy_label(&self, strategy: &RetrievalStrategy) -> &str {
        self.strategy_labels
            .get(strategy)
            .map(|s| s.as_str())
            .unwrap_or_else(|| default_strategy_label(strategy))
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
            self.strategy_label(&prediction.strategy),
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

        for (feature, weight) in &self.feature_weights {
            let value = features.value_by_name(feature).unwrap_or(0.0);
            importance.insert(feature.clone(), value * *weight);
        }

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
        Self::with_rules(
            MLPredictionConfig::default(),
            MlStaticRules::bundled().expect("Bundled ML strategy rules must be valid"),
        )
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
    fn value_by_name(&self, name: &str) -> Option<f32> {
        match name {
            "query_length" => Some(self.query_length),
            "query_complexity_score" => Some(self.query_complexity_score),
            "technical_term_count" => Some(self.technical_term_count),
            "question_word_presence" => Some(self.question_word_presence),
            "domain_specificity" => Some(self.domain_specificity),
            "has_code" => Some(self.has_code),
            "has_numbers" => Some(self.has_numbers),
            "intent_score" => Some(self.intent_score),
            "semantic_complexity" => Some(self.semantic_complexity),
            _ => None,
        }
    }
}

fn default_strategy_label(strategy: &RetrievalStrategy) -> &'static str {
    match strategy {
        RetrievalStrategy::BM25Only => "BM25-only",
        RetrievalStrategy::VectorOnly => "Vector-only",
        RetrievalStrategy::Hybrid => "Hybrid",
        RetrievalStrategy::HydeEnhanced => "HyDE-enhanced",
        RetrievalStrategy::MultiStep => "Multi-step",
        RetrievalStrategy::Adaptive => "Adaptive",
    }
}

fn parse_strategy(name: &str) -> std::result::Result<RetrievalStrategy, String> {
    match normalise_token(name).as_str() {
        "bm25only" | "bm25_only" => Ok(RetrievalStrategy::BM25Only),
        "vectoronly" | "vector_only" => Ok(RetrievalStrategy::VectorOnly),
        "hybrid" => Ok(RetrievalStrategy::Hybrid),
        "hyde" | "hyde_enhanced" => Ok(RetrievalStrategy::HydeEnhanced),
        "multi_step" | "multistep" => Ok(RetrievalStrategy::MultiStep),
        "adaptive" => Ok(RetrievalStrategy::Adaptive),
        other => Err(format!("{other}")),
    }
}

fn parse_complexity(name: &str) -> std::result::Result<QueryComplexity, String> {
    match normalise_token(name).as_str() {
        "simple" => Ok(QueryComplexity::Simple),
        "medium" => Ok(QueryComplexity::Medium),
        "complex" => Ok(QueryComplexity::Complex),
        "very_complex" | "verycomplex" => Ok(QueryComplexity::VeryComplex),
        other => Err(format!("{other}")),
    }
}

fn parse_intent(name: &str) -> std::result::Result<QueryIntent, String> {
    match normalise_token(name).as_str() {
        "search" => Ok(QueryIntent::Search),
        "explain" => Ok(QueryIntent::Explain),
        "code" => Ok(QueryIntent::Code),
        "debug" => Ok(QueryIntent::Debug),
        "compare" => Ok(QueryIntent::Compare),
        "guide" => Ok(QueryIntent::Guide),
        "assist" => Ok(QueryIntent::Assist),
        "chat" => Ok(QueryIntent::Chat),
        other => Err(format!("{other}")),
    }
}

fn normalise_token(input: &str) -> String {
    let mut token = input
        .trim()
        .to_lowercase()
        .replace('-', "_")
        .replace(' ', "_");
    while token.contains("__") {
        token = token.replace("__", "_");
    }
    token
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
