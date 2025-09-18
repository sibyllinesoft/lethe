use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

/// Core message type representing conversational turns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: Uuid,
    pub session_id: String,
    pub turn: i32,
    pub role: String,
    pub text: String,
    pub ts: DateTime<Utc>,
    pub meta: Option<serde_json::Value>,
}

/// Text chunk from message segmentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub message_id: Uuid,
    pub session_id: String,
    pub offset_start: usize,
    pub offset_end: usize,
    pub kind: String,
    pub text: String,
    pub tokens: i32,
}

/// Document frequency / inverse document frequency data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DfIdf {
    pub term: String,
    pub session_id: String,
    pub df: i32,
    pub idf: f64,
}

/// Search candidate with relevance score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub doc_id: String,
    pub score: f64,
    pub text: Option<String>,
    pub kind: Option<String>,
}

/// Enhanced candidate with sentence-level granularity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedCandidate {
    #[serde(flatten)]
    pub candidate: Candidate,
    pub sentences: Option<Vec<Sentence>>,
    pub pruned_result: Option<PrunedChunkResult>,
}

/// Individual sentence within a chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sentence {
    pub id: String,
    pub text: String,
    pub tokens: i32,
    pub importance: f64,
    pub sentence_index: usize,
    pub is_head_anchor: bool,
    pub is_tail_anchor: bool,
    pub co_entailing_group: Option<Vec<String>>,
}

/// Result of sentence pruning operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrunedChunkResult {
    pub original_sentences: i32,
    pub pruned_sentences: Vec<PrunedSentence>,
    pub total_tokens: i32,
    pub relevance_threshold: f64,
    pub processing_time_ms: f64,
}

/// Individual pruned sentence with relevance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrunedSentence {
    pub sentence_id: String,
    pub text: String,
    pub tokens: i32,
    pub relevance_score: f64,
    pub original_index: usize,
    pub is_code_fence: bool,
    pub co_entailing_ids: Option<Vec<String>>,
}

/// Context pack containing retrieved information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPack {
    pub id: String,
    pub session_id: String,
    pub query: String,
    pub created_at: DateTime<Utc>,
    pub summary: String,
    pub key_entities: Vec<String>,
    pub claims: Vec<String>,
    pub contradictions: Vec<String>,
    pub chunks: Vec<ContextChunk>,
    pub citations: Vec<Citation>,
}

/// Chunk within a context pack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextChunk {
    pub id: String,
    pub score: f64,
    pub kind: String,
    pub text: String,
}

/// Citation reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub id: i32,
    pub chunk_id: String,
    pub relevance: f64,
}

/// Plan selection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanSelection {
    pub plan: String,
    pub reasoning: String,
    pub parameters: PlanParameters,
}

/// Parameters for a selected plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanParameters {
    pub hyde_k: Option<i32>,
    pub beta: Option<f64>,
    pub granularity: Option<String>,
    pub k_final: Option<i32>,
}

/// Session information for tracking conversation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

/// Session state for adaptive planning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub session_id: String,
    pub last_pack_entities: Vec<String>,
    pub last_pack_claims: Vec<String>,
    pub last_pack_contradictions: Vec<String>,
    pub updated_at: DateTime<Utc>,
}

/// Query understanding result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryUnderstanding {
    pub canonical_query: Option<String>,
    pub subqueries: Option<Vec<String>>,
    pub rewrite_success: bool,
    pub decompose_success: bool,
    pub llm_calls_made: i32,
    pub errors: Vec<String>,
}

/// ML prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlPrediction {
    pub alpha: Option<f64>,
    pub beta: Option<f64>,
    pub predicted_plan: Option<String>,
    pub prediction_time_ms: f64,
    pub model_loaded: bool,
}

/// Enhanced query processing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedQueryResult {
    pub pack: ContextPack,
    pub plan: PlanSelection,
    pub hyde_queries: Option<Vec<String>>,
    pub query_understanding: Option<QueryUnderstanding>,
    pub ml_prediction: Option<MlPrediction>,
    pub duration: ProcessingDuration,
    pub debug: DebugInfo,
}

/// Processing time breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingDuration {
    pub total: f64,
    pub query_understanding: Option<f64>,
    pub hyde: Option<f64>,
    pub retrieval: f64,
    pub summarization: Option<f64>,
    pub ml_prediction: Option<f64>,
}

/// Debug information for query processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugInfo {
    pub original_query: String,
    pub final_queries: Vec<String>,
    pub retrieval_candidates: i32,
    pub plan: PlanSelection,
    pub query_processing_enabled: Option<bool>,
    pub rewrite_failure_rate: Option<f64>,
    pub decompose_failure_rate: Option<f64>,
    pub ml_prediction_enabled: Option<bool>,
    pub static_alpha: Option<f64>,
    pub static_beta: Option<f64>,
    pub predicted_alpha: Option<f64>,
    pub predicted_beta: Option<f64>,
}

/// Enhanced query options
#[derive(Debug, Clone, Validate)]
pub struct EnhancedQueryOptions {
    pub session_id: String,
    pub enable_hyde: bool,
    pub enable_summarization: bool,
    pub enable_plan_selection: bool,
    pub enable_query_understanding: bool,
    pub enable_ml_prediction: bool,
    pub recent_turns: Vec<ConversationTurn>,
}

/// Individual conversation turn
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

/// Embedding vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingVector {
    pub data: Vec<f32>,
    pub dimension: usize,
}
