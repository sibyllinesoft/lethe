pub mod chunker;
pub mod retrieval;
pub mod embeddings;
pub mod hyde;
pub mod query_understanding;
pub mod ml_prediction;
pub mod pipeline;
pub mod repository_indexer;

// Re-export all domain services
pub use chunker::*;
pub use retrieval::*;
pub use embeddings::*;
pub use hyde::*;
pub use query_understanding::*;
pub use ml_prediction::*;
pub use pipeline::*;
pub use repository_indexer::{
    RepositoryIndexer, RepositoryIndexerFactory, IndexingResult, IndexingError,
    ChunkingConfig as RepositoryChunkingConfig
};