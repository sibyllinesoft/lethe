pub mod health;
pub mod query;
pub mod messages;
pub mod chunks;
pub mod sessions;
pub mod embeddings;

// Re-export all handlers
pub use health::*;
pub use query::*;
pub use messages::*;
pub use chunks::*;
pub use sessions::*;
pub use embeddings::*;