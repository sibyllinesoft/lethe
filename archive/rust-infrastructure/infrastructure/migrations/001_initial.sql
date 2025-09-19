-- Initial database schema for Lethe
-- This is a placeholder migration file to satisfy sqlx::migrate! macro

-- Messages table for storing chat messages
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    turn INTEGER NOT NULL,
    role VARCHAR(50) NOT NULL,
    text TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    meta JSONB DEFAULT '{}'
);

-- Chunks table for storing text chunks for RAG
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Embeddings table for storing vector embeddings
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY,
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    embedding BYTEA NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Session state table for storing session-specific state
CREATE TABLE IF NOT EXISTS session_state (
    session_id VARCHAR(255) NOT NULL,
    state_key VARCHAR(255) NOT NULL,
    state_value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (session_id, state_key)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);
CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON chunks(created_at);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model_name ON embeddings(model_name);
CREATE INDEX IF NOT EXISTS idx_session_state_session_id ON session_state(session_id);