-- Core data model for ctx-run context manager
-- Messages represent individual conversation turns
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    text TEXT NOT NULL,
    ts INTEGER NOT NULL,
    meta JSON
);

-- Chunks represent segmented portions of messages for search
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    offset_start INTEGER NOT NULL,
    offset_end INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('prose', 'code', 'tool_result', 'user_code')),
    text TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
);

-- Per-session document frequency and inverse document frequency for BM25
CREATE TABLE IF NOT EXISTS dfidf (
    term TEXT NOT NULL,
    session_id TEXT NOT NULL,
    df INTEGER NOT NULL,
    idf REAL NOT NULL,
    PRIMARY KEY (term, session_id)
);

-- Vector embeddings for semantic search
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

-- Context packs generated for queries
CREATE TABLE IF NOT EXISTS packs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    json JSON NOT NULL
);

-- Minimal state tracking per session
CREATE TABLE IF NOT EXISTS state (
    session_id TEXT PRIMARY KEY,
    json JSON NOT NULL
);

-- System configuration storage
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value JSON NOT NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);
CREATE INDEX IF NOT EXISTS idx_chunks_message ON chunks(message_id);
CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(kind);
CREATE INDEX IF NOT EXISTS idx_dfidf_session ON dfidf(session_id);
CREATE INDEX IF NOT EXISTS idx_dfidf_idf ON dfidf(idf DESC);
CREATE INDEX IF NOT EXISTS idx_packs_session ON packs(session_id);
CREATE INDEX IF NOT EXISTS idx_packs_created ON packs(created_at DESC);

-- Full-text search index for BM25
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id,
    text,
    content='chunks',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(chunk_id, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = old.id;
    INSERT INTO chunks_fts(chunk_id, text) VALUES (new.id, new.text);
END;