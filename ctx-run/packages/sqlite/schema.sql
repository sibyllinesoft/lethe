-- Lethe project: SQLite schema

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn INT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    ts INT NOT NULL,
    meta JSON
);

-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    offset_start INT NOT NULL,
    offset_end INT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    tokens INT NOT NULL,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);

-- DF/IDF table (per-session stats)
CREATE TABLE IF NOT EXISTS dfidf (
    term TEXT NOT NULL,
    session_id TEXT NOT NULL,
    df INT NOT NULL,
    idf REAL NOT NULL,
    PRIMARY KEY (term, session_id)
);

-- Embeddings table
-- This might be a virtual table depending on the extension used.
-- We'll define a standard table for now.
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY,
    dim INT NOT NULL,
    vec BLOB NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);

-- Packs table
CREATE TABLE IF NOT EXISTS packs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    created_at INT NOT NULL,
    json JSON NOT NULL
);

-- State table (minimal state card)
CREATE TABLE IF NOT EXISTS state (
    session_id TEXT PRIMARY KEY,
    json JSON NOT NULL
);

-- Config table
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value JSON NOT NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_messages_session_turn ON messages(session_id, turn);
CREATE INDEX IF NOT EXISTS idx_chunks_session_id ON chunks(session_id, id);
CREATE INDEX IF NOT EXISTS idx_chunks_message_id ON chunks(message_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_dim ON embeddings(dim);
CREATE INDEX IF NOT EXISTS idx_dfidf_session_term ON dfidf(session_id, term);
