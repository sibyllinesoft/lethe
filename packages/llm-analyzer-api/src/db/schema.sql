-- LLM calls table
CREATE TABLE IF NOT EXISTS llm_calls (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    status INTEGER NOT NULL,
    
    -- Request data (JSON)
    request_headers TEXT NOT NULL,
    request_body TEXT NOT NULL,
    
    -- Response data (JSON)
    response_headers TEXT NOT NULL,
    response_body TEXT NOT NULL,
    
    -- Metrics
    duration INTEGER NOT NULL, -- milliseconds
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    cost REAL, -- in USD
    
    -- Metadata
    user_id TEXT,
    session_id TEXT,
    tags TEXT, -- JSON array
    
    -- Error information (JSON)
    error TEXT,
    
    -- Indexes for common queries
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_llm_calls_timestamp ON llm_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_calls_provider ON llm_calls(provider);
CREATE INDEX IF NOT EXISTS idx_llm_calls_model ON llm_calls(model);
CREATE INDEX IF NOT EXISTS idx_llm_calls_status ON llm_calls(status);
CREATE INDEX IF NOT EXISTS idx_llm_calls_duration ON llm_calls(duration);
CREATE INDEX IF NOT EXISTS idx_llm_calls_user_id ON llm_calls(user_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_session_id ON llm_calls(session_id);

-- Tags table for easier searching
CREATE TABLE IF NOT EXISTS call_tags (
    call_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (call_id, tag),
    FOREIGN KEY (call_id) REFERENCES llm_calls(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_call_tags_tag ON call_tags(tag);

-- Benchmark runs table
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    
    -- Configuration (JSON)
    config TEXT NOT NULL,
    
    -- Results
    total_calls INTEGER DEFAULT 0,
    completed_calls INTEGER DEFAULT 0,
    failed_calls INTEGER DEFAULT 0,
    start_time DATETIME,
    end_time DATETIME,
    duration INTEGER, -- milliseconds
    
    -- Aggregated metrics (JSON)
    metrics TEXT
);

CREATE INDEX IF NOT EXISTS idx_benchmark_runs_status ON benchmark_runs(status);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_created_at ON benchmark_runs(created_at);

-- Migration tracking
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
);