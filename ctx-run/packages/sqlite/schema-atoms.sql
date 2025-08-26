-- Lethe Agent-Context Manager: Conversation Atoms Schema
-- Milestone 1: Data Model for conversation atoms and indexes

-- Atoms table - Core conversation atoms (user, tool, action, observation, plan)
CREATE TABLE IF NOT EXISTS atoms (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_idx INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool', 'system')),
    type TEXT NOT NULL CHECK (type IN ('message', 'action', 'args', 'observation', 'plan', 'error', 'result')),
    text TEXT NOT NULL,
    json_meta JSON,
    ts INTEGER NOT NULL DEFAULT (unixepoch()),
    
    -- Constraints
    UNIQUE (session_id, turn_idx, type, role)
);

-- Entities table - Extracted entities with weights
CREATE TABLE IF NOT EXISTS entities (
    atom_id TEXT NOT NULL,
    entity TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('id', 'file', 'error', 'api', 'tool', 'person', 'org', 'misc')),
    weight REAL NOT NULL DEFAULT 1.0,
    
    PRIMARY KEY (atom_id, entity, kind),
    FOREIGN KEY (atom_id) REFERENCES atoms(id) ON DELETE CASCADE
);

-- Vectors table - Dense embeddings
CREATE TABLE IF NOT EXISTS vectors (
    atom_id TEXT PRIMARY KEY,
    dim INTEGER NOT NULL,
    blob BLOB NOT NULL,
    
    FOREIGN KEY (atom_id) REFERENCES atoms(id) ON DELETE CASCADE
);

-- Session IDF statistics
CREATE TABLE IF NOT EXISTS session_idf (
    session_id TEXT NOT NULL,
    term TEXT NOT NULL,
    df INTEGER NOT NULL,
    idf REAL NOT NULL,
    updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
    
    PRIMARY KEY (session_id, term)
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS fts_atoms USING fts5 (
    atom_id UNINDEXED,
    text,
    tokenize='porter'
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_atoms_session_turn ON atoms(session_id, turn_idx);
CREATE INDEX IF NOT EXISTS idx_atoms_session_type ON atoms(session_id, type);
CREATE INDEX IF NOT EXISTS idx_atoms_role ON atoms(role);
CREATE INDEX IF NOT EXISTS idx_atoms_ts ON atoms(ts);

CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind);
CREATE INDEX IF NOT EXISTS idx_entities_entity ON entities(entity);
CREATE INDEX IF NOT EXISTS idx_entities_weight ON entities(weight DESC);

CREATE INDEX IF NOT EXISTS idx_vectors_dim ON vectors(dim);

CREATE INDEX IF NOT EXISTS idx_session_idf_session ON session_idf(session_id);
CREATE INDEX IF NOT EXISTS idx_session_idf_idf ON session_idf(idf DESC);

-- Triggers to maintain FTS5 consistency
CREATE TRIGGER IF NOT EXISTS fts_atoms_insert_trigger 
AFTER INSERT ON atoms
BEGIN
    INSERT INTO fts_atoms(atom_id, text) VALUES (NEW.id, NEW.text);
END;

CREATE TRIGGER IF NOT EXISTS fts_atoms_update_trigger 
AFTER UPDATE OF text ON atoms
BEGIN
    UPDATE fts_atoms SET text = NEW.text WHERE atom_id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS fts_atoms_delete_trigger 
AFTER DELETE ON atoms
BEGIN
    DELETE FROM fts_atoms WHERE atom_id = OLD.id;
END;

-- Triggers to maintain vectors consistency
CREATE TRIGGER IF NOT EXISTS vectors_delete_trigger 
AFTER DELETE ON atoms
BEGIN
    DELETE FROM vectors WHERE atom_id = OLD.id;
END;

-- Triggers to maintain entities consistency
CREATE TRIGGER IF NOT EXISTS entities_delete_trigger 
AFTER DELETE ON atoms
BEGIN
    DELETE FROM entities WHERE atom_id = OLD.id;
END;

-- Views for convenience
CREATE VIEW IF NOT EXISTS atoms_with_entities AS
SELECT 
    a.*,
    GROUP_CONCAT(e.entity || ':' || e.kind || ':' || e.weight) as entities
FROM atoms a
LEFT JOIN entities e ON a.id = e.atom_id
GROUP BY a.id;

CREATE VIEW IF NOT EXISTS session_stats AS
SELECT 
    session_id,
    COUNT(*) as total_atoms,
    COUNT(DISTINCT type) as distinct_types,
    COUNT(DISTINCT role) as distinct_roles,
    MIN(ts) as start_ts,
    MAX(ts) as end_ts,
    MAX(turn_idx) as max_turns
FROM atoms
GROUP BY session_id;