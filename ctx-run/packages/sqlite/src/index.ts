import Database from 'better-sqlite3';
import { readFileSync } from 'fs';
import { join } from 'path';

export type DB = Database.Database;

// Milestone 2: Adaptive Planning Policy - New Exports
export * from './atoms-types.js';
export * from './adaptive-planning.js';
export * from './adaptive-atoms-db.js';
export * from './grid-search.js';
export * from './atoms-db.js';
export * from './entity-extraction.js';
export * from './session-idf.js';

export interface Message {
  id: string;
  session_id: string;
  turn: number;
  role: string;
  text: string;
  ts: number;
  meta?: any;
}

export interface Chunk {
  id: string;
  message_id: string;
  session_id: string;
  offset_start: number;
  offset_end: number;
  kind: string;
  text: string;
  tokens: number;
}

export interface DFIdf {
  term: string;
  session_id: string;
  df: number;
  idf: number;
}

export interface Pack {
  id: string;
  session_id: string;
  created_at: number;
  json: any;
}

export interface State {
  session_id: string;
  json: any;
}

export interface Config {
  key: string;
  value: any;
}

// Core functions as per package contract
export function openDb(path: string): DB {
  return new Database(path);
}

export async function migrate(db: DB): Promise<void> {
  const schemaPath = join(__dirname, '..', 'schema.sql');
  const schema = readFileSync(schemaPath, 'utf-8');
  db.exec(schema);
}

export async function loadVectorExtension(db: DB): Promise<boolean> {
  try {
    // Try loading sqlite-vec first
    db.exec("SELECT load_extension('sqlite-vec')");
    return true;
  } catch {
    try {
      // Try sqlite-vss as fallback
      db.exec("SELECT load_extension('sqlite-vss')");
      return true;
    } catch {
      return false;
    }
  }
}

// CRUD helpers
export function upsertConfig(db: DB, key: string, value: any): void {
  const stmt = db.prepare('INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)');
  stmt.run(key, JSON.stringify(value));
}

export function getConfig(db: DB, key: string): any {
  const stmt = db.prepare('SELECT value FROM config WHERE key = ?');
  const result = stmt.get(key) as { value: string } | undefined;
  return result ? JSON.parse(result.value) : undefined;
}

export function getState(db: DB, sessionId: string): any {
  const stmt = db.prepare('SELECT json FROM state WHERE session_id = ?');
  const result = stmt.get(sessionId) as { json: string } | undefined;
  return result ? JSON.parse(result.json) : null;
}

export function setState(db: DB, sessionId: string, json: any): void {
  const stmt = db.prepare('INSERT OR REPLACE INTO state (session_id, json) VALUES (?, ?)');
  stmt.run(sessionId, JSON.stringify(json));
}

export function insertMessages(db: DB, messages: Message[]): void {
  const stmt = db.prepare('INSERT OR REPLACE INTO messages (id, session_id, turn, role, text, ts, meta) VALUES (?, ?, ?, ?, ?, ?, ?)');
  const transaction = db.transaction((msgs: Message[]) => {
    for (const msg of msgs) {
      stmt.run(msg.id, msg.session_id, msg.turn, msg.role, msg.text, msg.ts, msg.meta ? JSON.stringify(msg.meta) : null);
    }
  });
  transaction(messages);
}

export function insertChunks(db: DB, chunks: Chunk[]): void {
  const stmt = db.prepare('INSERT OR REPLACE INTO chunks (id, message_id, session_id, offset_start, offset_end, kind, text, tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?)');
  const transaction = db.transaction((chks: Chunk[]) => {
    for (const chunk of chks) {
      stmt.run(chunk.id, chunk.message_id, chunk.session_id, chunk.offset_start, chunk.offset_end, chunk.kind, chunk.text, chunk.tokens);
    }
  });
  transaction(chunks);
}

export function insertEmbeddings(db: DB, rows: { chunk_id: string; dim: number; vec: Buffer }[]): void {
  const stmt = db.prepare('INSERT OR REPLACE INTO embeddings (chunk_id, dim, vec) VALUES (?, ?, ?)');
  const transaction = db.transaction((embeddings: typeof rows) => {
    for (const row of embeddings) {
      stmt.run(row.chunk_id, row.dim, row.vec);
    }
  });
  transaction(rows);
}

// Global vector index instances
let nativeVectorSupported: boolean | null = null;
let wasmIndex: any = null; // HnswWasm instance (imported dynamically)

export async function vectorSearch(db: DB, qVec: Float32Array, k: number): Promise<{ docId: string; score: number }[]> {
  // Check if native vector extension is available
  if (nativeVectorSupported === null) {
    nativeVectorSupported = await loadVectorExtension(db);
  }

  if (nativeVectorSupported) {
    // Use native vector extension if available
    try {
      const stmt = db.prepare('SELECT chunk_id as docId, distance as score FROM vec_index WHERE vec MATCH ? ORDER BY distance LIMIT ?');
      const results = stmt.all(qVec, k) as { docId: string; score: number }[];
      
      // Convert distance to similarity score (1 - distance for cosine)
      return results.map(r => ({ docId: r.docId, score: 1 - r.score }));
    } catch (error) {
      console.warn('Native vector search failed, falling back to WASM:', error);
      nativeVectorSupported = false;
    }
  }

  // Fall back to WASM implementation
  if (!wasmIndex) {
    // Import WASM dynamically to avoid circular dependencies
    const { HnswWasm } = await import('@lethe/wasm');
    wasmIndex = new HnswWasm();
    
    // Load all embeddings into WASM index
    const stmt = db.prepare('SELECT chunk_id, vec FROM embeddings');
    const embeddings = stmt.all() as { chunk_id: string; vec: Buffer }[];
    
    for (const embedding of embeddings) {
      const vector = new Float32Array(embedding.vec.buffer, embedding.vec.byteOffset, embedding.vec.byteLength / 4);
      wasmIndex.addVector(embedding.chunk_id, vector);
    }
    
    console.log(`Loaded ${embeddings.length} embeddings into WASM vector index`);
  }

  return wasmIndex.search(qVec, k);
}

// Function to refresh vector index when new embeddings are added
export function refreshVectorIndex(): void {
  if (wasmIndex) {
    wasmIndex.clear();
    wasmIndex = null;
  }
}

export function getChunksBySession(db: DB, sessionId: string): Chunk[] {
  const stmt = db.prepare('SELECT * FROM chunks WHERE session_id = ?');
  return stmt.all(sessionId) as Chunk[];
}

export function getDFIdf(db: DB, sessionId: string): DFIdf[] {
  const stmt = db.prepare('SELECT * FROM dfidf WHERE session_id = ?');
  return stmt.all(sessionId) as DFIdf[];
}

export function getChunkById(db: DB, chunkId: string): Chunk | null {
  const stmt = db.prepare('SELECT * FROM chunks WHERE id = ?');
  const result = stmt.get(chunkId) as Chunk | undefined;
  return result || null;
}
