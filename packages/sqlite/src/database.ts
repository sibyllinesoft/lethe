import Database from 'better-sqlite3';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export interface Message {
  id: string;
  sessionId: string;
  turn: number;
  role: 'user' | 'assistant' | 'system' | 'tool';
  text: string;
  ts: number;
  meta?: any;
}

export interface Chunk {
  id: string;
  messageId: string;
  offsetStart: number;
  offsetEnd: number;
  kind: 'prose' | 'code' | 'tool_result' | 'user_code';
  text: string;
  tokens: number;
}

export interface DfIdfEntry {
  term: string;
  sessionId: string;
  df: number;
  idf: number;
}

export interface Embedding {
  chunkId: string;
  dim: number;
  vec: Float32Array;
}

export interface ContextPack {
  id: string;
  sessionId: string;
  query: string;
  createdAt: number;
  summary: string;
  keyEntities: string[];
  claims: Array<{ text: string; chunks: string[] }>;
  contradictions: Array<{ issue: string; chunks: string[] }>;
  citations: Record<string, { messageId: string; span: [number, number] }>;
}

export interface SessionState {
  sessionId: string;
  keyEntities: string[];
  lastPackClaims: string[];
  lastPackContradictions: string[];
  planHint?: 'explore' | 'verify' | 'exploit';
}

export class CtxDatabase {
  private db: Database.Database;

  constructor(path: string) {
    this.db = new Database(path);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');
  }

  async migrate(): Promise<void> {
    const schema = readFileSync(join(__dirname, 'schema.sql'), 'utf-8');
    this.db.exec(schema);
  }

  async loadVectorExtension(): Promise<boolean> {
    try {
      // Try to load sqlite-vec first
      this.db.loadExtension('sqlite-vec');
      return true;
    } catch (error) {
      try {
        // Fallback to sqlite-vss
        this.db.loadExtension('sqlite-vss');
        return true;
      } catch (fallbackError) {
        console.warn('No SQLite vector extension available, will use WASM fallback');
        return false;
      }
    }
  }

  // Message operations
  insertMessage(message: Message): void {
    const stmt = this.db.prepare(`
      INSERT INTO messages (id, session_id, turn, role, text, ts, meta)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);
    stmt.run(
      message.id,
      message.sessionId,
      message.turn,
      message.role,
      message.text,
      message.ts,
      JSON.stringify(message.meta)
    );
  }

  insertMessages(messages: Message[]): void {
    const stmt = this.db.prepare(`
      INSERT INTO messages (id, session_id, turn, role, text, ts, meta)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);
    const insertMany = this.db.transaction((msgs: Message[]) => {
      for (const msg of msgs) {
        stmt.run(
          msg.id,
          msg.sessionId,
          msg.turn,
          msg.role,
          msg.text,
          msg.ts,
          JSON.stringify(msg.meta)
        );
      }
    });
    insertMany(messages);
  }

  getMessages(sessionId: string): Message[] {
    const stmt = this.db.prepare(`
      SELECT id, session_id, turn, role, text, ts, meta
      FROM messages 
      WHERE session_id = ?
      ORDER BY turn ASC
    `);
    return stmt.all(sessionId).map((row: any) => ({
      id: row.id,
      sessionId: row.session_id,
      turn: row.turn,
      role: row.role,
      text: row.text,
      ts: row.ts,
      meta: row.meta ? JSON.parse(row.meta) : undefined
    }));
  }

  // Chunk operations
  insertChunk(chunk: Chunk): void {
    const stmt = this.db.prepare(`
      INSERT INTO chunks (id, message_id, offset_start, offset_end, kind, text, tokens)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);
    stmt.run(
      chunk.id,
      chunk.messageId,
      chunk.offsetStart,
      chunk.offsetEnd,
      chunk.kind,
      chunk.text,
      chunk.tokens
    );
  }

  insertChunks(chunks: Chunk[]): void {
    const stmt = this.db.prepare(`
      INSERT INTO chunks (id, message_id, offset_start, offset_end, kind, text, tokens)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);
    const insertMany = this.db.transaction((chunks: Chunk[]) => {
      for (const chunk of chunks) {
        stmt.run(
          chunk.id,
          chunk.messageId,
          chunk.offsetStart,
          chunk.offsetEnd,
          chunk.kind,
          chunk.text,
          chunk.tokens
        );
      }
    });
    insertMany(chunks);
  }

  getChunks(sessionId: string): Chunk[] {
    const stmt = this.db.prepare(`
      SELECT c.id, c.message_id, c.offset_start, c.offset_end, c.kind, c.text, c.tokens
      FROM chunks c
      JOIN messages m ON c.message_id = m.id
      WHERE m.session_id = ?
      ORDER BY m.turn ASC, c.offset_start ASC
    `);
    return stmt.all(sessionId).map((row: any) => ({
      id: row.id,
      messageId: row.message_id,
      offsetStart: row.offset_start,
      offsetEnd: row.offset_end,
      kind: row.kind,
      text: row.text,
      tokens: row.tokens
    }));
  }

  getChunksByIds(chunkIds: string[]): Chunk[] {
    if (chunkIds.length === 0) return [];
    const placeholders = chunkIds.map(() => '?').join(',');
    const stmt = this.db.prepare(`
      SELECT id, message_id, offset_start, offset_end, kind, text, tokens
      FROM chunks
      WHERE id IN (${placeholders})
    `);
    return stmt.all(...chunkIds).map((row: any) => ({
      id: row.id,
      messageId: row.message_id,
      offsetStart: row.offset_start,
      offsetEnd: row.offset_end,
      kind: row.kind,
      text: row.text,
      tokens: row.tokens
    }));
  }

  // DF/IDF operations
  updateDfIdf(sessionId: string, termStats: Map<string, { df: number; idf: number }>): void {
    const deleteStmt = this.db.prepare('DELETE FROM dfidf WHERE session_id = ?');
    const insertStmt = this.db.prepare(`
      INSERT INTO dfidf (term, session_id, df, idf) VALUES (?, ?, ?, ?)
    `);
    
    const updateTransaction = this.db.transaction((sessionId: string, stats: Map<string, { df: number; idf: number }>) => {
      deleteStmt.run(sessionId);
      for (const [term, { df, idf }] of stats) {
        insertStmt.run(term, sessionId, df, idf);
      }
    });

    updateTransaction(sessionId, termStats);
  }

  getDfIdf(sessionId: string): Map<string, { df: number; idf: number }> {
    const stmt = this.db.prepare(`
      SELECT term, df, idf FROM dfidf WHERE session_id = ?
    `);
    const results = new Map();
    for (const row of stmt.all(sessionId) as any[]) {
      results.set(row.term, { df: row.df, idf: row.idf });
    }
    return results;
  }

  getTopRareTerms(sessionId: string, n: number): string[] {
    const stmt = this.db.prepare(`
      SELECT term FROM dfidf 
      WHERE session_id = ? 
      ORDER BY idf DESC 
      LIMIT ?
    `);
    return stmt.all(sessionId, n).map((row: any) => row.term);
  }

  getTopCommonTerms(sessionId: string, n: number): string[] {
    const stmt = this.db.prepare(`
      SELECT term FROM dfidf 
      WHERE session_id = ? 
      ORDER BY df DESC 
      LIMIT ?
    `);
    return stmt.all(sessionId, n).map((row: any) => row.term);
  }

  // Embedding operations
  insertEmbedding(embedding: Embedding): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO embeddings (chunk_id, dim, vec)
      VALUES (?, ?, ?)
    `);
    // Convert Float32Array to Buffer for storage
    const buffer = Buffer.from(embedding.vec.buffer);
    stmt.run(embedding.chunkId, embedding.dim, buffer);
  }

  insertEmbeddings(embeddings: Embedding[]): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO embeddings (chunk_id, dim, vec)
      VALUES (?, ?, ?)
    `);
    const insertMany = this.db.transaction((embeddings: Embedding[]) => {
      for (const embedding of embeddings) {
        const buffer = Buffer.from(embedding.vec.buffer);
        stmt.run(embedding.chunkId, embedding.dim, buffer);
      }
    });
    insertMany(embeddings);
  }

  getEmbedding(chunkId: string): Embedding | null {
    const stmt = this.db.prepare(`
      SELECT chunk_id, dim, vec FROM embeddings WHERE chunk_id = ?
    `);
    const row = stmt.get(chunkId) as any;
    if (!row) return null;

    return {
      chunkId: row.chunk_id,
      dim: row.dim,
      vec: new Float32Array(row.vec.buffer)
    };
  }

  getAllEmbeddings(): Embedding[] {
    const stmt = this.db.prepare(`
      SELECT chunk_id, dim, vec FROM embeddings
    `);
    return stmt.all().map((row: any) => ({
      chunkId: row.chunk_id,
      dim: row.dim,
      vec: new Float32Array(row.vec.buffer)
    }));
  }

  getChunksWithoutEmbeddings(sessionId: string): string[] {
    const stmt = this.db.prepare(`
      SELECT c.id FROM chunks c
      JOIN messages m ON c.message_id = m.id
      LEFT JOIN embeddings e ON c.id = e.chunk_id
      WHERE m.session_id = ? AND e.chunk_id IS NULL
    `);
    return stmt.all(sessionId).map((row: any) => row.id);
  }

  // Context pack operations
  insertPack(pack: ContextPack): void {
    const stmt = this.db.prepare(`
      INSERT INTO packs (id, session_id, query, created_at, json)
      VALUES (?, ?, ?, ?, ?)
    `);
    stmt.run(
      pack.id,
      pack.sessionId,
      pack.query,
      pack.createdAt,
      JSON.stringify({
        summary: pack.summary,
        keyEntities: pack.keyEntities,
        claims: pack.claims,
        contradictions: pack.contradictions,
        citations: pack.citations
      })
    );
  }

  getPacks(sessionId: string): ContextPack[] {
    const stmt = this.db.prepare(`
      SELECT id, session_id, query, created_at, json
      FROM packs
      WHERE session_id = ?
      ORDER BY created_at DESC
    `);
    return stmt.all(sessionId).map((row: any) => {
      const data = JSON.parse(row.json);
      return {
        id: row.id,
        sessionId: row.session_id,
        query: row.query,
        createdAt: row.created_at,
        ...data
      };
    });
  }

  // State operations
  updateState(sessionId: string, state: Partial<SessionState>): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO state (session_id, json) VALUES (?, ?)
    `);
    stmt.run(sessionId, JSON.stringify(state));
  }

  getState(sessionId: string): SessionState | null {
    const stmt = this.db.prepare(`
      SELECT json FROM state WHERE session_id = ?
    `);
    const row = stmt.get(sessionId) as any;
    if (!row) return null;
    
    return { sessionId, ...JSON.parse(row.json) };
  }

  // Configuration operations
  setConfig(key: string, value: any): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)
    `);
    stmt.run(key, JSON.stringify(value));
  }

  getConfig(key: string): any {
    const stmt = this.db.prepare(`
      SELECT value FROM config WHERE key = ?
    `);
    const row = stmt.get(key) as any;
    return row ? JSON.parse(row.value) : null;
  }

  getAllConfig(): Record<string, any> {
    const stmt = this.db.prepare(`SELECT key, value FROM config`);
    const result: Record<string, any> = {};
    for (const row of stmt.all() as any[]) {
      result[row.key] = JSON.parse(row.value);
    }
    return result;
  }

  // Utility operations
  close(): void {
    this.db.close();
  }

  // FTS search for BM25
  searchChunks(query: string, sessionId?: string): Array<{ chunkId: string; rank: number }> {
    let sql = `
      SELECT chunk_id, rank 
      FROM chunks_fts 
      WHERE chunks_fts MATCH ?
    `;
    
    const params: any[] = [query];
    
    if (sessionId) {
      sql += ` AND chunk_id IN (
        SELECT c.id FROM chunks c 
        JOIN messages m ON c.message_id = m.id 
        WHERE m.session_id = ?
      )`;
      params.push(sessionId);
    }
    
    sql += ` ORDER BY rank`;
    
    const stmt = this.db.prepare(sql);
    return stmt.all(...params).map((row: any) => ({
      chunkId: row.chunk_id,
      rank: row.rank
    }));
  }
}

export async function openDb(path: string): Promise<CtxDatabase> {
  const db = new CtxDatabase(path);
  await db.migrate();
  await db.loadVectorExtension();
  return db;
}