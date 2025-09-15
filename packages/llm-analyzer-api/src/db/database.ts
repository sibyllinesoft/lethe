import Database from 'better-sqlite3';
import { readFileSync } from 'fs';
import { join } from 'path';
import type { LLMCall, LLMCallSummary } from '@lethe/llm-analyzer-shared';

export class DatabaseManager {
  private db: Database.Database;

  constructor(dbPath: string = './data/llm-analyzer.db') {
    this.db = new Database(dbPath);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');
    
    this.initializeSchema();
  }

  private initializeSchema() {
    const schema = readFileSync(join(__dirname, 'schema.sql'), 'utf-8');
    this.db.exec(schema);
  }

  // Call operations
  insertCall(call: LLMCall): void {
    const stmt = this.db.prepare(`
      INSERT OR REPLACE INTO llm_calls (
        id, timestamp, provider, model, endpoint, method, status,
        request_headers, request_body, response_headers, response_body,
        duration, input_tokens, output_tokens, total_tokens, cost,
        user_id, session_id, tags, error
      ) VALUES (
        ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?, ?,
        ?, ?, ?, ?
      )
    `);

    const tagsJson = JSON.stringify(call.tags);
    const errorJson = call.error ? JSON.stringify(call.error) : null;

    stmt.run(
      call.id,
      call.timestamp,
      call.provider,
      call.model,
      call.endpoint,
      call.method,
      call.status,
      JSON.stringify(call.requestHeaders),
      JSON.stringify(call.requestBody),
      JSON.stringify(call.responseHeaders),
      JSON.stringify(call.responseBody),
      call.duration,
      call.inputTokens,
      call.outputTokens,
      call.totalTokens,
      call.cost,
      call.userId,
      call.sessionId,
      tagsJson,
      errorJson
    );

    // Insert tags
    if (call.tags.length > 0) {
      const tagStmt = this.db.prepare('INSERT OR IGNORE INTO call_tags (call_id, tag) VALUES (?, ?)');
      const insertTags = this.db.transaction((tags: string[]) => {
        for (const tag of tags) {
          tagStmt.run(call.id, tag);
        }
      });
      insertTags(call.tags);
    }
  }

  getCall(id: string): LLMCall | null {
    const stmt = this.db.prepare(`
      SELECT * FROM llm_calls WHERE id = ?
    `);
    
    const row = stmt.get(id) as any;
    if (!row) return null;

    return this.rowToCall(row);
  }

  getCalls(options: {
    page?: number;
    limit?: number;
    provider?: string[];
    model?: string[];
    status?: number[];
    dateFrom?: string;
    dateTo?: string;
    tags?: string[];
    hasError?: boolean;
    minDuration?: number;
    maxDuration?: number;
    sortBy?: string;
    sortOrder?: 'asc' | 'desc';
  } = {}): { calls: LLMCallSummary[]; total: number } {
    const {
      page = 1,
      limit = 50,
      provider,
      model,
      status,
      dateFrom,
      dateTo,
      tags,
      hasError,
      minDuration,
      maxDuration,
      sortBy = 'timestamp',
      sortOrder = 'desc'
    } = options;

    let query = 'SELECT * FROM llm_calls';
    let countQuery = 'SELECT COUNT(*) as count FROM llm_calls';
    const params: any[] = [];
    const conditions: string[] = [];

    // Build WHERE conditions
    if (provider?.length) {
      conditions.push(`provider IN (${provider.map(() => '?').join(', ')})`);
      params.push(...provider);
    }

    if (model?.length) {
      conditions.push(`model IN (${model.map(() => '?').join(', ')})`);
      params.push(...model);
    }

    if (status?.length) {
      conditions.push(`status IN (${status.map(() => '?').join(', ')})`);
      params.push(...status);
    }

    if (dateFrom) {
      conditions.push('timestamp >= ?');
      params.push(dateFrom);
    }

    if (dateTo) {
      conditions.push('timestamp <= ?');
      params.push(dateTo);
    }

    if (hasError !== undefined) {
      conditions.push(hasError ? 'error IS NOT NULL' : 'error IS NULL');
    }

    if (minDuration !== undefined) {
      conditions.push('duration >= ?');
      params.push(minDuration);
    }

    if (maxDuration !== undefined) {
      conditions.push('duration <= ?');
      params.push(maxDuration);
    }

    if (tags?.length) {
      conditions.push(`id IN (SELECT call_id FROM call_tags WHERE tag IN (${tags.map(() => '?').join(', ')}))`);
      params.push(...tags);
    }

    const whereClause = conditions.length > 0 ? ` WHERE ${conditions.join(' AND ')}` : '';
    query += whereClause;
    countQuery += whereClause;

    // Add sorting
    const validSortColumns = ['timestamp', 'duration', 'input_tokens', 'output_tokens', 'total_tokens', 'cost'];
    const sortColumn = validSortColumns.includes(sortBy) ? sortBy : 'timestamp';
    const sortDirection = sortOrder === 'asc' ? 'ASC' : 'DESC';
    query += ` ORDER BY ${sortColumn} ${sortDirection}`;

    // Add pagination
    const offset = (page - 1) * limit;
    query += ' LIMIT ? OFFSET ?';
    params.push(limit, offset);

    // Get total count
    const countResult = this.db.prepare(countQuery).get(...params.slice(0, -2)) as { count: number };
    const total = countResult.count;

    // Get calls
    const rows = this.db.prepare(query).all(...params) as any[];
    const calls = rows.map(row => this.rowToCallSummary(row));

    return { calls, total };
  }

  private rowToCall(row: any): LLMCall {
    return {
      id: row.id,
      timestamp: row.timestamp,
      provider: row.provider,
      model: row.model,
      endpoint: row.endpoint,
      method: row.method,
      status: row.status,
      requestHeaders: JSON.parse(row.request_headers),
      requestBody: JSON.parse(row.request_body),
      responseHeaders: JSON.parse(row.response_headers),
      responseBody: JSON.parse(row.response_body),
      duration: row.duration,
      inputTokens: row.input_tokens,
      outputTokens: row.output_tokens,
      totalTokens: row.total_tokens,
      cost: row.cost,
      userId: row.user_id,
      sessionId: row.session_id,
      tags: JSON.parse(row.tags || '[]'),
      error: row.error ? JSON.parse(row.error) : undefined
    };
  }

  private rowToCallSummary(row: any): LLMCallSummary {
    return {
      id: row.id,
      timestamp: row.timestamp,
      provider: row.provider,
      model: row.model,
      status: row.status,
      duration: row.duration,
      inputTokens: row.input_tokens,
      outputTokens: row.output_tokens,
      totalTokens: row.total_tokens,
      cost: row.cost,
      error: row.error ? JSON.parse(row.error).message : undefined,
      tags: JSON.parse(row.tags || '[]')
    };
  }

  close() {
    this.db.close();
  }
}