/**
 * @fileoverview Session-IDF implementation for Lethe agent context atoms
 * Implements the formula: idf_session(t) = log((N - df + 0.5)/(df + 0.5))
 */

import type Database from 'better-sqlite3';

type DB = Database.Database;
import { SessionIdf, SessionIdfConfig } from './atoms-types.js';

/**
 * Default session IDF configuration
 */
export const DEFAULT_SESSION_IDF_CONFIG: SessionIdfConfig = {
  smoothing: 0.5,
  minDf: 1,
  incremental: true,
  stopWords: new Set([
    // Common stop words to exclude from IDF calculation
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall', 'must',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
  ]),
};

/**
 * Session IDF calculator and manager
 */
export class SessionIdfCalculator {
  private db: DB;
  private config: SessionIdfConfig;

  constructor(db: DB, config: SessionIdfConfig = DEFAULT_SESSION_IDF_CONFIG) {
    this.db = db;
    this.config = config;
  }

  /**
   * Tokenize text for IDF calculation
   */
  private tokenize(text: string): string[] {
    // Simple tokenization - split on whitespace and punctuation
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ') // Replace punctuation with spaces
      .split(/\s+/)
      .filter(token => 
        token.length > 2 && // Minimum length
        !this.config.stopWords?.has(token) && // Not a stop word
        !/^\d+$/.test(token) // Not pure numbers
      );
  }

  /**
   * Calculate session IDF for a term
   * Formula: idf_session(t) = log((N - df + smoothing) / (df + smoothing))
   */
  private calculateIdf(totalAtoms: number, documentFrequency: number): number {
    const smoothing = this.config.smoothing;
    const numerator = totalAtoms - documentFrequency + smoothing;
    const denominator = documentFrequency + smoothing;
    
    return Math.log(numerator / denominator);
  }

  /**
   * Update session IDF statistics for new atom
   */
  async updateSessionIdf(sessionId: string, text: string): Promise<void> {
    const tokens = this.tokenize(text);
    const uniqueTokens = new Set(tokens); // Only count once per atom
    
    if (uniqueTokens.size === 0) {
      return; // No valid tokens
    }

    // Get current session atom count
    const sessionCountStmt = this.db.prepare(
      'SELECT COUNT(*) as count FROM atoms WHERE session_id = ?'
    );
    const sessionCount = (sessionCountStmt.get(sessionId) as { count: number }).count;

    // Update document frequencies and IDF values
    const updateIdfStmt = this.db.prepare(`
      INSERT INTO session_idf (session_id, term, df, idf, updated_at)
      VALUES (?, ?, 1, ?, unixepoch())
      ON CONFLICT (session_id, term) DO UPDATE SET
        df = df + 1,
        idf = ?,
        updated_at = unixepoch()
    `);

    const transaction = this.db.transaction((tokens: Set<string>) => {
      for (const token of tokens) {
        // Get current DF for this term
        const currentDfStmt = this.db.prepare(
          'SELECT df FROM session_idf WHERE session_id = ? AND term = ?'
        );
        const currentDf = (currentDfStmt.get(sessionId, token) as { df: number } | undefined)?.df || 0;
        
        // Calculate new IDF
        const newDf = currentDf + 1;
        const newIdf = this.calculateIdf(sessionCount, newDf);
        
        updateIdfStmt.run(sessionId, token, newIdf, newIdf);
      }
    });

    transaction(uniqueTokens);
  }

  /**
   * Recompute all IDF values for a session
   */
  async recomputeSessionIdf(sessionId: string): Promise<void> {
    // Get all atoms for the session
    const atomsStmt = this.db.prepare(
      'SELECT id, text FROM atoms WHERE session_id = ? ORDER BY turn_idx'
    );
    const atoms = atomsStmt.all(sessionId) as Array<{ id: string; text: string }>;
    
    if (atoms.length === 0) {
      return;
    }

    // Clear existing IDF data for session
    const clearStmt = this.db.prepare(
      'DELETE FROM session_idf WHERE session_id = ?'
    );
    clearStmt.run(sessionId);

    // Count document frequency for each term
    const termDocCounts = new Map<string, number>();
    
    for (const atom of atoms) {
      const tokens = this.tokenize(atom.text);
      const uniqueTokens = new Set(tokens);
      
      for (const token of uniqueTokens) {
        termDocCounts.set(token, (termDocCounts.get(token) || 0) + 1);
      }
    }

    // Calculate and insert IDF values
    const insertIdfStmt = this.db.prepare(
      'INSERT INTO session_idf (session_id, term, df, idf, updated_at) VALUES (?, ?, ?, ?, unixepoch())'
    );

    const transaction = this.db.transaction((termCounts: Map<string, number>) => {
      for (const [term, df] of termCounts) {
        if (df >= this.config.minDf) {
          const idf = this.calculateIdf(atoms.length, df);
          insertIdfStmt.run(sessionId, term, df, idf);
        }
      }
    });

    transaction(termDocCounts);
  }

  /**
   * Get IDF weights for terms in a session
   */
  getSessionIdfWeights(sessionId: string, terms?: string[]): Map<string, number> {
    let query = 'SELECT term, idf FROM session_idf WHERE session_id = ?';
    const params: any[] = [sessionId];

    if (terms && terms.length > 0) {
      query += ` AND term IN (${terms.map(() => '?').join(',')})`;
      params.push(...terms);
    }

    const stmt = this.db.prepare(query);
    const results = stmt.all(...params) as Array<{ term: string; idf: number }>;
    
    const weights = new Map<string, number>();
    for (const result of results) {
      weights.set(result.term, result.idf);
    }
    
    return weights;
  }

  /**
   * Get top terms by IDF for a session
   */
  getTopTermsByIdf(sessionId: string, limit: number = 50): Array<{ term: string; idf: number; df: number }> {
    const stmt = this.db.prepare(`
      SELECT term, idf, df 
      FROM session_idf 
      WHERE session_id = ? 
      ORDER BY idf DESC 
      LIMIT ?
    `);
    
    return stmt.all(sessionId, limit) as Array<{ term: string; idf: number; df: number }>;
  }

  /**
   * Get session IDF statistics
   */
  getSessionIdfStats(sessionId: string): {
    totalTerms: number;
    avgIdf: number;
    maxIdf: number;
    minIdf: number;
    medianIdf: number;
  } {
    const stmt = this.db.prepare(`
      SELECT 
        COUNT(*) as totalTerms,
        AVG(idf) as avgIdf,
        MAX(idf) as maxIdf,
        MIN(idf) as minIdf
      FROM session_idf 
      WHERE session_id = ?
    `);
    
    const result = stmt.get(sessionId) as {
      totalTerms: number;
      avgIdf: number;
      maxIdf: number;
      minIdf: number;
    };

    // Calculate median
    const medianStmt = this.db.prepare(`
      SELECT idf 
      FROM session_idf 
      WHERE session_id = ? 
      ORDER BY idf 
      LIMIT 1 
      OFFSET ?
    `);
    
    const medianOffset = Math.floor(result.totalTerms / 2);
    const medianResult = medianStmt.get(sessionId, medianOffset) as { idf: number } | undefined;
    const medianIdf = medianResult?.idf || 0;

    return {
      ...result,
      medianIdf,
    };
  }

  /**
   * Clean up old IDF data (for maintenance)
   */
  cleanupOldIdf(olderThanDays: number = 30): void {
    const cutoffTs = Math.floor(Date.now() / 1000) - (olderThanDays * 24 * 60 * 60);
    
    const stmt = this.db.prepare(
      'DELETE FROM session_idf WHERE updated_at < ?'
    );
    
    const result = stmt.run(cutoffTs);
    console.log(`Cleaned up ${result.changes} old session IDF entries`);
  }

  /**
   * Get IDF weight for a specific term and session
   */
  getTermIdf(sessionId: string, term: string): number {
    const stmt = this.db.prepare(
      'SELECT idf FROM session_idf WHERE session_id = ? AND term = ?'
    );
    
    const result = stmt.get(sessionId, term.toLowerCase()) as { idf: number } | undefined;
    return result?.idf || 0;
  }

  /**
   * Batch update IDF for multiple texts in a session
   */
  async batchUpdateSessionIdf(sessionId: string, texts: string[]): Promise<void> {
    if (texts.length === 0) return;

    // Collect all unique tokens from all texts
    const allTokenCounts = new Map<string, number>();
    
    for (const text of texts) {
      const tokens = this.tokenize(text);
      const uniqueTokens = new Set(tokens);
      
      for (const token of uniqueTokens) {
        allTokenCounts.set(token, (allTokenCounts.get(token) || 0) + 1);
      }
    }

    // Get total atoms in session (including new ones)
    const sessionCountStmt = this.db.prepare(
      'SELECT COUNT(*) as count FROM atoms WHERE session_id = ?'
    );
    const sessionCount = (sessionCountStmt.get(sessionId) as { count: number }).count + texts.length;

    // Batch update IDF values
    const updateIdfStmt = this.db.prepare(`
      INSERT INTO session_idf (session_id, term, df, idf, updated_at)
      VALUES (?, ?, ?, ?, unixepoch())
      ON CONFLICT (session_id, term) DO UPDATE SET
        df = df + ?,
        idf = ?,
        updated_at = unixepoch()
    `);

    const transaction = this.db.transaction((tokenCounts: Map<string, number>) => {
      for (const [token, newDf] of tokenCounts) {
        // Get current DF
        const currentDfStmt = this.db.prepare(
          'SELECT df FROM session_idf WHERE session_id = ? AND term = ?'
        );
        const currentDf = (currentDfStmt.get(sessionId, token) as { df: number } | undefined)?.df || 0;
        
        const totalDf = currentDf + newDf;
        const newIdf = this.calculateIdf(sessionCount, totalDf);
        
        updateIdfStmt.run(sessionId, token, totalDf, newIdf, newDf, newIdf);
      }
    });

    transaction(allTokenCounts);
  }
}

/**
 * Utility function to create session IDF calculator
 */
export function createSessionIdfCalculator(
  db: DB,
  config?: Partial<SessionIdfConfig>
): SessionIdfCalculator {
  const fullConfig = {
    ...DEFAULT_SESSION_IDF_CONFIG,
    ...config,
    stopWords: new Set([
      ...DEFAULT_SESSION_IDF_CONFIG.stopWords!,
      ...(config?.stopWords || []),
    ]),
  };
  
  return new SessionIdfCalculator(db, fullConfig);
}