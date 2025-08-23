import type { CtxDatabase, Chunk } from '@ctx-run/sqlite';
import * as natural from 'natural';

export class DfIdfBuilder {
  private db: CtxDatabase;
  private tokenizer: any;

  constructor(db: CtxDatabase) {
    this.db = db;
    this.tokenizer = new natural.WordTokenizer();
  }

  async rebuild(sessionId: string): Promise<void> {
    const chunks = this.db.getChunks(sessionId);
    
    if (chunks.length === 0) {
      return;
    }

    // Tokenize all chunks and build term frequency maps
    const documentTerms: Map<string, Set<string>> = new Map();
    const globalTermCounts: Map<string, number> = new Map();

    for (const chunk of chunks) {
      const terms = this.extractTerms(chunk.text);
      const uniqueTerms = new Set(terms);
      
      documentTerms.set(chunk.id, uniqueTerms);
      
      // Count document frequency for each term
      for (const term of uniqueTerms) {
        globalTermCounts.set(term, (globalTermCounts.get(term) || 0) + 1);
      }
    }

    // Calculate IDF for each term
    const N = chunks.length;
    const termStats = new Map<string, { df: number; idf: number }>();

    for (const [term, df] of globalTermCounts) {
      // BM25 IDF formula: log((N - df + 0.5) / (df + 0.5))
      const idf = Math.log((N - df + 0.5) / (df + 0.5));
      termStats.set(term, { df, idf });
    }

    // Store in database
    this.db.updateDfIdf(sessionId, termStats);
  }

  private extractTerms(text: string): string[] {
    // Tokenize text
    let tokens = this.tokenizer.tokenize(text.toLowerCase()) || [];
    
    // Filter out stop words and very short terms
    tokens = tokens.filter((token: string) => 
      token.length >= 2 && 
      !natural.stopwords.includes(token) &&
      /^[a-zA-Z0-9_-]+$/.test(token) // Only alphanumeric and basic symbols
    );
    
    // Apply stemming to normalize terms
    tokens = tokens.map((token: string) => natural.PorterStemmer.stem(token));
    
    return tokens;
  }

  async getTopRareTerms(sessionId: string, n: number): Promise<string[]> {
    return this.db.getTopRareTerms(sessionId, n);
  }

  async getTopCommonTerms(sessionId: string, n: number): Promise<string[]> {
    return this.db.getTopCommonTerms(sessionId, n);
  }

  // Calculate BM25 score for a query against chunks
  async calculateBM25Scores(
    sessionId: string, 
    query: string, 
    chunkIds: string[],
    k1: number = 1.2,
    b: number = 0.75
  ): Promise<Map<string, number>> {
    const chunks = this.db.getChunksByIds(chunkIds);
    const dfIdfMap = this.db.getDfIdf(sessionId);
    const queryTerms = this.extractTerms(query);
    
    if (chunks.length === 0 || queryTerms.length === 0) {
      return new Map();
    }

    // Calculate average document length
    const avgDocLength = chunks.reduce((sum, chunk) => sum + chunk.tokens, 0) / chunks.length;
    
    const scores = new Map<string, number>();

    for (const chunk of chunks) {
      const docTerms = this.extractTerms(chunk.text);
      const termFrequencies = this.getTermFrequencies(docTerms);
      const docLength = chunk.tokens;
      
      let score = 0;

      for (const term of queryTerms) {
        const tf = termFrequencies.get(term) || 0;
        const dfIdfEntry = dfIdfMap.get(term);
        
        if (!dfIdfEntry || tf === 0) {
          continue;
        }

        const { idf } = dfIdfEntry;
        
        // BM25 scoring formula
        const numerator = tf * (k1 + 1);
        const denominator = tf + k1 * (1 - b + b * (docLength / avgDocLength));
        const termScore = idf * (numerator / denominator);
        
        score += termScore;
      }

      scores.set(chunk.id, score);
    }

    return scores;
  }

  private getTermFrequencies(terms: string[]): Map<string, number> {
    const frequencies = new Map<string, number>();
    
    for (const term of terms) {
      frequencies.set(term, (frequencies.get(term) || 0) + 1);
    }
    
    return frequencies;
  }

  // Get terms that are neither too rare nor too common (good for query expansion)
  async getMidFrequencyTerms(sessionId: string, minIdf: number = 1.0, maxIdf: number = 5.0): Promise<string[]> {
    const dfIdfMap = this.db.getDfIdf(sessionId);
    const terms: string[] = [];
    
    for (const [term, stats] of dfIdfMap) {
      if (stats.idf >= minIdf && stats.idf <= maxIdf) {
        terms.push(term);
      }
    }
    
    return terms.sort((a, b) => {
      const aIdf = dfIdfMap.get(a)?.idf || 0;
      const bIdf = dfIdfMap.get(b)?.idf || 0;
      return bIdf - aIdf; // Sort by IDF descending
    });
  }

  // Analyze term overlap between query and session
  async analyzeQueryTerms(sessionId: string, query: string): Promise<{
    covered: string[];
    rare: string[];
    missing: string[];
  }> {
    const queryTerms = this.extractTerms(query);
    const dfIdfMap = this.db.getDfIdf(sessionId);
    
    const covered: string[] = [];
    const rare: string[] = [];
    const missing: string[] = [];
    
    for (const term of queryTerms) {
      const stats = dfIdfMap.get(term);
      if (stats) {
        covered.push(term);
        if (stats.idf > 3.0) { // High IDF = rare term
          rare.push(term);
        }
      } else {
        missing.push(term);
      }
    }
    
    return { covered, rare, missing };
  }
}