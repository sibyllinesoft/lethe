import type { Candidate } from '../retrieval/index.js';
import type { DB } from '@lethe/sqlite';

export interface Reranker {
  name: string;
  rerank(query: string, candidates: Candidate[]): Promise<Candidate[]>;
}

export interface RerankerConfig {
  use_llm: boolean;
  llm_budget_ms: number;
  llm_model: string;
  contradiction_enabled: boolean;
  contradiction_penalty: number;
}

export class CrossEncoderReranker implements Reranker {
  name: string;
  private model: any = null;

  constructor(modelId: string = "Xenova/ms-marco-MiniLM-L-6-v2") {
    this.name = modelId;
  }

  async init(): Promise<void> {
    if (!this.model) {
      try {
        console.log(`Loading cross-encoder model: ${this.name}`);
        // Dynamic import to handle ESM
        const { pipeline } = await import('@xenova/transformers');
        this.model = await pipeline('text-classification', this.name, {
          local_files_only: false,
        });
        console.log(`Cross-encoder loaded successfully`);
      } catch (error) {
        console.warn(`Failed to load cross-encoder: ${error}`);
        // Graceful degradation - use simple text similarity
        this.model = null;
      }
    }
  }

  async rerank(query: string, candidates: Candidate[]): Promise<Candidate[]> {
    if (!this.model) {
      await this.init();
    }

    if (!this.model) {
      // Fallback: simple text similarity scoring
      console.log("Using fallback text similarity for reranking");
      return this.fallbackRerank(query, candidates);
    }

    try {
      console.log(`Reranking ${candidates.length} candidates with cross-encoder`);
      
      // Prepare query-document pairs
      const pairs: string[] = [];
      for (const candidate of candidates) {
        if (candidate.text) {
          pairs.push(`${query} [SEP] ${candidate.text}`);
        }
      }

      if (pairs.length === 0) {
        return candidates; // No text to rerank
      }

      // Get relevance scores from cross-encoder
      const outputs = await this.model(pairs);
      
      // Update candidate scores with cross-encoder relevance
      const rerankedCandidates: Candidate[] = [];
      let pairIndex = 0;
      
      for (let i = 0; i < candidates.length; i++) {
        const candidate = candidates[i];
        
        if (candidate.text) {
          const output = outputs[pairIndex];
          // Cross-encoder typically outputs [irrelevant, relevant] scores
          const relevanceScore = Array.isArray(output) ? 
            (output.find(o => o.label === 'LABEL_1' || o.label === 'relevant')?.score || output[1]?.score || 0.5) :
            (output.score || 0.5);
          
          rerankedCandidates.push({
            ...candidate,
            score: relevanceScore // Replace with cross-encoder score
          });
          pairIndex++;
        } else {
          // Keep original score if no text available
          rerankedCandidates.push(candidate);
        }
      }

      // Sort by new relevance scores
      rerankedCandidates.sort((a, b) => b.score - a.score);
      
      console.log(`Reranking complete - score range: ${rerankedCandidates[0]?.score.toFixed(3)} to ${rerankedCandidates[rerankedCandidates.length-1]?.score.toFixed(3)}`);
      
      return rerankedCandidates;
      
    } catch (error) {
      console.error(`Cross-encoder reranking failed: ${error}`);
      return this.fallbackRerank(query, candidates);
    }
  }

  private fallbackRerank(query: string, candidates: Candidate[]): Candidate[] {
    // Simple text similarity fallback
    const queryTerms = this.tokenize(query.toLowerCase());
    
    return candidates.map(candidate => {
      if (!candidate.text) {
        return candidate;
      }
      
      const docTerms = this.tokenize(candidate.text.toLowerCase());
      const overlap = queryTerms.filter(term => docTerms.includes(term));
      const similarityScore = overlap.length / Math.sqrt(queryTerms.length * docTerms.length);
      
      return {
        ...candidate,
        score: similarityScore * 0.5 + candidate.score * 0.5 // Blend with original score
      };
    }).sort((a, b) => b.score - a.score);
  }

  private tokenize(text: string): string[] {
    return text
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(term => term.length > 1);
  }
}

export class LLMReranker implements Reranker {
  name = "llm-reranker";
  private db: DB;
  private config: RerankerConfig;

  constructor(db: DB, config: RerankerConfig) {
    this.db = db;
    this.config = config;
  }

  async rerank(query: string, candidates: Candidate[]): Promise<Candidate[]> {
    if (!this.config.use_llm || candidates.length === 0) {
      // Fall back to cross-encoder
      const crossEncoder = new CrossEncoderReranker();
      return crossEncoder.rerank(query, candidates);
    }

    const startTime = Date.now();
    let llmCallCount = 0;
    let contradictionsFound = 0;
    
    try {
      // Step 1: LLM Reranking of top candidates
      const batchSize = 25;
      const topCandidates = candidates.slice(0, batchSize);
      console.log(`🤖 LLM reranking ${topCandidates.length} candidates with ${this.config.llm_budget_ms}ms budget`);

      const llmScores = await this.llmRerankWithTimeout(query, topCandidates);
      llmCallCount += 1;

      if (Date.now() - startTime > this.config.llm_budget_ms) {
        console.log(`⏰ LLM reranking timeout, falling back to cross-encoder`);
        const crossEncoder = new CrossEncoderReranker();
        return crossEncoder.rerank(query, candidates);
      }

      // Step 2: Apply LLM scores to candidates
      let rerankedCandidates = this.applyLLMScores(topCandidates, llmScores);

      // Step 3: Contradiction-aware penalty (if enabled)
      if (this.config.contradiction_enabled && this.config.contradiction_penalty > 0) {
        const contradictionResult = await this.applyContradictionPenalties(
          rerankedCandidates, 
          startTime
        );
        rerankedCandidates = contradictionResult.candidates;
        contradictionsFound = contradictionResult.contradictionsFound;
        llmCallCount += contradictionResult.llmCalls;
      }

      // Step 4: Add remaining candidates (not processed by LLM)
      const remainingCandidates = candidates.slice(batchSize);
      const finalCandidates = [...rerankedCandidates, ...remainingCandidates];

      const totalTime = Date.now() - startTime;
      console.log(`✅ LLM reranking complete in ${totalTime}ms (${llmCallCount} LLM calls, ${contradictionsFound} contradictions)`);

      return finalCandidates;

    } catch (error) {
      console.error(`❌ LLM reranking failed: ${error}, falling back to cross-encoder`);
      const crossEncoder = new CrossEncoderReranker();
      return crossEncoder.rerank(query, candidates);
    }
  }

  private async llmRerankWithTimeout(query: string, candidates: Candidate[]): Promise<any> {
    const { getOllamaBridge, safeParseJSON } = await import('../ollama/index.js');
    
    const ollama = await getOllamaBridge(this.db);

    // Prepare candidate list for prompt
    const candidateList = candidates
      .map((c, idx) => `C${idx + 1}: ${c.text?.substring(0, 300) || '[no text]'}`)
      .join('\n');

    const prompt = `Given a query and candidate chunks, assign a relevance score in [0,1]. JSON: {"scores": [{"id":"C1","s":0.87}, ...]}.

Query: ${query}

Candidates:
${candidateList}

No text, no explanation. Return JSON only.`;

    try {
      const response = await ollama.generate({
        model: this.config.llm_model,
        prompt,
        temperature: 0,
        max_tokens: 200
      });

      const parsed = safeParseJSON(response.response, {
        scores: candidates.map((_, idx) => ({ id: `C${idx + 1}`, s: 0.5 }))
      });

      return parsed;

    } catch (error) {
      console.warn(`LLM reranking call failed: ${error}`);
      // Return neutral scores as fallback
      return {
        scores: candidates.map((_, idx) => ({ id: `C${idx + 1}`, s: 0.5 }))
      };
    }
  }

  private applyLLMScores(candidates: Candidate[], llmScores: any): Candidate[] {
    // Create a map of LLM scores
    const scoreMap = new Map<string, number>();
    for (const score of llmScores.scores || []) {
      scoreMap.set(score.id, Math.max(0, Math.min(1, score.s))); // Clamp to [0,1]
    }

    // Apply LLM scores to candidates
    const rerankedCandidates = candidates.map((candidate, idx) => {
      const llmScore = scoreMap.get(`C${idx + 1}`) ?? 0.5;
      return {
        ...candidate,
        score: llmScore
      };
    });

    // Sort by LLM scores descending
    rerankedCandidates.sort((a, b) => b.score - a.score);
    return rerankedCandidates;
  }

  private async applyContradictionPenalties(
    candidates: Candidate[], 
    startTime: number
  ): Promise<{ candidates: Candidate[], contradictionsFound: number, llmCalls: number }> {
    const selected: Candidate[] = [];
    let contradictionsFound = 0;
    let llmCalls = 0;

    for (const candidate of candidates) {
      // Check budget
      if (Date.now() - startTime > this.config.llm_budget_ms * 0.8) {
        console.log(`⏰ Contradiction checking budget exhausted, adding remaining candidates`);
        selected.push(...candidates.slice(selected.length));
        break;
      }

      if (selected.length === 0) {
        // First candidate is always selected
        selected.push(candidate);
        continue;
      }

      // Check if candidate contradicts any selected candidates
      const hasContradiction = await this.checkContradiction(selected, candidate);
      llmCalls += 1;

      if (hasContradiction) {
        contradictionsFound += 1;
        // Apply penalty
        const penalizedCandidate = {
          ...candidate,
          score: Math.max(0, candidate.score - this.config.contradiction_penalty)
        };
        selected.push(penalizedCandidate);
        console.log(`⚠️ Contradiction detected, penalized candidate ${candidate.docId} (score: ${candidate.score.toFixed(3)} → ${penalizedCandidate.score.toFixed(3)})`);
      } else {
        selected.push(candidate);
      }
    }

    // Re-sort after penalty application
    selected.sort((a, b) => b.score - a.score);

    return { 
      candidates: selected, 
      contradictionsFound, 
      llmCalls 
    };
  }

  private async checkContradiction(selectedCandidates: Candidate[], candidate: Candidate): Promise<boolean> {
    if (!candidate.text) return false;

    const { getOllamaBridge, safeParseJSON } = await import('../ollama/index.js');
    
    const ollama = await getOllamaBridge(this.db);

    // Create summary of selected candidates
    const selectedTexts = selectedCandidates
      .filter(c => c.text)
      .map(c => c.text!.substring(0, 200))
      .join(' ');

    const prompt = `Given a selected set S and a candidate c, return JSON {"contradicts": true|false} by checking direct factual conflict; be conservative.

Selected set S: ${selectedTexts}

Candidate c: ${candidate.text.substring(0, 200)}

Return JSON only.`;

    try {
      const response = await ollama.generate({
        model: this.config.llm_model,
        prompt,
        temperature: 0,
        max_tokens: 50
      });

      const parsed = safeParseJSON(response.response, {
        contradicts: false
      });

      return parsed.contradicts;

    } catch (error) {
      console.warn(`Contradiction check failed: ${error}`);
      return false; // Conservative: assume no contradiction on error
    }
  }
}

export class NoOpReranker implements Reranker {
  name = "noop";

  async rerank(query: string, candidates: Candidate[]): Promise<Candidate[]> {
    return candidates; // Pass through unchanged
  }
}

export async function getReranker(
  enabled: boolean = true, 
  config?: RerankerConfig, 
  db?: DB
): Promise<Reranker> {
  if (!enabled) {
    return new NoOpReranker();
  }

  // Use LLM reranker if configured and DB is available
  if (config?.use_llm && db) {
    try {
      const llmReranker = new LLMReranker(db, config);
      return llmReranker;
    } catch (error) {
      console.warn(`LLM reranker initialization failed: ${error}, falling back to cross-encoder`);
    }
  }

  // Fallback to cross-encoder reranker
  try {
    const reranker = new CrossEncoderReranker();
    await reranker.init();
    return reranker;
  } catch (error) {
    console.warn(`Cross-encoder reranker initialization failed: ${error}, using no-op`);
    return new NoOpReranker();
  }
}

// Default configuration for LLM reranker
export const DEFAULT_LLM_RERANKER_CONFIG: RerankerConfig = {
  use_llm: true,
  llm_budget_ms: 1200,
  llm_model: 'llama3.2:1b',
  contradiction_enabled: true,
  contradiction_penalty: 0.15
};