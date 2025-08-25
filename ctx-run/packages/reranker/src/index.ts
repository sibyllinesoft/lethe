import type { DB } from '@lethe/sqlite';

export interface Candidate {
  docId: string;
  score: number;
  text?: string;
  kind?: string;
}

export interface LLMRerankerConfig {
  enabled: boolean;
  budgetMs: number;
  model: string;
  batchSize: number;
  contradictionPenalty: number;
}

export interface RerankerResult {
  candidates: Candidate[];
  timeoutOccurred: boolean;
  llmCallCount: number;
  contradictionsFound: number;
  fallbackUsed: boolean;
}

// LLM-based reranking scores for candidates
export interface LLMScoreResponse {
  scores: Array<{ id: string; s: number }>;
}

// Contradiction check response
export interface ContradictionResponse {
  contradicts: boolean;
}

export class LLMReranker {
  private config: LLMRerankerConfig;

  constructor(config: Partial<LLMRerankerConfig> = {}) {
    this.config = {
      enabled: config.enabled ?? true,
      budgetMs: config.budgetMs ?? 1200,
      model: config.model ?? 'llama3.2:1b', 
      batchSize: config.batchSize ?? 25,
      contradictionPenalty: config.contradictionPenalty ?? 0.15,
      ...config
    };
  }

  async rerank(
    db: DB,
    query: string,
    candidates: Candidate[],
    crossEncoderFallback: (query: string, candidates: Candidate[]) => Promise<Candidate[]>
  ): Promise<RerankerResult> {
    if (!this.config.enabled || candidates.length === 0) {
      return {
        candidates,
        timeoutOccurred: false,
        llmCallCount: 0,
        contradictionsFound: 0,
        fallbackUsed: false
      };
    }

    const startTime = Date.now();
    let llmCallCount = 0;
    let contradictionsFound = 0;
    let timeoutOccurred = false;
    let fallbackUsed = false;

    try {
      // Step 1: LLM Reranking of top candidates
      const topCandidates = candidates.slice(0, this.config.batchSize);
      console.log(`🤖 LLM reranking ${topCandidates.length} candidates with ${this.config.budgetMs}ms budget`);

      const llmScores = await this.llmRerankWithTimeout(db, query, topCandidates);
      llmCallCount += 1;

      if (Date.now() - startTime > this.config.budgetMs) {
        console.log(`⏰ LLM reranking timeout, falling back to cross-encoder`);
        timeoutOccurred = true;
        fallbackUsed = true;
        const fallbackResult = await crossEncoderFallback(query, candidates);
        return {
          candidates: fallbackResult,
          timeoutOccurred,
          llmCallCount,
          contradictionsFound,
          fallbackUsed
        };
      }

      // Step 2: Apply LLM scores to candidates
      let rerankedCandidates = this.applyLLMScores(topCandidates, llmScores);

      // Step 3: Contradiction-aware penalty
      if (this.config.contradictionPenalty > 0) {
        const contradictionResult = await this.applyContradictionPenalties(
          db, 
          rerankedCandidates, 
          startTime
        );
        rerankedCandidates = contradictionResult.candidates;
        contradictionsFound = contradictionResult.contradictionsFound;
        llmCallCount += contradictionResult.llmCalls;
      }

      // Step 4: Add remaining candidates (not processed by LLM)
      const remainingCandidates = candidates.slice(this.config.batchSize);
      const finalCandidates = [...rerankedCandidates, ...remainingCandidates];

      const totalTime = Date.now() - startTime;
      console.log(`✅ LLM reranking complete in ${totalTime}ms (${llmCallCount} LLM calls, ${contradictionsFound} contradictions)`);

      return {
        candidates: finalCandidates,
        timeoutOccurred,
        llmCallCount,
        contradictionsFound,
        fallbackUsed
      };

    } catch (error) {
      console.error(`❌ LLM reranking failed: ${error}, falling back to cross-encoder`);
      fallbackUsed = true;
      const fallbackResult = await crossEncoderFallback(query, candidates);
      
      return {
        candidates: fallbackResult,
        timeoutOccurred,
        llmCallCount,
        contradictionsFound,
        fallbackUsed
      };
    }
  }

  private async llmRerankWithTimeout(
    db: DB, 
    query: string, 
    candidates: Candidate[]
  ): Promise<LLMScoreResponse> {
    const { getOllamaBridge, safeParseJSON } = await import('@lethe/core');
    
    const ollama = await getOllamaBridge(db);

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
        model: this.config.model,
        prompt,
        temperature: 0,
        max_tokens: 200
      });

      const parsed = safeParseJSON(response.response, {
        scores: candidates.map((_, idx) => ({ id: `C${idx + 1}`, s: 0.5 }))
      }) as LLMScoreResponse;

      return parsed;

    } catch (error) {
      console.warn(`LLM reranking call failed: ${error}`);
      // Return neutral scores as fallback
      return {
        scores: candidates.map((_, idx) => ({ id: `C${idx + 1}`, s: 0.5 }))
      };
    }
  }

  private applyLLMScores(candidates: Candidate[], llmScores: LLMScoreResponse): Candidate[] {
    // Create a map of LLM scores
    const scoreMap = new Map<string, number>();
    for (const score of llmScores.scores) {
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
    db: DB, 
    candidates: Candidate[], 
    startTime: number
  ): Promise<{ candidates: Candidate[], contradictionsFound: number, llmCalls: number }> {
    const selected: Candidate[] = [];
    let contradictionsFound = 0;
    let llmCalls = 0;

    for (const candidate of candidates) {
      // Check budget
      if (Date.now() - startTime > this.config.budgetMs * 0.8) {
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
      const hasContradiction = await this.checkContradiction(
        db, 
        selected, 
        candidate
      );
      llmCalls += 1;

      if (hasContradiction) {
        contradictionsFound += 1;
        // Apply penalty
        const penalizedCandidate = {
          ...candidate,
          score: Math.max(0, candidate.score - this.config.contradictionPenalty)
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

  private async checkContradiction(
    db: DB, 
    selectedCandidates: Candidate[], 
    candidate: Candidate
  ): Promise<boolean> {
    if (!candidate.text) return false;

    const { getOllamaBridge, safeParseJSON } = await import('@lethe/core');
    
    const ollama = await getOllamaBridge(db);

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
        model: this.config.model,
        prompt,
        temperature: 0,
        max_tokens: 50
      });

      const parsed = safeParseJSON(response.response, {
        contradicts: false
      }) as ContradictionResponse;

      return parsed.contradicts;

    } catch (error) {
      console.warn(`Contradiction check failed: ${error}`);
      return false; // Conservative: assume no contradiction on error
    }
  }
}

// Export factory function for integration
export function createLLMReranker(config?: Partial<LLMRerankerConfig>): LLMReranker {
  return new LLMReranker(config);
}