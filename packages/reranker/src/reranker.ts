import { pipeline, env } from '@xenova/transformers';

// Configure transformers.js cache
env.cacheDir = './.transformers-cache';

export interface RankingPair {
  query: string;
  text: string;
  id: string;
}

export interface RankingResult {
  id: string;
  score: number;
}

export class CrossEncoderReranker {
  private model: any;
  private modelName: string;
  private batchSize: number;

  constructor(modelName: string = 'Xenova/bge-reranker-base', batchSize: number = 8) {
    this.modelName = modelName;
    this.batchSize = batchSize;
    this.model = null;
  }

  private async initModel(): Promise<void> {
    if (!this.model) {
      console.log(`Loading reranker model: ${this.modelName}...`);
      this.model = await pipeline('text-classification', this.modelName, {
        quantized: true,
      });
      console.log('Reranker model loaded successfully');
    }
  }

  async rankPairs(pairs: RankingPair[], topkOut: number): Promise<RankingResult[]> {
    await this.initModel();
    
    const results: RankingResult[] = [];
    
    // Process in batches to manage memory
    for (let i = 0; i < pairs.length; i += this.batchSize) {
      const batch = pairs.slice(i, i + this.batchSize);
      const batchResults = await this.processBatch(batch);
      results.push(...batchResults);
    }
    
    // Sort by score descending and take top-k
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topkOut);
  }

  private async processBatch(batch: RankingPair[]): Promise<RankingResult[]> {
    // Prepare inputs for cross-encoder
    // Format: [query, document] pairs
    const inputs = batch.map(pair => `${pair.query} [SEP] ${pair.text}`);
    
    try {
      const outputs = await this.model(inputs);
      
      const results: RankingResult[] = [];
      
      // Handle different output formats
      if (Array.isArray(outputs)) {
        // Multiple inputs - array of classification results
        for (let i = 0; i < batch.length; i++) {
          const output = outputs[i];
          const score = this.extractRelevanceScore(output);
          results.push({
            id: batch[i].id,
            score: score
          });
        }
      } else {
        // Single batch - classification result with multiple items
        const scores = this.extractBatchScores(outputs, batch.length);
        for (let i = 0; i < batch.length; i++) {
          results.push({
            id: batch[i].id,
            score: scores[i]
          });
        }
      }
      
      return results;
    } catch (error) {
      console.warn(`Reranking batch failed: ${error}`);
      // Return original order with neutral scores as fallback
      return batch.map(pair => ({ id: pair.id, score: 0.5 }));
    }
  }

  private extractRelevanceScore(output: any): number {
    // Handle different model output formats
    if (Array.isArray(output)) {
      // Look for positive/relevant class
      const relevantItem = output.find((item: any) => 
        item.label?.toLowerCase().includes('relevant') ||
        item.label?.toLowerCase().includes('positive') ||
        item.label === 'LABEL_1'
      );
      if (relevantItem) {
        return relevantItem.score;
      }
      // Fall back to highest score
      return Math.max(...output.map((item: any) => item.score));
    }
    
    // Direct score
    if (typeof output.score === 'number') {
      return output.score;
    }
    
    // Default neutral score
    return 0.5;
  }

  private extractBatchScores(output: any, batchSize: number): number[] {
    const scores: number[] = [];
    
    if (output.logits && Array.isArray(output.logits)) {
      // Extract logits and apply softmax
      for (let i = 0; i < batchSize; i++) {
        const logits = output.logits[i];
        if (Array.isArray(logits)) {
          // Apply softmax and take relevant class probability
          const softmax = this.softmax(logits);
          // Assume second class is "relevant" (common convention)
          scores.push(softmax[1] || softmax[0]);
        } else {
          scores.push(0.5);
        }
      }
    } else {
      // Fallback: neutral scores
      for (let i = 0; i < batchSize; i++) {
        scores.push(0.5);
      }
    }
    
    return scores;
  }

  private softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(x => Math.exp(x - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map(x => x / sumExp);
  }

  // Convenience method for single query-document pair
  async rankSingle(query: string, texts: string[], topk: number): Promise<RankingResult[]> {
    const pairs: RankingPair[] = texts.map((text, index) => ({
      query,
      text,
      id: index.toString()
    }));
    
    return this.rankPairs(pairs, topk);
  }

  // Utility method to rerank a list of candidates with metadata
  async rerankCandidates<T>(
    query: string,
    candidates: Array<T & { id: string; text: string }>,
    topk: number
  ): Promise<Array<T & { score: number }>> {
    const pairs: RankingPair[] = candidates.map(candidate => ({
      query,
      text: candidate.text,
      id: candidate.id
    }));
    
    const rankings = await this.rankPairs(pairs, topk);
    const rankingMap = new Map(rankings.map(r => [r.id, r.score]));
    
    return candidates
      .map(candidate => ({
        ...candidate,
        score: rankingMap.get(candidate.id) ?? 0
      }))
      .filter(candidate => rankingMap.has(candidate.id))
      .sort((a, b) => b.score - a.score);
  }
}