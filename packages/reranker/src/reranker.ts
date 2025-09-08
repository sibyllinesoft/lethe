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

export interface EnhancedRankingPair extends RankingPair {
  metadata?: Record<string, unknown>;
  timestamp?: string;
  source?: string;
}

export interface EnhancedRankingResult extends RankingResult {
  llmScore?: number;
  contradictions?: ContradictionResult[];
  metadata?: Record<string, unknown>;
  processingTime?: number;
  modelConfidence?: number;
}

export interface ContradictionResult {
  contradictedBy: string[];
  confidenceScore: number;
  contradictionType: 'semantic' | 'factual' | 'temporal';
  description: string;
  resolved?: boolean;
}

export interface LLMConfig {
  model: string;
  apiKey?: string;
  baseUrl?: string;
  maxTokens?: number;
  temperature?: number;
  enableContradictionDetection?: boolean;
  contradictionThreshold?: number;
  batchSize?: number;
}

export interface SchemaValidationResult {
  valid: boolean;
  errors: string[];
  schema: string;
}

export type Result<T, E = Error> = 
  | { success: true; data: T }
  | { success: false; error: E };

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

/**
 * Enhanced LLM-based reranker with contradiction detection and schema validation.
 * Supports multiple LLM providers and advanced contradiction-aware ranking.
 */
export class LLMEnhancedReranker {
  private crossEncoder: CrossEncoderReranker;
  private llmConfig: LLMConfig;
  private contradictionCache = new Map<string, ContradictionResult[]>();
  private schemaCache = new Map<string, SchemaValidationResult>();
  
  constructor(
    crossEncoderModel: string = 'Xenova/bge-reranker-base',
    llmConfig: LLMConfig,
    batchSize: number = 8
  ) {
    this.crossEncoder = new CrossEncoderReranker(crossEncoderModel, batchSize);
    this.llmConfig = {
      maxTokens: 4096,
      temperature: 0.1,
      contradictionThreshold: 0.7,
      batchSize: 4,
      ...llmConfig
    };
  }

  /**
   * Enhanced ranking with LLM integration and contradiction detection
   */
  async enhancedRank(
    pairs: EnhancedRankingPair[],
    topk: number,
    options: {
      enableLLMScoring?: boolean;
      enableContradictionDetection?: boolean;
      schemaValidation?: { schema: string; strict?: boolean };
      fusionWeight?: number; // Weight between cross-encoder and LLM scores
    } = {}
  ): Promise<Result<EnhancedRankingResult[], Error>> {
    const startTime = performance.now();
    
    try {
      // Step 1: Cross-encoder baseline ranking
      const crossEncoderResults = await this.crossEncoder.rankPairs(pairs, Math.min(pairs.length, topk * 2));
      
      // Step 2: LLM-based scoring if enabled
      let llmScores: Map<string, number> = new Map();
      if (options.enableLLMScoring) {
        const llmScoringResult = await this.getLLMScores(pairs, crossEncoderResults);
        if (!llmScoringResult.success) {
          return llmScoringResult;
        }
        llmScores = llmScoringResult.data;
      }
      
      // Step 3: Schema validation if requested
      let schemaResults: Map<string, SchemaValidationResult> = new Map();
      if (options.schemaValidation) {
        const validationResult = await this.validateAgainstSchema(pairs, options.schemaValidation.schema);
        if (validationResult.success) {
          schemaResults = validationResult.data;
        }
      }
      
      // Step 4: Contradiction detection if enabled
      let contradictions: Map<string, ContradictionResult[]> = new Map();
      if (options.enableContradictionDetection || this.llmConfig.enableContradictionDetection) {
        const contradictionResult = await this.detectContradictions(pairs);
        if (contradictionResult.success) {
          contradictions = contradictionResult.data;
        }
      }
      
      // Step 5: Fusion scoring and final ranking
      const fusionWeight = options.fusionWeight ?? 0.3; // Default: 70% cross-encoder, 30% LLM
      const enhancedResults: EnhancedRankingResult[] = [];
      
      for (const result of crossEncoderResults) {
        const pair = pairs.find(p => p.id === result.id);
        if (!pair) continue;
        
        const llmScore = llmScores.get(result.id);
        const pairContradictions = contradictions.get(result.id) || [];
        const schemaResult = schemaResults.get(result.id);
        
        // Calculate fusion score
        let finalScore = result.score;
        if (llmScore !== undefined) {
          finalScore = (1 - fusionWeight) * result.score + fusionWeight * llmScore;
        }
        
        // Apply contradiction penalty
        if (pairContradictions.length > 0) {
          const contradictionPenalty = pairContradictions.reduce((penalty, contradiction) => {
            return penalty + (contradiction.confidenceScore * 0.2); // Max 20% penalty per contradiction
          }, 0);
          finalScore = Math.max(0.1, finalScore - contradictionPenalty);
        }
        
        // Apply schema validation penalty
        if (schemaResult && !schemaResult.valid && options.schemaValidation?.strict) {
          finalScore *= 0.5; // 50% penalty for schema violations in strict mode
        }
        
        enhancedResults.push({
          id: result.id,
          score: finalScore,
          llmScore: llmScore,
          contradictions: pairContradictions,
          metadata: {
            ...pair.metadata,
            crossEncoderScore: result.score,
            schemaValid: schemaResult?.valid,
            schemaErrors: schemaResult?.errors,
            processingTime: performance.now() - startTime
          },
          processingTime: performance.now() - startTime,
          modelConfidence: this.calculateModelConfidence(result.score, llmScore, pairContradictions)
        });
      }
      
      // Sort by final score and return top-k
      enhancedResults.sort((a, b) => b.score - a.score);
      const finalResults = enhancedResults.slice(0, topk);
      
      return { success: true, data: finalResults };
      
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error : new Error('Enhanced ranking failed')
      };
    }
  }

  /**
   * Get LLM-based relevance scores for ranking pairs
   */
  private async getLLMScores(
    pairs: EnhancedRankingPair[],
    crossEncoderResults: RankingResult[]
  ): Promise<Result<Map<string, number>, Error>> {
    try {
      const scores = new Map<string, number>();
      const batchSize = this.llmConfig.batchSize || 4;
      
      // Process in batches to manage API rate limits
      for (let i = 0; i < pairs.length; i += batchSize) {
        const batch = pairs.slice(i, i + batchSize);
        const batchScores = await this.processLLMBatch(batch);
        
        if (!batchScores.success) {
          return batchScores;
        }
        
        for (const [id, score] of batchScores.data.entries()) {
          scores.set(id, score);
        }
      }
      
      return { success: true, data: scores };
      
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error : new Error('LLM scoring failed')
      };
    }
  }

  /**
   * Process a batch of pairs through LLM for relevance scoring
   */
  private async processLLMBatch(
    batch: EnhancedRankingPair[]
  ): Promise<Result<Map<string, number>, Error>> {
    const scores = new Map<string, number>();
    
    // Create prompt for relevance scoring
    const prompt = this.createRelevancePrompt(batch);
    
    try {
      const response = await this.callLLM(prompt, {
        maxTokens: this.llmConfig.maxTokens,
        temperature: this.llmConfig.temperature
      });
      
      if (!response.success) {
        return response;
      }
      
      // Parse LLM response for scores
      const parsedScores = this.parseRelevanceScores(response.data.content, batch);
      
      if (!parsedScores.success) {
        return parsedScores;
      }
      
      return { success: true, data: parsedScores.data };
      
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error : new Error('LLM batch processing failed')
      };
    }
  }

  /**
   * Create prompt for LLM relevance scoring
   */
  private createRelevancePrompt(batch: EnhancedRankingPair[]): string {
    let prompt = `You are an expert relevance scorer. For each query-document pair below, provide a relevance score from 0.0 to 1.0, where 1.0 means highly relevant and 0.0 means not relevant at all.

Consider the following criteria:
- Semantic relevance to the query
- Factual accuracy and completeness  
- Clarity and comprehensiveness of the answer
- Temporal relevance (if time-sensitive)

Format your response as JSON with the structure: {"scores": [{"id": "pair_id", "score": 0.85, "reasoning": "brief explanation"}, ...]}

Query-Document Pairs:

`;

    batch.forEach((pair, index) => {
      prompt += `Pair ${index + 1} (ID: ${pair.id}):
Query: "${pair.query}"
Document: "${pair.text}"
${pair.source ? `Source: ${pair.source}` : ''}
${pair.timestamp ? `Timestamp: ${pair.timestamp}` : ''}

`;
    });

    prompt += `\nProvide your analysis in the JSON format specified above.`;
    
    return prompt;
  }

  /**
   * Parse LLM response to extract relevance scores
   */
  private parseRelevanceScores(
    content: string,
    batch: EnhancedRankingPair[]
  ): Result<Map<string, number>, Error> {
    try {
      // Try to extract JSON from the response
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        return {
          success: false,
          error: new Error('No JSON found in LLM response')
        };
      }
      
      const parsed = JSON.parse(jsonMatch[0]);
      const scores = new Map<string, number>();
      
      if (parsed.scores && Array.isArray(parsed.scores)) {
        for (const scoreData of parsed.scores) {
          if (scoreData.id && typeof scoreData.score === 'number') {
            // Ensure score is within valid range
            const normalizedScore = Math.max(0, Math.min(1, scoreData.score));
            scores.set(scoreData.id, normalizedScore);
          }
        }
      }
      
      // Fallback: assign default scores for missing IDs
      for (const pair of batch) {
        if (!scores.has(pair.id)) {
          scores.set(pair.id, 0.5); // Neutral score as fallback
        }
      }
      
      return { success: true, data: scores };
      
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error : new Error('Failed to parse LLM scores')
      };
    }
  }

  /**
   * Detect contradictions between ranking pairs
   */
  private async detectContradictions(
    pairs: EnhancedRankingPair[]
  ): Promise<Result<Map<string, ContradictionResult[]>, Error>> {
    const contradictionsMap = new Map<string, ContradictionResult[]>();
    
    try {
      // Generate cache key for this set of pairs
      const cacheKey = this.generateContradictionCacheKey(pairs);
      
      // Check cache first
      if (this.contradictionCache.has(cacheKey)) {
        const cached = this.contradictionCache.get(cacheKey)!;
        
        // Rebuild map from cached results
        for (let i = 0; i < pairs.length; i++) {
          const pairContradictions = cached.filter(c => c.contradictedBy.includes(pairs[i].id));
          if (pairContradictions.length > 0) {
            contradictionsMap.set(pairs[i].id, pairContradictions);
          }
        }
        
        return { success: true, data: contradictionsMap };
      }
      
      // Perform pairwise contradiction analysis
      const allContradictions: ContradictionResult[] = [];
      
      for (let i = 0; i < pairs.length; i++) {
        for (let j = i + 1; j < pairs.length; j++) {
          const pair1 = pairs[i];
          const pair2 = pairs[j];
          
          const contradictionResult = await this.analyzeContradiction(pair1, pair2);
          
          if (contradictionResult.success && contradictionResult.data) {
            const contradiction = contradictionResult.data;
            allContradictions.push(contradiction);
            
            // Add to both pairs' contradiction lists
            if (!contradictionsMap.has(pair1.id)) {
              contradictionsMap.set(pair1.id, []);
            }
            if (!contradictionsMap.has(pair2.id)) {
              contradictionsMap.set(pair2.id, []);
            }
            
            contradictionsMap.get(pair1.id)!.push(contradiction);
            contradictionsMap.get(pair2.id)!.push(contradiction);
          }
        }
      }
      
      // Cache the results
      this.contradictionCache.set(cacheKey, allContradictions);
      
      return { success: true, data: contradictionsMap };
      
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error : new Error('Contradiction detection failed')
      };
    }
  }

  /**
   * Analyze potential contradiction between two pairs
   */
  private async analyzeContradiction(
    pair1: EnhancedRankingPair,
    pair2: EnhancedRankingPair
  ): Promise<Result<ContradictionResult | null, Error>> {
    // Quick heuristic check first
    const heuristicResult = this.performHeuristicContradictionCheck(pair1, pair2);
    
    if (heuristicResult.confidenceScore < (this.llmConfig.contradictionThreshold || 0.7)) {
      return { success: true, data: null };
    }
    
    // LLM-based detailed analysis for high-confidence cases
    if (this.llmConfig.enableContradictionDetection) {
      return await this.performLLMContradictionAnalysis(pair1, pair2, heuristicResult);
    }
    
    return { success: true, data: heuristicResult };
  }

  /**
   * Perform heuristic contradiction checking
   */
  private performHeuristicContradictionCheck(
    pair1: EnhancedRankingPair,
    pair2: EnhancedRankingPair
  ): ContradictionResult {
    const text1 = pair1.text.toLowerCase();
    const text2 = pair2.text.toLowerCase();
    
    // Define contradiction patterns
    const contradictoryPairs = [
      { positive: 'yes', negative: 'no' },
      { positive: 'true', negative: 'false' },
      { positive: 'enable', negative: 'disable' },
      { positive: 'allow', negative: 'deny' },
      { positive: 'always', negative: 'never' },
      { positive: 'should', negative: 'should not' },
      { positive: 'can', negative: 'cannot' },
      { positive: 'will', negative: 'will not' },
      { positive: 'is', negative: 'is not' },
      { positive: 'has', negative: 'has not' }
    ];
    
    let maxConfidence = 0;
    let bestMatch = { positive: '', negative: '' };
    
    // Check for contradictory patterns
    for (const pattern of contradictoryPairs) {
      const hasPositive1 = text1.includes(pattern.positive);
      const hasNegative1 = text1.includes(pattern.negative);
      const hasPositive2 = text2.includes(pattern.positive);
      const hasNegative2 = text2.includes(pattern.negative);
      
      // Check for direct contradiction
      if ((hasPositive1 && hasNegative2) || (hasNegative1 && hasPositive2)) {
        // Calculate confidence based on context overlap
        const contextOverlap = this.calculateContextOverlap(pair1, pair2);
        const confidence = Math.min(0.9, contextOverlap * 1.2);
        
        if (confidence > maxConfidence) {
          maxConfidence = confidence;
          bestMatch = pattern;
        }
      }
    }
    
    return {
      contradictedBy: [pair1.id, pair2.id],
      confidenceScore: maxConfidence,
      contradictionType: maxConfidence > 0 ? 'semantic' : 'semantic',
      description: maxConfidence > 0 ? 
        `Contradiction detected between "${bestMatch.positive}" and "${bestMatch.negative}"` :
        'No significant contradiction detected',
      resolved: false
    };
  }

  /**
   * Calculate context overlap between two pairs
   */
  private calculateContextOverlap(pair1: EnhancedRankingPair, pair2: EnhancedRankingPair): number {
    const words1 = new Set(pair1.text.toLowerCase().split(/\s+/));
    const words2 = new Set(pair2.text.toLowerCase().split(/\s+/));
    
    const intersection = new Set([...words1].filter(word => words2.has(word)));
    const union = new Set([...words1, ...words2]);
    
    return intersection.size / union.size;
  }

  /**
   * Perform LLM-based contradiction analysis
   */
  private async performLLMContradictionAnalysis(
    pair1: EnhancedRankingPair,
    pair2: EnhancedRankingPair,
    heuristicResult: ContradictionResult
  ): Promise<Result<ContradictionResult, Error>> {
    const prompt = `You are an expert at detecting contradictions between pieces of text. Analyze the following two documents and determine if they contradict each other regarding the query.

Query: "${pair1.query}"

Document A (ID: ${pair1.id}):
"${pair1.text}"

Document B (ID: ${pair2.id}):
"${pair2.text}"

Analyze whether these documents contradict each other. Consider:
1. Direct factual contradictions
2. Semantic contradictions (different conclusions about the same topic)
3. Temporal contradictions (conflicting information about timing)

Provide your analysis in JSON format:
{
  "contradiction_detected": true/false,
  "confidence_score": 0.0-1.0,
  "contradiction_type": "semantic|factual|temporal|none",
  "description": "detailed explanation of the contradiction or why no contradiction exists",
  "key_differences": ["difference 1", "difference 2"]
}`;

    try {
      const response = await this.callLLM(prompt, {
        maxTokens: 1000,
        temperature: 0.1
      });
      
      if (!response.success) {
        return { success: true, data: heuristicResult }; // Fallback to heuristic
      }
      
      const parsed = this.parseLLMContradictionResponse(response.data.content, pair1, pair2);
      
      if (parsed.success) {
        return { success: true, data: parsed.data };
      }
      
      return { success: true, data: heuristicResult }; // Fallback
      
    } catch (error) {
      return { success: true, data: heuristicResult }; // Fallback on error
    }
  }

  /**
   * Parse LLM contradiction analysis response
   */
  private parseLLMContradictionResponse(
    content: string,
    pair1: EnhancedRankingPair,
    pair2: EnhancedRankingPair
  ): Result<ContradictionResult, Error> {
    try {
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        return {
          success: false,
          error: new Error('No JSON found in LLM contradiction response')
        };
      }
      
      const parsed = JSON.parse(jsonMatch[0]);
      
      if (!parsed.contradiction_detected) {
        return {
          success: true,
          data: {
            contradictedBy: [],
            confidenceScore: 0,
            contradictionType: 'semantic',
            description: parsed.description || 'No contradiction detected by LLM',
            resolved: false
          }
        };
      }
      
      return {
        success: true,
        data: {
          contradictedBy: [pair1.id, pair2.id],
          confidenceScore: Math.max(0, Math.min(1, parsed.confidence_score || 0.5)),
          contradictionType: parsed.contradiction_type || 'semantic',
          description: parsed.description || 'Contradiction detected by LLM',
          resolved: false
        }
      };
      
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error : new Error('Failed to parse LLM contradiction response')
      };
    }
  }

  /**
   * Validate ranking pairs against a JSON schema
   */
  private async validateAgainstSchema(
    pairs: EnhancedRankingPair[],
    schema: string
  ): Promise<Result<Map<string, SchemaValidationResult>, Error>> {
    const results = new Map<string, SchemaValidationResult>();
    
    // Check cache first
    const cacheKey = `${schema}_${pairs.length}`;
    if (this.schemaCache.has(cacheKey)) {
      const cachedResult = this.schemaCache.get(cacheKey)!;
      pairs.forEach(pair => {
        results.set(pair.id, cachedResult);
      });
      return { success: true, data: results };
    }
    
    try {
      // Simple schema validation (in a real implementation, use a proper JSON schema validator)
      const schemaObj = JSON.parse(schema);
      
      for (const pair of pairs) {
        try {
          // Try to parse the text as JSON for validation
          const data = JSON.parse(pair.text);
          const validationResult = this.validateObject(data, schemaObj);
          
          results.set(pair.id, {
            valid: validationResult.valid,
            errors: validationResult.errors,
            schema: schema
          });
          
        } catch (parseError) {
          // If text is not JSON, mark as invalid
          results.set(pair.id, {
            valid: false,
            errors: ['Content is not valid JSON'],
            schema: schema
          });
        }
      }
      
      return { success: true, data: results };
      
    } catch (schemaError) {
      return {
        success: false,
        error: schemaError instanceof Error ? schemaError : new Error('Schema validation failed')
      };
    }
  }

  /**
   * Simple object validation against schema (placeholder implementation)
   */
  private validateObject(obj: any, schema: any): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    
    // Basic type checking
    if (schema.type && typeof obj !== schema.type) {
      errors.push(`Expected type ${schema.type}, got ${typeof obj}`);
    }
    
    // Check required properties
    if (schema.required && Array.isArray(schema.required)) {
      for (const prop of schema.required) {
        if (!(prop in obj)) {
          errors.push(`Missing required property: ${prop}`);
        }
      }
    }
    
    // Check properties if object
    if (schema.properties && typeof obj === 'object') {
      for (const [key, propSchema] of Object.entries(schema.properties)) {
        if (key in obj) {
          const propValidation = this.validateObject(obj[key], propSchema);
          errors.push(...propValidation.errors.map(err => `${key}.${err}`));
        }
      }
    }
    
    return { valid: errors.length === 0, errors };
  }

  /**
   * Calculate model confidence based on various factors
   */
  private calculateModelConfidence(
    crossEncoderScore: number,
    llmScore?: number,
    contradictions?: ContradictionResult[]
  ): number {
    let confidence = crossEncoderScore;
    
    // Factor in LLM agreement
    if (llmScore !== undefined) {
      const agreement = 1 - Math.abs(crossEncoderScore - llmScore);
      confidence = confidence * 0.7 + agreement * 0.3;
    }
    
    // Reduce confidence for contradictions
    if (contradictions && contradictions.length > 0) {
      const contradictionPenalty = contradictions.reduce((penalty, c) => {
        return penalty + (c.confidenceScore * 0.1);
      }, 0);
      confidence = Math.max(0.1, confidence - contradictionPenalty);
    }
    
    return Math.max(0, Math.min(1, confidence));
  }

  /**
   * Generate cache key for contradiction detection
   */
  private generateContradictionCacheKey(pairs: EnhancedRankingPair[]): string {
    const sortedIds = pairs.map(p => p.id).sort();
    return `contradiction_${sortedIds.join('_')}`;
  }

  /**
   * Call LLM API (placeholder implementation - replace with actual LLM provider)
   */
  private async callLLM(
    prompt: string,
    options: { maxTokens?: number; temperature?: number }
  ): Promise<Result<{ content: string; usage?: any }, Error>> {
    // This is a placeholder implementation
    // In a real scenario, replace with actual LLM API calls (OpenAI, Anthropic, etc.)
    
    try {
      // Simulate LLM call delay
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Mock response based on prompt analysis
      if (prompt.includes('relevance scoring') || prompt.includes('relevance scorer')) {
        const mockScores = this.generateMockRelevanceScores(prompt);
        return {
          success: true,
          data: { content: mockScores }
        };
      } else if (prompt.includes('contradiction')) {
        const mockContradiction = this.generateMockContradictionAnalysis(prompt);
        return {
          success: true,
          data: { content: mockContradiction }
        };
      }
      
      return {
        success: true,
        data: { content: 'Mock LLM response' }
      };
      
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error : new Error('LLM API call failed')
      };
    }
  }

  /**
   * Generate mock relevance scores for testing
   */
  private generateMockRelevanceScores(prompt: string): string {
    // Extract pair IDs from prompt
    const pairIds = prompt.match(/ID: ([^)]+)/g)?.map(match => match.replace('ID: ', '')) || [];
    
    const scores = pairIds.map(id => ({
      id: id,
      score: Math.random() * 0.6 + 0.4, // Random score between 0.4-1.0
      reasoning: "Mock reasoning for relevance assessment"
    }));
    
    return JSON.stringify({ scores });
  }

  /**
   * Generate mock contradiction analysis for testing
   */
  private generateMockContradictionAnalysis(prompt: string): string {
    // Simple heuristic for mock contradiction detection
    const hasContradictoryTerms = prompt.toLowerCase().includes('no') && prompt.toLowerCase().includes('yes') ||
                                  prompt.toLowerCase().includes('true') && prompt.toLowerCase().includes('false');
    
    return JSON.stringify({
      contradiction_detected: hasContradictoryTerms,
      confidence_score: hasContradictoryTerms ? 0.8 : 0.2,
      contradiction_type: hasContradictoryTerms ? "semantic" : "none",
      description: hasContradictoryTerms ? 
        "Mock contradiction detected based on opposing terms" : 
        "No clear contradiction found in mock analysis",
      key_differences: hasContradictoryTerms ? ["opposing statements"] : []
    });
  }
}

/**
 * Factory function to create LLM enhanced reranker
 */
export function createLLMEnhancedReranker(
  crossEncoderModel?: string,
  llmConfig?: Partial<LLMConfig>,
  batchSize?: number
): LLMEnhancedReranker {
  const defaultLLMConfig: LLMConfig = {
    model: 'gpt-4',
    maxTokens: 4096,
    temperature: 0.1,
    enableContradictionDetection: true,
    contradictionThreshold: 0.7,
    batchSize: 4,
    ...llmConfig
  };
  
  return new LLMEnhancedReranker(crossEncoderModel, defaultLLMConfig, batchSize);
}