import axios, { AxiosInstance } from 'axios';
import type { CtxDatabase, SessionState } from '@ctx-run/sqlite';
import type { HydeResult, ContextPack, PlanType, Candidate } from './types.js';
import { DfIdfBuilder } from './dfidf.js';
import { v4 as uuidv4 } from 'uuid';

export interface AIConfig {
  ollamaUrl: string;
  hydeModel: string;
  summarizeModel: string;
  timeout: number;
}

export class AIIntegration {
  private db: CtxDatabase;
  private client: AxiosInstance;
  private dfIdfBuilder: DfIdfBuilder;
  private config: AIConfig;

  constructor(db: CtxDatabase, config: AIConfig) {
    this.db = db;
    this.dfIdfBuilder = new DfIdfBuilder(db);
    this.config = config;
    
    this.client = axios.create({
      baseURL: config.ollamaUrl,
      timeout: config.timeout,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  async isAvailable(): Promise<boolean> {
    try {
      const response = await this.client.get('/api/version');
      return response.status === 200;
    } catch {
      return false;
    }
  }

  // Generate HyDE queries
  async generateHyde(
    userQuery: string,
    sessionId: string,
    k: number,
    planType: PlanType
  ): Promise<HydeResult> {
    const isAvailable = await this.isAvailable();
    if (!isAvailable) {
      // Fallback: return the original query
      return {
        queries: [userQuery],
        pseudo: userQuery
      };
    }

    try {
      // Get context for HyDE generation
      const rareTerms = await this.dfIdfBuilder.getTopRareTerms(sessionId, 10);
      const commonTerms = await this.dfIdfBuilder.getTopCommonTerms(sessionId, 10);
      const midTerms = await this.dfIdfBuilder.getMidFrequencyTerms(sessionId, 1.0, 4.0);
      
      const prompt = this.buildHydePrompt(
        userQuery,
        rareTerms.slice(0, 5),
        midTerms.slice(0, 10),
        commonTerms.slice(0, 5),
        k
      );

      const response = await this.client.post('/api/generate', {
        model: this.config.hydeModel,
        prompt,
        format: 'json',
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9
        }
      });

      const result = this.parseHydeResponse(response.data.response);
      return result;
    } catch (error) {
      console.warn(`HyDE generation failed: ${error}`);
      return {
        queries: [userQuery],
        pseudo: userQuery
      };
    }
  }

  private buildHydePrompt(
    userQuery: string,
    anchorTerms: string[],
    rareTerms: string[],
    excludeTerms: string[],
    k: number
  ): string {
    return `You generate discriminative search queries and one short pseudo-document for retrieving past conversation chunks.

User need: ${userQuery}
Anchors (include): ${anchorTerms.join(', ')}
Prefer rare terms: ${rareTerms.join(', ')}
Avoid generic terms: ${excludeTerms.join(', ')}

Generate ${k} diverse search queries that would find relevant conversation history. Each query should use specific technical terms and avoid generic language.

Also create a short pseudo-document (3-5 sentences) that represents the ideal response using specific terms.

Return JSON:
{"queries":["...","...","..."], "pseudo_doc":"3-5 sentences using specific terms"}`;
  }

  private parseHydeResponse(response: string): HydeResult {
    try {
      const parsed = JSON.parse(response);
      return {
        queries: parsed.queries || [response],
        pseudo: parsed.pseudo_doc || response
      };
    } catch {
      // Fallback parsing
      const lines = response.split('\n').filter(line => line.trim());
      return {
        queries: [lines[0] || response],
        pseudo: response
      };
    }
  }

  // Generate context pack summary
  async buildPack(
    query: string,
    selectedChunks: Candidate[],
    sessionId: string,
    granularity: string = 'medium'
  ): Promise<ContextPack> {
    const isAvailable = await this.isAvailable();
    if (!isAvailable) {
      // Fallback: basic pack without AI summarization
      return this.buildFallbackPack(query, selectedChunks, sessionId);
    }

    try {
      const chunks = this.db.getChunksByIds(selectedChunks.map(c => c.id));
      const chunkContext = chunks.map((chunk, i) => ({
        id: chunk.id,
        text: chunk.text,
        messageId: chunk.messageId,
        offset: [chunk.offsetStart, chunk.offsetEnd] as [number, number]
      }));

      const prompt = this.buildSummarizePrompt(query, chunkContext, granularity);

      const response = await this.client.post('/api/generate', {
        model: this.config.summarizeModel,
        prompt,
        format: 'json',
        stream: false,
        options: {
          temperature: 0.3,
          top_p: 0.8
        }
      });

      const packData = this.parseSummaryResponse(response.data.response);
      
      return {
        id: uuidv4(),
        sessionId,
        query,
        ...packData
      };
    } catch (error) {
      console.warn(`Pack generation failed: ${error}`);
      return this.buildFallbackPack(query, selectedChunks, sessionId);
    }
  }

  private buildSummarizePrompt(
    query: string,
    chunks: Array<{ id: string; text: string; messageId: string; offset: [number, number] }>,
    granularity: string
  ): string {
    const chunkText = chunks.map((chunk, i) => 
      `[${chunk.id}] ${chunk.text}\nCITE: message=${chunk.messageId} span=${chunk.offset[0]}-${chunk.offset[1]}`
    ).join('\n\n');

    return `Build a context pack answering the user query from the provided conversation chunks.

Query: ${query}
Granularity: ${granularity}

Chunks:
${chunkText}

Output JSON:
{
 "summary":"Comprehensive summary addressing the query",
 "key_entities":["entity1", "entity2", "entity3"],
 "claims":[{"text":"Specific claim with evidence","chunks":["C12","C18"]}],
 "contradictions":[{"issue":"Any conflicting information","chunks":["C7","C9"]}],
 "citations":{"C12":{"messageId":"M5","span":[10,180]}}
}

Prioritize precision and cite chunk IDs for each claim. Include key technical terms and specific details.`;
  }

  private parseSummaryResponse(response: string): Omit<ContextPack, 'id' | 'sessionId' | 'query'> {
    try {
      const parsed = JSON.parse(response);
      return {
        summary: parsed.summary || 'Summary not available',
        keyEntities: parsed.key_entities || [],
        claims: parsed.claims || [],
        contradictions: parsed.contradictions || [],
        citations: parsed.citations || {}
      };
    } catch {
      // Fallback parsing
      return {
        summary: response.length > 100 ? response.substring(0, 500) + '...' : response,
        keyEntities: [],
        claims: [],
        contradictions: [],
        citations: {}
      };
    }
  }

  private buildFallbackPack(
    query: string,
    selectedChunks: Candidate[],
    sessionId: string
  ): ContextPack {
    const chunks = this.db.getChunksByIds(selectedChunks.map(c => c.id));
    
    // Simple extractive summary
    const summaryChunks = chunks.slice(0, 3);
    const summary = summaryChunks.map(c => c.text).join(' ... ');
    
    // Basic entity extraction
    const allText = chunks.map(c => c.text).join(' ');
    const entities = this.extractBasicEntities(allText);
    
    // Build citations map
    const citations: Record<string, { messageId: string; span: [number, number] }> = {};
    for (const chunk of chunks) {
      citations[chunk.id] = {
        messageId: chunk.messageId,
        span: [chunk.offsetStart, chunk.offsetEnd]
      };
    }

    return {
      id: uuidv4(),
      sessionId,
      query,
      summary: summary.length > 1000 ? summary.substring(0, 1000) + '...' : summary,
      keyEntities: entities.slice(0, 10),
      claims: [{
        text: 'Information extracted from conversation history',
        chunks: chunks.map(c => c.id).slice(0, 5)
      }],
      contradictions: [],
      citations
    };
  }

  private extractBasicEntities(text: string): string[] {
    const entities = new Set<string>();
    
    // Capitalized words
    const capitalizedWords = text.match(/\b[A-Z][a-zA-Z]+\b/g) || [];
    capitalizedWords.forEach(word => entities.add(word));
    
    // Code identifiers
    const codeIds = text.match(/\b[a-zA-Z_][a-zA-Z0-9_]*[A-Z][a-zA-Z0-9_]*\b/g) || [];
    codeIds.forEach(id => entities.add(id));
    
    // Technical terms (words with specific patterns)
    const techTerms = text.match(/\b[a-z]+(?:[A-Z][a-z]*)+\b/g) || [];
    techTerms.forEach(term => entities.add(term));
    
    return Array.from(entities).slice(0, 20);
  }

  // State management and planning
  async updateState(sessionId: string, pack: ContextPack): Promise<void> {
    const currentState = this.db.getState(sessionId);
    
    const newEntities = pack.keyEntities;
    const existingEntities = currentState?.keyEntities || [];
    
    // Merge entities, keeping recent ones
    const mergedEntities = [...new Set([...newEntities, ...existingEntities])].slice(0, 200);
    
    const updatedState: SessionState = {
      sessionId,
      keyEntities: mergedEntities,
      lastPackClaims: pack.claims.map(c => c.text),
      lastPackContradictions: pack.contradictions.map(c => c.issue),
      planHint: this.determinePlanHint(pack, currentState)
    };
    
    this.db.updateState(sessionId, updatedState);
  }

  private determinePlanHint(pack: ContextPack, currentState: SessionState | null): PlanType {
    // If there are contradictions, suggest verify
    if (pack.contradictions.length > 0) {
      return 'verify';
    }
    
    // If there's low entity overlap with previous state, explore
    if (currentState?.keyEntities) {
      const overlap = pack.keyEntities.filter(e => 
        currentState.keyEntities.includes(e)
      ).length;
      const overlapRatio = overlap / Math.max(pack.keyEntities.length, 1);
      
      if (overlapRatio < 0.3) {
        return 'explore';
      }
    }
    
    // Default to exploit for focused information
    return 'exploit';
  }

  async getPlanType(sessionId: string): Promise<PlanType> {
    const state = this.db.getState(sessionId);
    return state?.planHint || 'explore';
  }
}