import type { CtxDatabase, Message } from '@ctx-run/sqlite';
import type { EmbeddingProvider } from '@ctx-run/embeddings';
import type { CrossEncoderReranker } from '@ctx-run/reranker';
import type { Config, ContextPack, PlanType } from './types.js';

import { MessageChunker } from './chunker.js';
import { DfIdfBuilder } from './dfidf.js';
import { RetrievalSystem } from './retrieval.js';
import { AIIntegration } from './ai-integration.js';

export interface OrchestrationResult {
  pack: ContextPack;
  debug: {
    planType: PlanType;
    hydeQueries: string[];
    candidateCount: number;
    rerankTime: number;
    totalTime: number;
  };
}

export class ContextOrchestrator {
  private db: CtxDatabase;
  private chunker: MessageChunker;
  private dfIdfBuilder: DfIdfBuilder;
  private retrieval: RetrievalSystem;
  private ai: AIIntegration;
  private config: Config;

  constructor(
    db: CtxDatabase,
    embeddings: EmbeddingProvider,
    reranker: CrossEncoderReranker,
    config: Config
  ) {
    this.db = db;
    this.config = config;
    
    this.chunker = new MessageChunker({
      targetTokens: config.chunking.target_tokens,
      overlap: config.chunking.overlap,
      splitCodeBlocks: config.chunking.split_code_blocks,
      splitSentences: config.chunking.split_sentences
    });
    
    this.dfIdfBuilder = new DfIdfBuilder(db);
    this.retrieval = new RetrievalSystem(db, embeddings, reranker, config);
    
    this.ai = new AIIntegration(db, {
      ollamaUrl: 'http://localhost:11434',
      hydeModel: config.models.hyde || 'xgen-small:4b',
      summarizeModel: config.models.summarize || 'xgen-small:4b',
      timeout: 30000
    });
  }

  // Main orchestration method
  async orchestrateTurn(
    sessionId: string,
    userQuery: string
  ): Promise<OrchestrationResult> {
    const startTime = Date.now();
    
    // Step 1: Determine plan type
    const planType = await this.ai.getPlanType(sessionId);
    const planConfig = this.config.plan[planType];
    
    console.log(`Using plan: ${planType} (β=${planConfig.beta}, k=${planConfig.hyde_k})`);
    
    // Step 2: Generate HyDE queries
    const hydeResult = await this.ai.generateHyde(
      userQuery,
      sessionId,
      planConfig.hyde_k,
      planType
    );
    
    console.log(`Generated ${hydeResult.queries.length} HyDE queries`);
    
    // Step 3: Retrieve candidates using hybrid search
    const searchQueries = [...hydeResult.queries, hydeResult.pseudo];
    const candidates = await this.retrieval.search(
      sessionId,
      searchQueries,
      this.config.rerank.topk_in
    );
    
    console.log(`Retrieved ${candidates.length} candidates`);
    
    if (candidates.length === 0) {
      const emptyPack: ContextPack = {
        id: 'empty-' + Date.now(),
        sessionId,
        query: userQuery,
        summary: 'No relevant information found in conversation history.',
        keyEntities: [],
        claims: [],
        contradictions: [],
        citations: {}
      };
      
      return {
        pack: emptyPack,
        debug: {
          planType,
          hydeQueries: hydeResult.queries,
          candidateCount: 0,
          rerankTime: 0,
          totalTime: Date.now() - startTime
        }
      };
    }
    
    // Step 4: Rerank candidates
    const rerankStart = Date.now();
    const rerankedCandidates = await this.retrieval.rerank(
      userQuery,
      candidates,
      this.config.rerank.topk_out
    );
    const rerankTime = Date.now() - rerankStart;
    
    console.log(`Reranked to ${rerankedCandidates.length} candidates`);
    
    // Step 5: Submodular selection for diversity
    const selectedCandidates = this.retrieval.submodularSelect(
      rerankedCandidates,
      this.config.diversify.pack_chunks
    );
    
    console.log(`Selected ${selectedCandidates.length} chunks for pack`);
    
    // Step 6: Build context pack
    const pack = await this.ai.buildPack(
      userQuery,
      selectedCandidates,
      sessionId,
      planConfig.granularity
    );
    
    // Step 7: Update state and store pack
    await this.ai.updateState(sessionId, pack);
    this.db.insertPack(pack);
    
    const totalTime = Date.now() - startTime;
    console.log(`Orchestration complete in ${totalTime}ms`);
    
    return {
      pack,
      debug: {
        planType,
        hydeQueries: hydeResult.queries,
        candidateCount: candidates.length,
        rerankTime,
        totalTime
      }
    };
  }

  // Ingestion pipeline
  async ingestMessages(sessionId: string, messages: Message[]): Promise<void> {
    console.log(`Ingesting ${messages.length} messages for session ${sessionId}`);
    
    // Store messages
    this.db.insertMessages(messages);
    
    // Chunk messages
    const allChunks = this.chunker.rechunkMessages(messages);
    console.log(`Generated ${allChunks.length} chunks`);
    
    // Store chunks
    this.db.insertChunks(allChunks);
    
    // Rebuild DF/IDF asynchronously
    setTimeout(async () => {
      try {
        await this.dfIdfBuilder.rebuild(sessionId);
        console.log(`DF/IDF rebuilt for session ${sessionId}`);
      } catch (error) {
        console.error(`Failed to rebuild DF/IDF: ${error}`);
      }
    }, 100);
  }

  // Indexing pipeline
  async indexSession(sessionId: string): Promise<void> {
    console.log(`Indexing session ${sessionId}`);
    
    // Ensure embeddings for all chunks
    await this.retrieval.ensureEmbeddings(sessionId);
    
    // Rebuild DF/IDF
    await this.retrieval.rebuildDfIdf(sessionId);
    
    console.log(`Indexing complete for session ${sessionId}`);
  }

  // Configuration management
  getConfig(): Config {
    return this.config;
  }

  updateConfig(updates: Partial<Config>): void {
    this.config = { ...this.config, ...updates };
    
    // Store updated config
    this.db.setConfig('system', this.config);
  }

  // Session management utilities
  async getSessionStats(sessionId: string): Promise<{
    messageCount: number;
    chunkCount: number;
    embeddingCount: number;
    packCount: number;
    lastActivity: number;
  }> {
    const messages = this.db.getMessages(sessionId);
    const chunks = this.db.getChunks(sessionId);
    const packs = this.db.getPacks(sessionId);
    
    const embeddedChunks = this.db.getChunksWithoutEmbeddings(sessionId);
    const embeddingCount = chunks.length - embeddedChunks.length;
    
    const lastActivity = messages.length > 0 ? 
      Math.max(...messages.map(m => m.ts)) : 0;
    
    return {
      messageCount: messages.length,
      chunkCount: chunks.length,
      embeddingCount,
      packCount: packs.length,
      lastActivity
    };
  }

  async getSessions(): Promise<Array<{
    sessionId: string;
    messageCount: number;
    lastActivity: number;
  }>> {
    // This would need a proper session index in a real implementation
    // For now, we'll return empty array - this can be enhanced
    return [];
  }

  // Default configuration
  static getDefaultConfig(): Config {
    return {
      models: {
        embed: 'Xenova/bge-small-en-v1.5',
        rerank: 'Xenova/bge-reranker-base',
        hyde: 'xgen-small:4b',
        summarize: 'xgen-small:4b'
      },
      retrieval: {
        alpha: 1.0,
        beta: 0.9,
        gamma_kind_boost: {
          tool_result: 0.15,
          user_code: 0.08
        }
      },
      chunking: {
        target_tokens: 320,
        overlap: 64,
        split_code_blocks: true,
        split_sentences: true
      },
      rerank: {
        topk_in: 100,
        topk_out: 50,
        batch_size: 8
      },
      diversify: {
        pack_chunks: 24
      },
      plan: {
        explore: { hyde_k: 3, granularity: 'loose', beta: 1.1 },
        verify: { hyde_k: 5, granularity: 'tight', beta: 0.7 },
        exploit: { hyde_k: 3, granularity: 'medium', beta: 0.9 }
      }
    };
  }
}