import type { LLMCall } from '@lethe/llm-analyzer-shared';
import { DatabaseManager } from '../db/database';
import { z } from 'zod';

// Schema for validating incoming log entries
const LogEntrySchema = z.object({
  id: z.string().optional(),
  timestamp: z.string(),
  provider: z.string(),
  model: z.string(),
  endpoint: z.string(),
  method: z.string().default('POST'),
  status: z.number(),
  
  // Request data
  requestHeaders: z.record(z.string()).default({}),
  requestBody: z.unknown(),
  
  // Response data
  responseHeaders: z.record(z.string()).default({}),
  responseBody: z.unknown(),
  
  // Metrics
  duration: z.number(),
  inputTokens: z.number().optional(),
  outputTokens: z.number().optional(),
  totalTokens: z.number().optional(),
  cost: z.number().optional(),
  
  // Metadata
  userId: z.string().optional(),
  sessionId: z.string().optional(),
  tags: z.array(z.string()).default([]),
  
  // Error information
  error: z.object({
    type: z.string(),
    message: z.string(),
    stack: z.string().optional()
  }).optional()
});

export interface IngestResult {
  processed: number;
  errors: Array<{
    line: number;
    error: string;
  }>;
}

export class NDJSONIngester {
  constructor(private db: DatabaseManager) {}

  async ingest(ndjsonData: string): Promise<IngestResult> {
    const lines = ndjsonData.split('\n').filter(line => line.trim());
    const errors: Array<{ line: number; error: string }> = [];
    let processed = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const lineNumber = i + 1;

      try {
        if (!line?.trim()) continue;

        const rawData = JSON.parse(line);
        const validatedData = LogEntrySchema.parse(rawData);
        
        // Generate ID if not provided
        const id = validatedData.id || this.generateId(validatedData);
        
        const llmCall: LLMCall = {
          ...validatedData,
          id,
          // Ensure all required fields have values
          requestHeaders: validatedData.requestHeaders,
          responseHeaders: validatedData.responseHeaders,
          tags: validatedData.tags
        };

        this.db.insertCall(llmCall);
        processed++;

      } catch (error) {
        errors.push({
          line: lineNumber,
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    }

    return { processed, errors };
  }

  private generateId(data: Omit<LLMCall, 'id'>): string {
    // Generate a unique ID based on timestamp, provider, and model
    const timestamp = new Date(data.timestamp).getTime();
    const hash = this.simpleHash(`${timestamp}-${data.provider}-${data.model}-${Math.random()}`);
    return `call_${hash}`;
  }

  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  async ingestFromFile(filePath: string): Promise<IngestResult> {
    const { readFileSync } = await import('fs');
    const content = readFileSync(filePath, 'utf-8');
    return this.ingest(content);
  }

  // Utility method to transform proxy logs to our format
  static transformProxyLog(proxyLog: any): Partial<LLMCall> {
    return {
      timestamp: proxyLog.timestamp || new Date().toISOString(),
      provider: proxyLog.provider || 'unknown',
      model: proxyLog.model || 'unknown',
      endpoint: proxyLog.url || proxyLog.endpoint || '',
      method: proxyLog.method || 'POST',
      status: proxyLog.status || proxyLog.statusCode || 0,
      
      requestHeaders: proxyLog.requestHeaders || {},
      requestBody: proxyLog.requestBody || proxyLog.request || {},
      
      responseHeaders: proxyLog.responseHeaders || {},
      responseBody: proxyLog.responseBody || proxyLog.response || {},
      
      duration: proxyLog.duration || proxyLog.responseTime || 0,
      inputTokens: proxyLog.inputTokens || proxyLog.usage?.prompt_tokens,
      outputTokens: proxyLog.outputTokens || proxyLog.usage?.completion_tokens,
      totalTokens: proxyLog.totalTokens || proxyLog.usage?.total_tokens,
      cost: proxyLog.cost,
      
      userId: proxyLog.userId || proxyLog.user_id,
      sessionId: proxyLog.sessionId || proxyLog.session_id,
      tags: proxyLog.tags || [],
      
      error: proxyLog.error ? {
        type: proxyLog.error.type || 'unknown',
        message: proxyLog.error.message || 'Unknown error',
        stack: proxyLog.error.stack
      } : undefined
    };
  }
}