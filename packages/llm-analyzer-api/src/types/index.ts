// Re-export shared types
export * from '@lethe/llm-analyzer-shared';

// API-specific types
export interface ServerConfig {
  port: number;
  host: string;
  dbPath: string;
  corsOrigins: string[];
  logLevel: 'error' | 'warn' | 'info' | 'debug';
}

export interface RequestContext {
  requestId: string;
  startTime: number;
  userAgent?: string;
  ip: string;
}