import { mergeConfig, defaultConfig } from './config';
import { SessionStore } from './store';
import { generateContextPack } from './retriever';
import type {
  ContextPack,
  LetheConfig,
  Result,
  SessionMessage,
  LetheError,
} from '@lethe/types';

export interface OrchestratorOptions {
  config?: Partial<LetheConfig>;
  store?: SessionStore;
}

export class ContextOrchestrator {
  private readonly store: SessionStore;
  private config: LetheConfig;

  constructor(options: OrchestratorOptions = {}) {
    this.store = options.store ?? new SessionStore();
    this.config = mergeConfig(options.config);
  }

  getStore(): SessionStore {
    return this.store;
  }

  getConfig(): LetheConfig {
    return this.config;
  }

  updateConfig(partial: Partial<LetheConfig>): void {
    this.config = mergeConfig(partial);
  }

  resetConfig(): void {
    this.config = structuredClone(defaultConfig);
  }

  ingestMessage(message: SessionMessage): Result<SessionMessage, LetheError> {
    return this.execute(() => this.store.upsertMessage(message));
  }

  ingestMessages(messages: SessionMessage[]): Result<SessionMessage[], LetheError> {
    return this.execute(() => {
      this.store.upsertMessages(messages);
      return this.store.getMessages(messages[0]?.sessionId ?? '');
    });
  }

  listSessions() {
    return this.store.listSessions();
  }

  getMessages(sessionId: string) {
    return this.store.getMessages(sessionId);
  }

  buildContext(sessionId: string, query: string): Result<ContextPack, LetheError> {
    return this.execute(() => {
      const messages = this.store.getMessages(sessionId);
      const pack = generateContextPack({ messages, query, config: this.config });
      this.store.savePack(pack);
      return pack;
    });
  }

  private execute<T>(fn: () => T): Result<T, LetheError> {
    try {
      const data = fn();
      return { success: true, data };
    } catch (error) {
      const err = error as Error;
      return {
        success: false,
        error: {
          code: 'ORCHESTRATOR_ERROR',
          message: err.message,
          timestamp: Date.now(),
          stack: err.stack,
        },
      };
    }
  }
}
