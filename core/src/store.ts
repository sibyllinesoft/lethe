import { randomUUID } from 'crypto';
import type {
  Result,
  LetheError,
  SessionMessage,
  SessionSummary,
  ContextPack,
} from '@lethe/types';

export class SessionStore {
  private sessions = new Map<string, SessionSummary>();
  private messages = new Map<string, SessionMessage[]>();
  private packs = new Map<string, ContextPack>();

  listSessions(): SessionSummary[] {
    return Array.from(this.sessions.values()).sort((a, b) => b.updatedAt - a.updatedAt);
  }

  getSession(sessionId: string): SessionSummary | undefined {
    return this.sessions.get(sessionId);
  }

  getMessages(sessionId: string): SessionMessage[] {
    return this.messages.get(sessionId) ?? [];
  }

  upsertMessage(message: Omit<SessionMessage, 'id'> & { id?: string }): SessionMessage {
    const id = message.id ?? randomUUID();
    const record: SessionMessage = { ...message, id };
    const list = this.messages.get(message.sessionId) ?? [];
    const next = list.filter((item) => item.id !== id);
    next.push(record);
    next.sort((a, b) => a.timestamp - b.timestamp);
    this.messages.set(message.sessionId, next);

    const summary = this.sessions.get(message.sessionId) ?? {
      sessionId: message.sessionId,
      title: `Session ${message.sessionId.slice(0, 6)}`,
      createdAt: record.timestamp,
      updatedAt: record.timestamp,
      messageCount: 0,
    };

    this.sessions.set(message.sessionId, {
      ...summary,
      updatedAt: record.timestamp,
      messageCount: next.length,
    });

    return record;
  }

  upsertMessages(messages: Array<Omit<SessionMessage, 'id'> & { id?: string }>): void {
    for (const message of messages) {
      this.upsertMessage(message);
    }
  }

  savePack(pack: ContextPack): ContextPack {
    this.packs.set(pack.id, pack);
    return pack;
  }

  listPacks(sessionId?: string): ContextPack[] {
    const packs = Array.from(this.packs.values());
    return sessionId ? packs.filter((pack) => pack.sessionId === sessionId) : packs;
  }

  clear(): void {
    this.sessions.clear();
    this.messages.clear();
    this.packs.clear();
  }
}

export function guard<T>(cb: () => T): Result<T, LetheError> {
  try {
    return { success: true, data: cb() };
  } catch (error) {
    const err = error as Error;
    return {
      success: false,
      error: {
        code: 'STORE_ERROR',
        message: err.message,
        timestamp: Date.now(),
        stack: err.stack,
      },
    };
  }
}
