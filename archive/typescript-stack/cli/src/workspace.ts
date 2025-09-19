import { readFile, writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import { join } from 'path';
import { randomUUID } from 'crypto';
import type {
  LetheConfig,
  SessionMessage,
  SessionSummary,
} from '@lethe/types';
import { defaultConfig } from '@lethe/core';

export interface WorkspaceData {
  config: LetheConfig;
  sessions: Record<string, {
    summary: SessionSummary;
    messages: SessionMessage[];
  }>;
}

export class Workspace {
  private readonly root: string;
  private readonly file: string;

  constructor(root = process.cwd()) {
    this.root = root;
    this.file = join(root, '.lethe', 'workspace.json');
  }

  get path(): string {
    return this.file;
  }

  async ensure(): Promise<void> {
    if (!existsSync(this.file)) {
      await mkdir(join(this.root, '.lethe'), { recursive: true });
      await writeFile(
        this.file,
        JSON.stringify({
          config: defaultConfig,
          sessions: {},
        } satisfies WorkspaceData, null, 2)
      );
    }
  }

  async load(): Promise<WorkspaceData> {
    await this.ensure();
    const raw = await readFile(this.file, 'utf8');
    const parsed = JSON.parse(raw) as WorkspaceData;
    return parsed;
  }

  async save(data: WorkspaceData): Promise<void> {
    await mkdir(join(this.root, '.lethe'), { recursive: true });
    await writeFile(this.file, JSON.stringify(data, null, 2));
  }

  async appendMessages(sessionId: string, messages: Omit<SessionMessage, 'id'>[]): Promise<SessionMessage[]> {
    const data = await this.load();
    const session = data.sessions[sessionId] ?? {
      summary: {
        sessionId,
        title: `Session ${sessionId.slice(0, 6)}`,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messageCount: 0,
      },
      messages: [],
    };

    const enriched: SessionMessage[] = messages.map((message) => ({
      ...message,
      id: randomUUID(),
    }));

    session.messages = session.messages.concat(enriched).sort((a, b) => a.timestamp - b.timestamp);
    session.summary = {
      ...session.summary,
      updatedAt: session.messages.at(-1)?.timestamp ?? session.summary.updatedAt,
      messageCount: session.messages.length,
    };

    data.sessions[sessionId] = session;
    await this.save(data);
    return session.messages;
  }
}
