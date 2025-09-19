import { readFile } from 'fs/promises';
import { Workspace } from '../workspace';
import type { MessageRole, SessionMessage } from '@lethe/types';

export interface IngestArgs {
  file: string;
  session?: string;
}

interface RawMessage {
  role: MessageRole;
  text: string;
  timestamp?: number;
}

export async function ingestCommand(args: IngestArgs): Promise<void> {
  if (!args.file) {
    throw new Error('Missing --file argument');
  }

  const raw = await readFile(args.file, 'utf8');
  const messages = JSON.parse(raw) as RawMessage[];
  const sessionId = args.session ?? 'default';

  const now = Date.now();
  const normalized: Array<Omit<SessionMessage, 'id'>> = messages.map((message, index) => ({
    role: message.role ?? 'user',
    text: message.text,
    sessionId,
    timestamp: message.timestamp ?? now + index,
  }));

  const workspace = new Workspace();
  const stored = await workspace.appendMessages(sessionId, normalized);

  console.log(`📥 Ingested ${normalized.length} messages into session ${sessionId}.`);
  console.log(` Session now contains ${stored.length} messages.`);
}
