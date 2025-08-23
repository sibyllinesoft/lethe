import { promises as fs } from 'fs';
import { createReadStream } from 'fs';
import { createInterface } from 'readline';
import { join } from 'path';
import chalk from 'chalk';
import ora from 'ora';
import { openDb } from '@ctx-run/sqlite';
import { ContextOrchestrator } from '@ctx-run/core';
import { getDefaultProvider } from '@ctx-run/embeddings';
import { CrossEncoderReranker } from '@ctx-run/reranker';
import type { Message } from '@ctx-run/sqlite';
import { v4 as uuidv4 } from 'uuid';

export interface IngestOptions {
  session: string;
  from: string;
  format?: 'auto' | 'json' | 'jsonl' | 'claude-export';
}

interface ClaudeExportMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

export async function ingestCommand(options: IngestOptions): Promise<void> {
  const ctxDir = join(process.cwd(), '.ctx');
  const dbPath = join(ctxDir, 'ctx.db');

  console.log(chalk.blue(`📥 Ingesting data for session: ${options.session}`));

  // Check if workspace is initialized
  try {
    await fs.access(dbPath);
  } catch {
    console.error(chalk.red('❌ No ctx-run workspace found. Run `npx ctx-run init` first.'));
    process.exit(1);
  }

  const spinner = ora('Loading messages...').start();

  try {
    // Load messages based on source
    let messages: Message[];
    
    if (options.from === 'stdio' || options.from === '-') {
      messages = await loadFromStdin(options.session);
    } else {
      messages = await loadFromFile(options.from, options.session, options.format);
    }

    if (messages.length === 0) {
      spinner.fail(chalk.yellow('⚠️  No messages found to ingest'));
      return;
    }

    spinner.text = `Processing ${messages.length} messages...`;

    // Open database and initialize orchestrator
    const db = await openDb(dbPath);
    const config = db.getConfig('system') || ContextOrchestrator.getDefaultConfig();
    const embeddings = await getDefaultProvider();
    const reranker = new CrossEncoderReranker();
    
    const orchestrator = new ContextOrchestrator(db, embeddings, reranker, config);

    // Ingest messages
    await orchestrator.ingestMessages(options.session, messages);

    db.close();
    
    spinner.succeed(chalk.green(`✅ Successfully ingested ${messages.length} messages`));
    
    console.log();
    console.log(chalk.blue('📊 Session stats:'));
    console.log(chalk.gray(`   Messages: ${messages.length}`));
    console.log(chalk.gray(`   Session ID: ${options.session}`));
    console.log(chalk.gray(`   Time range: ${getTimeRange(messages)}`));
    
    console.log();
    console.log(chalk.blue('🚀 Next step:'));
    console.log(chalk.blue(`   npx ctx-run index --session ${options.session}`));

  } catch (error) {
    spinner.fail(chalk.red('❌ Ingestion failed'));
    console.error(chalk.red('Error:'), error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

async function loadFromFile(
  filePath: string, 
  sessionId: string,
  format?: string
): Promise<Message[]> {
  try {
    await fs.access(filePath);
  } catch {
    throw new Error(`File not found: ${filePath}`);
  }

  const content = await fs.readFile(filePath, 'utf-8');
  const detectedFormat = format || detectFormat(filePath, content);

  switch (detectedFormat) {
    case 'json':
      return parseJsonMessages(content, sessionId);
    case 'jsonl':
      return parseJsonlMessages(content, sessionId);
    case 'claude-export':
      return parseClaudeExport(content, sessionId);
    default:
      throw new Error(`Unsupported format: ${detectedFormat}`);
  }
}

async function loadFromStdin(sessionId: string): Promise<Message[]> {
  const rl = createInterface({
    input: process.stdin,
    crlfDelay: Infinity
  });

  const lines: string[] = [];
  for await (const line of rl) {
    lines.push(line);
  }

  const content = lines.join('\n');
  
  // Try to detect format from content
  if (lines.every(line => {
    try { JSON.parse(line); return true; } catch { return false; }
  })) {
    return parseJsonlMessages(content, sessionId);
  } else {
    try {
      return parseJsonMessages(content, sessionId);
    } catch {
      throw new Error('Unable to parse stdin input. Expected JSON or JSONL format.');
    }
  }
}

function detectFormat(filePath: string, content: string): string {
  const ext = filePath.toLowerCase().split('.').pop();
  
  if (ext === 'jsonl' || ext === 'ndjson') {
    return 'jsonl';
  }
  
  if (ext === 'json') {
    // Try to detect if it's a Claude export by looking for specific structure
    try {
      const parsed = JSON.parse(content);
      if (Array.isArray(parsed) && parsed[0]?.role && parsed[0]?.content) {
        return 'claude-export';
      }
    } catch {}
    return 'json';
  }
  
  return 'json'; // Default
}

function parseJsonMessages(content: string, sessionId: string): Message[] {
  const parsed = JSON.parse(content);
  
  if (!Array.isArray(parsed)) {
    throw new Error('JSON content must be an array of messages');
  }
  
  return parsed.map((item, index) => normalizeMessage(item, sessionId, index));
}

function parseJsonlMessages(content: string, sessionId: string): Message[] {
  const lines = content.trim().split('\n').filter(line => line.trim());
  const messages: Message[] = [];
  
  for (let i = 0; i < lines.length; i++) {
    try {
      const parsed = JSON.parse(lines[i]);
      messages.push(normalizeMessage(parsed, sessionId, i));
    } catch (error) {
      console.warn(chalk.yellow(`⚠️  Skipping invalid JSON line ${i + 1}`));
    }
  }
  
  return messages;
}

function parseClaudeExport(content: string, sessionId: string): Message[] {
  const parsed = JSON.parse(content) as ClaudeExportMessage[];
  
  return parsed.map((msg, index) => ({
    id: uuidv4(),
    sessionId,
    turn: index,
    role: msg.role,
    text: msg.content,
    ts: msg.timestamp ? new Date(msg.timestamp).getTime() : Date.now() + index,
    meta: { originalTimestamp: msg.timestamp }
  }));
}

function normalizeMessage(item: any, sessionId: string, index: number): Message {
  // Handle different message formats
  const message: Message = {
    id: item.id || uuidv4(),
    sessionId,
    turn: item.turn !== undefined ? item.turn : index,
    role: item.role || 'user',
    text: item.text || item.content || item.message || '',
    ts: item.ts || item.timestamp || (Date.now() + index),
    meta: item.meta || {}
  };
  
  // Validate required fields
  if (!message.text) {
    throw new Error(`Message at index ${index} has no text content`);
  }
  
  if (!['user', 'assistant', 'system', 'tool'].includes(message.role)) {
    console.warn(chalk.yellow(`⚠️  Unknown role '${message.role}' at index ${index}, using 'user'`));
    message.role = 'user';
  }
  
  return message;
}

function getTimeRange(messages: Message[]): string {
  if (messages.length === 0) return 'N/A';
  
  const timestamps = messages.map(m => m.ts).filter(ts => ts > 0);
  if (timestamps.length === 0) return 'N/A';
  
  const min = Math.min(...timestamps);
  const max = Math.max(...timestamps);
  
  const start = new Date(min).toISOString().split('T')[0];
  const end = new Date(max).toISOString().split('T')[0];
  
  return start === end ? start : `${start} to ${end}`;
}