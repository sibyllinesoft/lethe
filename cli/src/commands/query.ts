import { ContextOrchestrator } from '@lethe/core';
import type { LetheConfig } from '@lethe/types';
import { Workspace } from '../workspace';

export interface QueryArgs {
  session: string;
  query: string;
  topK?: number;
  config?: string;
}

export async function queryCommand(args: QueryArgs): Promise<void> {
  if (!args.session) {
    throw new Error('Missing --session');
  }

  if (!args.query) {
    throw new Error('Missing --query');
  }

  const workspace = new Workspace();
  const data = await workspace.load();
  const config = args.config
    ? (JSON.parse(args.config) as Partial<LetheConfig>)
    : data.config;
  const orchestrator = new ContextOrchestrator({ config });

  const messages = data.sessions[args.session]?.messages ?? [];
  if (messages.length === 0) {
    console.log('⚠️  No messages found for session', args.session);
    return;
  }

  const ingest = orchestrator.ingestMessages(messages);
  if (!ingest.success) {
    console.error('❌ Failed to load workspace messages:', ingest.error.message);
    return;
  }

  const result = orchestrator.buildContext(args.session, args.query);
  if (!result.success) {
    console.error('❌ Failed to build context:', result.error.message);
    return;
  }

  const top = args.topK ? result.data.messages.slice(0, args.topK) : result.data.messages;

  console.log('🧠 Context Pack:', result.data.summary);
  console.log('Messages:');
  top.forEach((candidate, index) => {
    console.log(` ${index + 1}. [${candidate.hybridScore.toFixed(2)}] ${candidate.message.role}`);
    console.log(`    ${candidate.message.text.slice(0, 180)}`);
  });
}
