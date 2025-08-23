import { join } from 'path';
import chalk from 'chalk';
import ora from 'ora';
import { openDb } from '@ctx-run/sqlite';
import { ContextOrchestrator } from '@ctx-run/core';
import { getDefaultProvider } from '@ctx-run/embeddings';
import { CrossEncoderReranker } from '@ctx-run/reranker';

export interface IndexOptions {
  session: string;
  force?: boolean;
}

export async function indexCommand(options: IndexOptions): Promise<void> {
  const ctxDir = join(process.cwd(), '.ctx');
  const dbPath = join(ctxDir, 'ctx.db');

  console.log(chalk.blue(`🔍 Building search indexes for session: ${options.session}`));

  const spinner = ora('Loading workspace...').start();

  try {
    // Open database
    const db = await openDb(dbPath);
    const config = db.getConfig('system') || ContextOrchestrator.getDefaultConfig();
    
    spinner.text = 'Initializing providers...';
    
    // Initialize providers
    const embeddings = await getDefaultProvider();
    const reranker = new CrossEncoderReranker();
    
    const orchestrator = new ContextOrchestrator(db, embeddings, reranker, config);
    
    // Get session stats
    const stats = await orchestrator.getSessionStats(options.session);
    
    if (stats.messageCount === 0) {
      spinner.fail(chalk.yellow(`⚠️  No messages found for session: ${options.session}`));
      db.close();
      return;
    }
    
    spinner.text = `Processing ${stats.chunkCount} chunks...`;
    
    // Check if already indexed
    if (stats.embeddingCount === stats.chunkCount && !options.force) {
      spinner.succeed(chalk.green('✅ Session is already fully indexed'));
      console.log();
      console.log(chalk.blue('📊 Index stats:'));
      console.log(chalk.gray(`   Messages: ${stats.messageCount}`));
      console.log(chalk.gray(`   Chunks: ${stats.chunkCount}`));
      console.log(chalk.gray(`   Embeddings: ${stats.embeddingCount}`));
      console.log(chalk.gray(`   Packs: ${stats.packCount}`));
      console.log(chalk.gray('   Use --force to rebuild indexes'));
      db.close();
      return;
    }
    
    const missingEmbeddings = stats.chunkCount - stats.embeddingCount;
    if (missingEmbeddings > 0) {
      spinner.text = `Generating ${missingEmbeddings} embeddings...`;
    }
    
    // Run indexing
    await orchestrator.indexSession(options.session);
    
    // Get updated stats
    const updatedStats = await orchestrator.getSessionStats(options.session);
    
    db.close();
    
    spinner.succeed(chalk.green('✅ Indexing completed successfully!'));
    
    console.log();
    console.log(chalk.blue('📊 Final index stats:'));
    console.log(chalk.gray(`   Messages: ${updatedStats.messageCount}`));
    console.log(chalk.gray(`   Chunks: ${updatedStats.chunkCount}`));
    console.log(chalk.gray(`   Embeddings: ${updatedStats.embeddingCount}`));
    console.log(chalk.gray(`   Coverage: ${(updatedStats.embeddingCount / updatedStats.chunkCount * 100).toFixed(1)}%`));
    
    console.log();
    console.log(chalk.blue('🚀 Ready to query:'));
    console.log(chalk.blue(`   npx ctx-run query "your question" --session ${options.session}`));

  } catch (error) {
    spinner.fail(chalk.red('❌ Indexing failed'));
    console.error(chalk.red('Error:'), error instanceof Error ? error.message : error);
    process.exit(1);
  }
}