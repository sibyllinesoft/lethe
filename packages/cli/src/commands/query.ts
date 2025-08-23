import { join } from 'path';
import chalk from 'chalk';
import ora from 'ora';
import { openDb } from '@ctx-run/sqlite';
import { ContextOrchestrator } from '@ctx-run/core';
import { getDefaultProvider } from '@ctx-run/embeddings';
import { CrossEncoderReranker } from '@ctx-run/reranker';

export interface QueryOptions {
  session: string;
  format?: 'json' | 'pretty' | 'summary';
  output?: string;
  debug?: boolean;
}

export async function queryCommand(query: string, options: QueryOptions): Promise<void> {
  const ctxDir = join(process.cwd(), '.ctx');
  const dbPath = join(ctxDir, 'ctx.db');

  if (!query?.trim()) {
    console.error(chalk.red('❌ Query cannot be empty'));
    process.exit(1);
  }

  console.log(chalk.blue(`❓ Querying session: ${options.session}`));
  console.log(chalk.gray(`   Query: "${query}"`));

  const spinner = ora('Initializing search...').start();

  try {
    // Open database and initialize
    const db = await openDb(dbPath);
    const config = db.getConfig('system') || ContextOrchestrator.getDefaultConfig();
    
    spinner.text = 'Loading AI models...';
    const embeddings = await getDefaultProvider();
    const reranker = new CrossEncoderReranker();
    
    const orchestrator = new ContextOrchestrator(db, embeddings, reranker, config);
    
    // Check session exists and is indexed
    const stats = await orchestrator.getSessionStats(options.session);
    if (stats.messageCount === 0) {
      spinner.fail(chalk.yellow(`⚠️  Session '${options.session}' not found or empty`));
      db.close();
      return;
    }
    
    if (stats.embeddingCount === 0) {
      spinner.fail(chalk.yellow(`⚠️  Session '${options.session}' is not indexed. Run: npx ctx-run index --session ${options.session}`));
      db.close();
      return;
    }
    
    spinner.text = 'Searching conversation history...';
    
    // Execute search
    const result = await orchestrator.orchestrateTurn(options.session, query);
    
    db.close();
    
    spinner.succeed(chalk.green('✅ Search completed'));
    
    // Output results
    console.log();
    
    switch (options.format || 'pretty') {
      case 'json':
        if (options.output) {
          const { writeFile } = await import('fs/promises');
          await writeFile(options.output, JSON.stringify(result.pack, null, 2));
          console.log(chalk.green(`📁 Results saved to: ${options.output}`));
        } else {
          console.log(JSON.stringify(result.pack, null, 2));
        }
        break;
        
      case 'summary':
        outputSummary(result);
        break;
        
      default: // 'pretty'
        outputPretty(result, options.debug);
    }

  } catch (error) {
    spinner.fail(chalk.red('❌ Query failed'));
    console.error(chalk.red('Error:'), error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

function outputPretty(result: any, debug?: boolean): void {
  const { pack } = result;
  
  console.log(chalk.blue('📋 Summary:'));
  console.log(chalk.white(pack.summary));
  console.log();
  
  if (pack.keyEntities.length > 0) {
    console.log(chalk.blue('🏷️  Key Entities:'));
    console.log(chalk.gray('   ' + pack.keyEntities.join(', ')));
    console.log();
  }
  
  if (pack.claims.length > 0) {
    console.log(chalk.blue('✅ Claims:'));
    pack.claims.forEach((claim: any, i: number) => {
      console.log(chalk.white(`   ${i + 1}. ${claim.text}`));
      if (claim.chunks.length > 0) {
        console.log(chalk.gray(`      Sources: ${claim.chunks.join(', ')}`));
      }
    });
    console.log();
  }
  
  if (pack.contradictions.length > 0) {
    console.log(chalk.yellow('⚠️  Contradictions:'));
    pack.contradictions.forEach((contradiction: any, i: number) => {
      console.log(chalk.yellow(`   ${i + 1}. ${contradiction.issue}`));
      if (contradiction.chunks.length > 0) {
        console.log(chalk.gray(`      Sources: ${contradiction.chunks.join(', ')}`));
      }
    });
    console.log();
  }
  
  if (Object.keys(pack.citations).length > 0) {
    console.log(chalk.blue('📖 Citations:'));
    Object.entries(pack.citations).forEach(([chunkId, citation]: [string, any]) => {
      console.log(chalk.gray(`   ${chunkId}: message ${citation.messageId} (${citation.span[0]}-${citation.span[1]})`));
    });
    console.log();
  }
  
  if (debug && result.debug) {
    console.log(chalk.blue('🔍 Debug Information:'));
    console.log(chalk.gray(`   Plan type: ${result.debug.planType}`));
    console.log(chalk.gray(`   HyDE queries: ${result.debug.hydeQueries.length}`));
    console.log(chalk.gray(`   Candidates: ${result.debug.candidateCount}`));
    console.log(chalk.gray(`   Rerank time: ${result.debug.rerankTime}ms`));
    console.log(chalk.gray(`   Total time: ${result.debug.totalTime}ms`));
    
    if (result.debug.hydeQueries.length > 0) {
      console.log(chalk.blue('🤖 Generated queries:'));
      result.debug.hydeQueries.forEach((query: string, i: number) => {
        console.log(chalk.gray(`   ${i + 1}. ${query}`));
      });
    }
  }
}

function outputSummary(result: any): void {
  const { pack } = result;
  
  console.log(pack.summary);
  
  if (pack.claims.length > 0) {
    console.log();
    console.log('Key points:');
    pack.claims.forEach((claim: any, i: number) => {
      console.log(`• ${claim.text}`);
    });
  }
  
  if (pack.contradictions.length > 0) {
    console.log();
    console.log('Potential issues:');
    pack.contradictions.forEach((contradiction: any) => {
      console.log(`⚠ ${contradiction.issue}`);
    });
  }
}