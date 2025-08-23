import { join } from 'path';
import { writeFile } from 'fs/promises';
import chalk from 'chalk';
import ora from 'ora';
import { openDb } from '@ctx-run/sqlite';
import { ContextOrchestrator } from '@ctx-run/core';
import { getDefaultProvider } from '@ctx-run/embeddings';
import { CrossEncoderReranker } from '@ctx-run/reranker';

export interface EvalOptions {
  suite: string;
  output?: string;
  tune?: boolean;
  iterations?: number;
}

interface EvaluationSuite {
  name: string;
  description: string;
  sessionId: string;
  queries: EvaluationQuery[];
}

interface EvaluationQuery {
  id: string;
  query: string;
  relevantChunks: string[];
  expectedSummary?: string;
  tags?: string[];
}

interface EvaluationResult {
  queryId: string;
  query: string;
  metrics: {
    ndcg_5: number;
    ndcg_10: number;
    recall_5: number;
    recall_10: number;
    latency: number;
  };
  retrievedChunks: string[];
  relevantChunks: string[];
}

interface EvaluationReport {
  suite: string;
  sessionId: string;
  timestamp: number;
  config: any;
  results: EvaluationResult[];
  aggregated: {
    mean_ndcg_5: number;
    mean_ndcg_10: number;
    mean_recall_5: number;
    mean_recall_10: number;
    mean_latency: number;
    p50_latency: number;
    p90_latency: number;
  };
}

export async function evalCommand(options: EvalOptions): Promise<void> {
  const ctxDir = join(process.cwd(), '.ctx');
  const dbPath = join(ctxDir, 'ctx.db');

  console.log(chalk.blue(`📊 Running evaluation suite: ${options.suite}`));

  const spinner = ora('Loading workspace...').start();

  try {
    // Load evaluation suite
    const suite = await loadEvaluationSuite(options.suite);
    
    spinner.text = 'Initializing providers...';
    
    // Open database and initialize providers
    const db = await openDb(dbPath);
    let config = db.getConfig('system') || ContextOrchestrator.getDefaultConfig();
    
    const embeddings = await getDefaultProvider();
    const reranker = new CrossEncoderReranker();
    
    let bestConfig = config;
    let bestScore = 0;
    
    if (options.tune) {
      spinner.text = 'Tuning parameters...';
      console.log(chalk.yellow('\n🔧 Parameter tuning enabled'));
      
      const tuningResult = await tuneParameters(db, embeddings, reranker, suite, options.iterations || 10);
      bestConfig = tuningResult.config;
      bestScore = tuningResult.score;
      
      console.log(chalk.green(`✅ Best configuration found with nDCG@10 = ${bestScore.toFixed(4)}`));
      console.log(chalk.blue(`   α = ${bestConfig.retrieval.alpha}, β = ${bestConfig.retrieval.beta}`));
      
      // Update configuration in database
      db.setConfig('system', bestConfig);
    }

    const orchestrator = new ContextOrchestrator(db, embeddings, reranker, bestConfig);
    
    spinner.text = `Evaluating ${suite.queries.length} queries...`;
    
    // Run evaluation
    const results: EvaluationResult[] = [];
    
    for (let i = 0; i < suite.queries.length; i++) {
      const evalQuery = suite.queries[i];
      spinner.text = `Evaluating query ${i + 1}/${suite.queries.length}: ${evalQuery.query.substring(0, 40)}...`;
      
      const startTime = Date.now();
      
      try {
        const result = await orchestrator.orchestrateTurn(suite.sessionId, evalQuery.query);
        
        const latency = Date.now() - startTime;
        
        // Extract retrieved chunk IDs (simplified - in reality would need to map from result)
        const retrievedChunks = result.pack.citations ? 
          Object.values(result.pack.citations).map(c => c.messageId).slice(0, 10) : 
          [];
        
        // Calculate metrics
        const metrics = calculateMetrics(retrievedChunks, evalQuery.relevantChunks, latency);
        
        results.push({
          queryId: evalQuery.id,
          query: evalQuery.query,
          metrics,
          retrievedChunks,
          relevantChunks: evalQuery.relevantChunks
        });
        
      } catch (error) {
        console.warn(chalk.yellow(`⚠️  Query ${evalQuery.id} failed: ${error.message}`));
        results.push({
          queryId: evalQuery.id,
          query: evalQuery.query,
          metrics: {
            ndcg_5: 0,
            ndcg_10: 0,
            recall_5: 0,
            recall_10: 0,
            latency: 0
          },
          retrievedChunks: [],
          relevantChunks: evalQuery.relevantChunks
        });
      }
    }
    
    db.close();
    
    // Calculate aggregated metrics
    const aggregated = calculateAggregatedMetrics(results);
    
    // Generate report
    const report: EvaluationReport = {
      suite: suite.name,
      sessionId: suite.sessionId,
      timestamp: Date.now(),
      config: bestConfig,
      results,
      aggregated
    };
    
    spinner.succeed(chalk.green('✅ Evaluation completed!'));
    
    // Display results
    displayResults(report);
    
    // Save report
    const outputPath = options.output || `eval-report-${Date.now()}.json`;
    await writeFile(outputPath, JSON.stringify(report, null, 2));
    
    console.log(chalk.blue(`📁 Report saved to: ${outputPath}`));
    
    // Save CSV summary
    const csvPath = outputPath.replace('.json', '.csv');
    await writeFile(csvPath, generateCSV(report));
    
    console.log(chalk.blue(`📊 CSV summary saved to: ${csvPath}`));

  } catch (error) {
    spinner.fail(chalk.red('❌ Evaluation failed'));
    console.error(chalk.red('Error:'), error instanceof Error ? error.message : error);
    process.exit(1);
  }
}

async function loadEvaluationSuite(suitePath: string): Promise<EvaluationSuite> {
  const { readFile } = await import('fs/promises');
  
  try {
    const content = await readFile(suitePath, 'utf-8');
    return JSON.parse(content);
  } catch (error) {
    if ((error as any).code === 'ENOENT') {
      // Create a sample evaluation suite
      const sampleSuite: EvaluationSuite = {
        name: 'Sample Code Discussion Suite',
        description: 'Evaluation suite for code-heavy conversations',
        sessionId: 'sample-session',
        queries: [
          {
            id: 'q1',
            query: 'How do I implement JWT authentication?',
            relevantChunks: ['chunk-auth-1', 'chunk-auth-2', 'chunk-jwt-impl'],
            tags: ['authentication', 'security']
          },
          {
            id: 'q2', 
            query: 'What are the best practices for error handling?',
            relevantChunks: ['chunk-error-1', 'chunk-error-patterns'],
            tags: ['error-handling', 'best-practices']
          },
          {
            id: 'q3',
            query: 'How to optimize database queries for performance?',
            relevantChunks: ['chunk-db-1', 'chunk-perf-1', 'chunk-indexes'],
            tags: ['database', 'performance']
          }
        ]
      };
      
      await writeFile(suitePath, JSON.stringify(sampleSuite, null, 2));
      console.log(chalk.yellow(`📝 Created sample evaluation suite at: ${suitePath}`));
      
      return sampleSuite;
    }
    throw error;
  }
}

async function tuneParameters(
  db: any,
  embeddings: any,
  reranker: any,
  suite: EvaluationSuite,
  iterations: number
): Promise<{ config: any; score: number }> {
  let bestConfig: any = null;
  let bestScore = 0;
  
  const baseConfig = db.getConfig('system') || ContextOrchestrator.getDefaultConfig();
  
  // Grid search parameters
  const alphaValues = [0.5, 0.7, 1.0, 1.2, 1.5];
  const betaValues = [0.3, 0.5, 0.8, 1.0, 1.2];
  
  console.log(chalk.blue(`🎯 Grid searching ${alphaValues.length} × ${betaValues.length} parameter combinations`));
  
  let evaluationCount = 0;
  const totalEvaluations = Math.min(iterations, alphaValues.length * betaValues.length);
  
  for (const alpha of alphaValues) {
    if (evaluationCount >= iterations) break;
    
    for (const beta of betaValues) {
      if (evaluationCount >= iterations) break;
      
      console.log(chalk.gray(`   Testing α=${alpha}, β=${beta} (${evaluationCount + 1}/${totalEvaluations})`));
      
      const testConfig = {
        ...baseConfig,
        retrieval: {
          ...baseConfig.retrieval,
          alpha,
          beta
        }
      };
      
      const orchestrator = new ContextOrchestrator(db, embeddings, reranker, testConfig);
      
      // Quick evaluation on subset of queries
      const testQueries = suite.queries.slice(0, Math.min(3, suite.queries.length));
      const results: EvaluationResult[] = [];
      
      for (const evalQuery of testQueries) {
        try {
          const startTime = Date.now();
          const result = await orchestrator.orchestrateTurn(suite.sessionId, evalQuery.query);
          const latency = Date.now() - startTime;
          
          const retrievedChunks = result.pack.citations ? 
            Object.values(result.pack.citations).map(c => c.messageId).slice(0, 10) : 
            [];
          
          const metrics = calculateMetrics(retrievedChunks, evalQuery.relevantChunks, latency);
          
          results.push({
            queryId: evalQuery.id,
            query: evalQuery.query,
            metrics,
            retrievedChunks,
            relevantChunks: evalQuery.relevantChunks
          });
        } catch (error) {
          // Skip failed queries
          continue;
        }
      }
      
      // Calculate mean nDCG@10 for this configuration
      const meanNdcg10 = results.length > 0 ? 
        results.reduce((sum, r) => sum + r.metrics.ndcg_10, 0) / results.length : 
        0;
      
      if (meanNdcg10 > bestScore) {
        bestScore = meanNdcg10;
        bestConfig = testConfig;
        console.log(chalk.green(`   ✨ New best: nDCG@10 = ${meanNdcg10.toFixed(4)}`));
      }
      
      evaluationCount++;
    }
  }
  
  return { config: bestConfig || baseConfig, score: bestScore };
}

function calculateMetrics(
  retrieved: string[], 
  relevant: string[], 
  latency: number
): EvaluationResult['metrics'] {
  const relevantSet = new Set(relevant);
  
  // Calculate nDCG@k
  const ndcg5 = calculateNDCG(retrieved.slice(0, 5), relevantSet);
  const ndcg10 = calculateNDCG(retrieved.slice(0, 10), relevantSet);
  
  // Calculate Recall@k
  const recall5 = calculateRecall(retrieved.slice(0, 5), relevantSet);
  const recall10 = calculateRecall(retrieved.slice(0, 10), relevantSet);
  
  return {
    ndcg_5: ndcg5,
    ndcg_10: ndcg10,
    recall_5: recall5,
    recall_10: recall10,
    latency
  };
}

function calculateNDCG(retrieved: string[], relevantSet: Set<string>): number {
  if (retrieved.length === 0) return 0;
  
  // Calculate DCG
  let dcg = 0;
  for (let i = 0; i < retrieved.length; i++) {
    const relevance = relevantSet.has(retrieved[i]) ? 1 : 0;
    dcg += relevance / Math.log2(i + 2); // i+2 because log2(1) = 0
  }
  
  // Calculate IDCG (perfect ordering)
  const numRelevant = Math.min(retrieved.length, relevantSet.size);
  let idcg = 0;
  for (let i = 0; i < numRelevant; i++) {
    idcg += 1 / Math.log2(i + 2);
  }
  
  return idcg > 0 ? dcg / idcg : 0;
}

function calculateRecall(retrieved: string[], relevantSet: Set<string>): number {
  if (relevantSet.size === 0) return 0;
  
  let correctlyRetrieved = 0;
  for (const item of retrieved) {
    if (relevantSet.has(item)) {
      correctlyRetrieved++;
    }
  }
  
  return correctlyRetrieved / relevantSet.size;
}

function calculateAggregatedMetrics(results: EvaluationResult[]): EvaluationReport['aggregated'] {
  if (results.length === 0) {
    return {
      mean_ndcg_5: 0,
      mean_ndcg_10: 0,
      mean_recall_5: 0,
      mean_recall_10: 0,
      mean_latency: 0,
      p50_latency: 0,
      p90_latency: 0
    };
  }
  
  const latencies = results.map(r => r.metrics.latency).sort((a, b) => a - b);
  const p50Index = Math.floor(latencies.length * 0.5);
  const p90Index = Math.floor(latencies.length * 0.9);
  
  return {
    mean_ndcg_5: results.reduce((sum, r) => sum + r.metrics.ndcg_5, 0) / results.length,
    mean_ndcg_10: results.reduce((sum, r) => sum + r.metrics.ndcg_10, 0) / results.length,
    mean_recall_5: results.reduce((sum, r) => sum + r.metrics.recall_5, 0) / results.length,
    mean_recall_10: results.reduce((sum, r) => sum + r.metrics.recall_10, 0) / results.length,
    mean_latency: results.reduce((sum, r) => sum + r.metrics.latency, 0) / results.length,
    p50_latency: latencies[p50Index] || 0,
    p90_latency: latencies[p90Index] || 0
  };
}

function displayResults(report: EvaluationReport): void {
  console.log();
  console.log(chalk.blue('📊 Evaluation Results'));
  console.log(chalk.blue('═══════════════════'));
  
  console.log();
  console.log(chalk.white(`Suite: ${report.suite}`));
  console.log(chalk.white(`Session: ${report.sessionId}`));
  console.log(chalk.white(`Queries: ${report.results.length}`));
  
  console.log();
  console.log(chalk.blue('🎯 Aggregated Metrics'));
  console.log(chalk.blue('─────────────────────'));
  console.log(`${chalk.cyan('nDCG@5:')}  ${report.aggregated.mean_ndcg_5.toFixed(4)}`);
  console.log(`${chalk.cyan('nDCG@10:')} ${report.aggregated.mean_ndcg_10.toFixed(4)}`);
  console.log(`${chalk.cyan('Recall@5:')}  ${report.aggregated.mean_recall_5.toFixed(4)}`);
  console.log(`${chalk.cyan('Recall@10:')} ${report.aggregated.mean_recall_10.toFixed(4)}`);
  
  console.log();
  console.log(chalk.blue('⚡ Latency Metrics'));
  console.log(chalk.blue('─────────────────'));
  console.log(`${chalk.cyan('Mean:')} ${report.aggregated.mean_latency.toFixed(0)}ms`);
  console.log(`${chalk.cyan('P50:')}  ${report.aggregated.p50_latency.toFixed(0)}ms`);
  console.log(`${chalk.cyan('P90:')}  ${report.aggregated.p90_latency.toFixed(0)}ms`);
  
  // Show top 3 best and worst performing queries
  const sortedResults = [...report.results].sort((a, b) => b.metrics.ndcg_10 - a.metrics.ndcg_10);
  
  console.log();
  console.log(chalk.green('🏆 Top Performing Queries'));
  console.log(chalk.green('─────────────────────────'));
  for (let i = 0; i < Math.min(3, sortedResults.length); i++) {
    const result = sortedResults[i];
    console.log(`${i + 1}. nDCG@10=${result.metrics.ndcg_10.toFixed(3)} | ${result.query.substring(0, 60)}...`);
  }
  
  console.log();
  console.log(chalk.red('📉 Lowest Performing Queries'));
  console.log(chalk.red('─────────────────────────────'));
  for (let i = Math.max(0, sortedResults.length - 3); i < sortedResults.length; i++) {
    const result = sortedResults[i];
    const rank = sortedResults.length - i;
    console.log(`${rank}. nDCG@10=${result.metrics.ndcg_10.toFixed(3)} | ${result.query.substring(0, 60)}...`);
  }
}

function generateCSV(report: EvaluationReport): string {
  const rows = [
    'query_id,query,ndcg_5,ndcg_10,recall_5,recall_10,latency_ms'
  ];
  
  for (const result of report.results) {
    const row = [
      result.queryId,
      `"${result.query.replace(/"/g, '""')}"`,
      result.metrics.ndcg_5.toFixed(4),
      result.metrics.ndcg_10.toFixed(4),
      result.metrics.recall_5.toFixed(4),
      result.metrics.recall_10.toFixed(4),
      result.metrics.latency.toString()
    ].join(',');
    
    rows.push(row);
  }
  
  // Add summary row
  rows.push('');
  rows.push('SUMMARY,mean_values,,,,,');
  rows.push([
    'aggregate',
    'mean',
    report.aggregated.mean_ndcg_5.toFixed(4),
    report.aggregated.mean_ndcg_10.toFixed(4),
    report.aggregated.mean_recall_5.toFixed(4),
    report.aggregated.mean_recall_10.toFixed(4),
    report.aggregated.mean_latency.toFixed(0)
  ].join(','));
  
  return rows.join('\n');
}