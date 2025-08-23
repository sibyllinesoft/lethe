#!/usr/bin/env node

import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { openDb, migrate, loadVectorExtension, Message, upsertConfig } from '@lethe/sqlite';
import { upsertMessagesWithDb, ensureEmbeddingsWithDb } from '@lethe/core';
import { join } from 'path';
import { mkdirSync, readFileSync, writeFileSync } from 'fs';

yargs(hideBin(process.argv))
  .command('init <path>', 'Initialize the Lethe context', (yargs) => {
    return yargs
      .positional('path', {
        describe: 'Path to initialize the context in',
        type: 'string',
        demandOption: true,
      });
  }, async (argv) => {
    const ctxPath = join(argv.path, '.ctx');
    mkdirSync(ctxPath, { recursive: true });
    const dbPath = join(ctxPath, 'lethe.db');
    
    console.log(`Initializing Lethe context at ${argv.path}`);
    
    const db = openDb(dbPath);
    await migrate(db);
    console.log('Database migrated.');
    
    // Probe vector extension
    const hasVectorExtension = await loadVectorExtension(db);
    const vectorBackend = hasVectorExtension ? 'native' : 'wasm';
    
    console.log(`Vector backend: ${vectorBackend}`);
    
    // Create lock.json with environment + extension status
    const lockInfo = {
      created_at: new Date().toISOString(),
      node_version: process.version,
      platform: process.platform,
      arch: process.arch,
      vector_backend: vectorBackend,
      database_path: dbPath,
    };
    
    writeFileSync(join(ctxPath, 'lock.json'), JSON.stringify(lockInfo, null, 2));
    console.log('Lock file created with environment status.');
    
    // Initialize default config
    upsertConfig(db, 'retrieval', {
      alpha: 0.7,
      beta: 0.5,
      gamma_kind_boost: { code: 0.1, text: 0.0 },
    });
    
    upsertConfig(db, 'chunking', {
      target_tokens: 320,
      overlap: 64,
    });
    
    upsertConfig(db, 'timeouts', {
      hyde_ms: 10000,
      summarize_ms: 10000,
      ollama_connect_ms: 500,
    });
    
    console.log('Context initialization complete.');
    db.close();
  })
  .command('ingest', 'Ingest messages from a file or stdin', (yargs) => {
    return yargs
      .option('session', {
        describe: 'Session ID',
        type: 'string',
        demandOption: true,
      })
      .option('from', {
        describe: 'Source: file path or "stdio"',
        type: 'string',
        demandOption: true,
      });
  }, async (argv) => {
    const dbPath = join(process.cwd(), '.ctx', 'lethe.db');
    const db = openDb(dbPath);
    
    let messages: Message[];
    
    if (argv.from === 'stdio') {
      // Read from stdin
      const input = await new Promise<string>((resolve, reject) => {
        let data = '';
        process.stdin.on('data', chunk => data += chunk);
        process.stdin.on('end', () => resolve(data));
        process.stdin.on('error', reject);
      });
      messages = JSON.parse(input);
    } else {
      // Read from file
      messages = JSON.parse(readFileSync(argv.from, 'utf-8'));
    }
    
    console.log(`Ingesting ${messages.length} messages for session ${argv.session}`);
    
    await upsertMessagesWithDb(db, argv.session, messages);
    
    console.log(`Ingestion complete. DF/IDF rebuilt for session.`);
    db.close();
  })
  .command('index', 'Ensure embeddings and vector index for new chunks', (yargs) => {
    return yargs
      .option('session', {
        describe: 'Session ID',
        type: 'string',
        demandOption: true,
      });
  }, async (argv) => {
    const dbPath = join(process.cwd(), '.ctx', 'lethe.db');
    const db = openDb(dbPath);
    
    console.log(`Ensuring embeddings for session ${argv.session}`);
    await ensureEmbeddingsWithDb(db, argv.session);
    
    console.log('Indexing complete.');
    db.close();
  })
  .command('query <text>', 'Run enhanced retrieval pipeline with HyDE and summarization', (yargs) => {
    return yargs
      .positional('text', {
        describe: 'Query text',
        type: 'string',
        demandOption: true,
      })
      .option('session', {
        describe: 'Session ID',
        type: 'string',
        demandOption: true,
      })
      .option('debug', {
        describe: 'Show debug information',
        type: 'boolean',
        default: false,
      })
      .option('no-hyde', {
        describe: 'Disable HyDE query expansion',
        type: 'boolean',
        default: false,
      })
      .option('no-summarize', {
        describe: 'Disable AI summarization',
        type: 'boolean',
        default: false,
      })
      .option('no-planning', {
        describe: 'Disable adaptive plan selection',
        type: 'boolean',
        default: false,
      });
  }, async (argv) => {
    const dbPath = join(process.cwd(), '.ctx', 'lethe.db');
    const db = openDb(dbPath);
    
    try {
      console.log(`🚀 Enhanced retrieval: "${argv.text}" for session: ${argv.session}`);
      
      // Initialize embeddings provider
      const { getProvider } = await import('@lethe/embeddings');
      const embeddings = await getProvider();
      
      // Run enhanced pipeline
      const { enhancedQuery } = await import('@lethe/core/dist/pipeline/index.js');
      const result = await enhancedQuery(argv.text, {
        db,
        embeddings,
        sessionId: argv.session,
        enableHyde: !argv['no-hyde'],
        enableSummarization: !argv['no-summarize'],
        enablePlanSelection: !argv['no-planning']
      });
      
      if (argv.debug) {
        console.log(`\n=== DEBUG INFO ===`);
        console.log(`Plan: ${result.plan.plan} (${result.plan.reasoning})`);
        console.log(`Parameters:`, result.plan.parameters);
        if (result.hydeQueries) {
          console.log(`HyDE queries: ${result.hydeQueries.join(' | ')}`);
        }
        console.log(`Timing: ${result.duration.total}ms total (retrieval: ${result.duration.retrieval}ms${result.duration.hyde ? `, hyde: ${result.duration.hyde}ms` : ''}${result.duration.summarization ? `, summarization: ${result.duration.summarization}ms` : ''})`);
        console.log(`Found ${result.debug.retrievalCandidates} candidates`);
        console.log(`Entities: ${result.pack.key_entities.slice(0, 5).join(', ')}${result.pack.key_entities.length > 5 ? '...' : ''}`);
        console.log(`Claims: ${result.pack.claims.length}, Contradictions: ${result.pack.contradictions.length}`);
        console.log(`===================\n`);
      }
      
      console.log(JSON.stringify(result.pack, null, 2));
      
    } catch (error: any) {
      console.error(`Enhanced retrieval failed: ${error}`);
      console.error(`Stack trace:`, error?.stack);
      process.exit(1);
    } finally {
      db.close();
    }
  })
  .command('diagnose', 'Run system diagnostics', (yargs) => {
    return yargs;
  }, async (argv) => {
    const dbPath = join(process.cwd(), '.ctx', 'lethe.db');
    const db = openDb(dbPath);
    
    try {
      console.log('🔍 Running Lethe diagnostics...\n');
      
      // Test enhanced pipeline
      const { diagnoseEnhancedPipeline } = await import('@lethe/core/dist/pipeline/index.js');
      const diagnosis = await diagnoseEnhancedPipeline(db);
      
      // Ollama status
      console.log('📡 Ollama Integration:');
      console.log(`  Available: ${diagnosis.ollama.available ? '✅' : '❌'}`);
      if (diagnosis.ollama.models.length > 0) {
        console.log(`  Models: ${diagnosis.ollama.models.slice(0, 3).join(', ')}${diagnosis.ollama.models.length > 3 ? '...' : ''} (${diagnosis.ollama.models.length} total)`);
      } else {
        console.log('  Models: None available');
      }
      
      // State management
      console.log('\n💾 State Management:');
      console.log(`  Sessions tracked: ${diagnosis.state.sessions.length}`);
      if (diagnosis.state.sessions.length > 0) {
        console.log(`  Recent sessions: ${diagnosis.state.sessions.slice(0, 3).join(', ')}`);
      }
      
      // Configuration
      console.log('\n⚙️  Configuration:');
      console.log(`  Retrieval α=${diagnosis.config.retrieval?.alpha}, β=${diagnosis.config.retrieval?.beta}`);
      console.log(`  Timeouts: HyDE ${diagnosis.config.timeouts?.hyde_ms}ms, Summarize ${diagnosis.config.timeouts?.summarize_ms}ms`);
      
      // Test individual components
      console.log('\n🧪 Component Tests:');
      
      // Test HyDE
      try {
        const { testHyde } = await import('@lethe/core/dist/hyde/index.js');
        const hydeTest = await testHyde(db, 'async error handling');
        console.log(`  HyDE: ${hydeTest.success ? '✅' : '❌'} (${hydeTest.duration}ms)`);
        if (hydeTest.result) {
          console.log(`    Generated: ${hydeTest.result.queries.length} queries`);
        }
      } catch (error: any) {
        console.log(`  HyDE: ❌ ${error?.message || error}`);
      }
      
      // Test Summarization
      try {
        const { testSummarization } = await import('@lethe/core/dist/summarize/index.js');
        const sumTest = await testSummarization(db, 'async error handling');
        console.log(`  Summarization: ${sumTest.success ? '✅' : '❌'} (${sumTest.duration}ms)`);
        if (sumTest.result) {
          console.log(`    Extracted: ${sumTest.result.key_entities.length} entities, ${sumTest.result.claims.length} claims`);
        }
      } catch (error: any) {
        console.log(`  Summarization: ❌ ${error?.message || error}`);
      }
      
      console.log('\n✅ Diagnostics complete');
      
    } catch (error: any) {
      console.error(`Diagnostics failed: ${error}`);
      process.exit(1);
    } finally {
      db.close();
    }
  })
  .command('state', 'Manage session state', (yargs) => {
    return yargs
      .command('list', 'List all sessions with state', {}, async () => {
        const dbPath = join(process.cwd(), '.ctx', 'lethe.db');
        const db = openDb(dbPath);
        
        try {
          const { getStateManager } = await import('@lethe/core/dist/state/index.js');
          const stateManager = getStateManager(db);
          const sessions = stateManager.getAllSessions();
          
          if (sessions.length === 0) {
            console.log('No sessions with state found');
            return;
          }
          
          console.log(`Found ${sessions.length} sessions with state:`);
          sessions.forEach((sessionId, i) => {
            const context = stateManager.getRecentContext(sessionId);
            console.log(`${i + 1}. ${sessionId} (${context.entityCount} entities)`);
            if (context.lastPackContradictions.length > 0) {
              console.log(`   ⚠️  ${context.lastPackContradictions.length} contradictions in last pack`);
            }
          });
          
        } finally {
          db.close();
        }
      })
      .command('show <session-id>', 'Show state for a session', (yargs) => {
        return yargs.positional('session-id', {
          describe: 'Session ID to show state for',
          type: 'string',
          demandOption: true,
        });
      }, async (argv) => {
        const dbPath = join(process.cwd(), '.ctx', 'lethe.db');
        const db = openDb(dbPath);
        
        try {
          const { getStateManager } = await import('@lethe/core/dist/state/index.js');
          const stateManager = getStateManager(db);
          const state = stateManager.getSessionState(argv['session-id']);
          const context = stateManager.getRecentContext(argv['session-id']);
          
          console.log(`State for session: ${argv['session-id']}`);
          console.log(`Updated: ${state.updatedAt}`);
          console.log(`Recent entities (${context.entityCount} total):`);
          context.recentEntities.forEach((entity, i) => {
            console.log(`  ${i + 1}. ${entity}`);
          });
          
          if (state.lastPackClaims.length > 0) {
            console.log(`\nLast pack claims:`);
            state.lastPackClaims.forEach((claim, i) => {
              console.log(`  ${i + 1}. ${claim}`);
            });
          }
          
          if (state.lastPackContradictions.length > 0) {
            console.log(`\nLast pack contradictions:`);
            state.lastPackContradictions.forEach((contradiction, i) => {
              console.log(`  ${i + 1}. ${contradiction}`);
            });
          }
          
        } finally {
          db.close();
        }
      })
      .command('clear <session-id>', 'Clear state for a session', (yargs) => {
        return yargs.positional('session-id', {
          describe: 'Session ID to clear state for',
          type: 'string',
          demandOption: true,
        });
      }, async (argv) => {
        const dbPath = join(process.cwd(), '.ctx', 'lethe.db');
        const db = openDb(dbPath);
        
        try {
          const { getStateManager } = await import('@lethe/core/dist/state/index.js');
          const stateManager = getStateManager(db);
          stateManager.clearSessionState(argv['session-id']);
          console.log(`Cleared state for session: ${argv['session-id']}`);
          
        } finally {
          db.close();
        }
      });
  })
  .command('serve', 'Start development server', (yargs) => {
    return yargs
      .option('port', {
        describe: 'Port to serve on',
        type: 'number',
        default: 7071,
      });
  }, async (argv) => {
    console.log(`Dev server will be implemented in M5`);
    console.log(`Would start server on port ${argv.port}`);
  })
  .command('checkpoint', 'Create a snapshot checkpoint', (yargs) => {
    return yargs
      .option('session', {
        describe: 'Session ID',
        type: 'string',
        demandOption: true,
      })
      .option('label', {
        describe: 'Checkpoint label',
        type: 'string',
        demandOption: true,
      });
  }, async (argv) => {
    console.log(`Checkpoint creation will be implemented in M5+`);
    console.log(`Would create checkpoint "${argv.label}" for session ${argv.session}`);
  })
  .command('eval', 'Run evaluation harness', (yargs) => {
    return yargs
      .option('suite', {
        describe: 'Path to evaluation suite',
        type: 'string',
        demandOption: true,
      });
  }, async (argv) => {
    console.log(`Evaluation harness will be implemented in M6`);
    console.log(`Would run evaluation suite: ${argv.suite}`);
  })
  .demandCommand(1)
  .help()
  .parse();
