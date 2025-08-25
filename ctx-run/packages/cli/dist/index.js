#!/usr/bin/env node
"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const yargs_1 = __importDefault(require("yargs"));
const helpers_1 = require("yargs/helpers");
const sqlite_1 = require("@lethe/sqlite");
const core_1 = require("@lethe/core");
const path_1 = require("path");
const fs_1 = require("fs");
(0, yargs_1.default)((0, helpers_1.hideBin)(process.argv))
    .command('init <path>', 'Initialize the Lethe context', (yargs) => {
    return yargs
        .positional('path', {
        describe: 'Path to initialize the context in',
        type: 'string',
        demandOption: true,
    });
}, async (argv) => {
    const ctxPath = (0, path_1.join)(argv.path, '.ctx');
    (0, fs_1.mkdirSync)(ctxPath, { recursive: true });
    const dbPath = (0, path_1.join)(ctxPath, 'lethe.db');
    const configPath = (0, path_1.join)(argv.path, 'ctx.config.json');
    console.log(`Initializing Lethe context at ${argv.path}`);
    const db = (0, sqlite_1.openDb)(dbPath);
    await (0, sqlite_1.migrate)(db);
    console.log('Database migrated.');
    // Probe vector extension
    const hasVectorExtension = await (0, sqlite_1.loadVectorExtension)(db);
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
        config_path: configPath
    };
    (0, fs_1.writeFileSync)((0, path_1.join)(ctxPath, 'lock.json'), JSON.stringify(lockInfo, null, 2));
    console.log('Lock file created with environment status.');
    // Create external configuration file (replaces database config)
    if (!(0, fs_1.existsSync)(configPath)) {
        (0, core_1.createDefaultConfigFile)(configPath);
        console.log('✅ External configuration created - edit ctx.config.json to customize settings');
    }
    else {
        console.log('📋 Using existing ctx.config.json configuration');
    }
    // Load and validate configuration
    try {
        const config = (0, core_1.loadConfig)(argv.path);
        console.log('✅ Configuration loaded and validated');
        // Keep backward compatibility by also storing in database for legacy code
        (0, sqlite_1.upsertConfig)(db, 'retrieval', config.retrieval);
        (0, sqlite_1.upsertConfig)(db, 'chunking', config.chunking);
        (0, sqlite_1.upsertConfig)(db, 'timeouts', config.timeouts);
        console.log('📊 Configuration synchronized to database for compatibility');
    }
    catch (error) {
        console.error('❌ Configuration validation failed:', error);
        process.exit(1);
    }
    console.log('🎉 Context initialization complete.');
    console.log('📝 Edit ctx.config.json to customize retrieval, chunking, and other settings.');
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
    const dbPath = (0, path_1.join)(process.cwd(), '.ctx', 'lethe.db');
    const db = (0, sqlite_1.openDb)(dbPath);
    let messages;
    if (argv.from === 'stdio') {
        // Read from stdin
        const input = await new Promise((resolve, reject) => {
            let data = '';
            process.stdin.on('data', chunk => data += chunk);
            process.stdin.on('end', () => resolve(data));
            process.stdin.on('error', reject);
        });
        messages = JSON.parse(input);
    }
    else {
        // Read from file
        messages = JSON.parse((0, fs_1.readFileSync)(argv.from, 'utf-8'));
    }
    console.log(`Ingesting ${messages.length} messages for session ${argv.session}`);
    await (0, core_1.upsertMessagesWithDb)(db, argv.session, messages);
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
    const dbPath = (0, path_1.join)(process.cwd(), '.ctx', 'lethe.db');
    const db = (0, sqlite_1.openDb)(dbPath);
    console.log(`Ensuring embeddings for session ${argv.session}`);
    await (0, core_1.ensureEmbeddingsWithDb)(db, argv.session);
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
    const dbPath = (0, path_1.join)(process.cwd(), '.ctx', 'lethe.db');
    const db = (0, sqlite_1.openDb)(dbPath);
    try {
        console.log(`🚀 Enhanced retrieval: "${argv.text}" for session: ${argv.session}`);
        // Initialize embeddings provider
        const { getProvider } = await Promise.resolve().then(() => __importStar(require('@lethe/embeddings')));
        const embeddings = await getProvider();
        // Run enhanced pipeline
        const { enhancedQuery } = await Promise.resolve().then(() => __importStar(require('@lethe/core/dist/pipeline/index.js')));
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
    }
    catch (error) {
        console.error(`Enhanced retrieval failed: ${error}`);
        console.error(`Stack trace:`, error?.stack);
        process.exit(1);
    }
    finally {
        db.close();
    }
})
    .command('diagnose', 'Run system diagnostics', (yargs) => {
    return yargs;
}, async (argv) => {
    const dbPath = (0, path_1.join)(process.cwd(), '.ctx', 'lethe.db');
    const db = (0, sqlite_1.openDb)(dbPath);
    try {
        console.log('🔍 Running Lethe diagnostics...\n');
        // Test enhanced pipeline
        const { diagnoseEnhancedPipeline } = await Promise.resolve().then(() => __importStar(require('@lethe/core/dist/pipeline/index.js')));
        const diagnosis = await diagnoseEnhancedPipeline(db);
        // Ollama status
        console.log('📡 Ollama Integration:');
        console.log(`  Available: ${diagnosis.ollama.available ? '✅' : '❌'}`);
        if (diagnosis.ollama.models.length > 0) {
            console.log(`  Models: ${diagnosis.ollama.models.slice(0, 3).join(', ')}${diagnosis.ollama.models.length > 3 ? '...' : ''} (${diagnosis.ollama.models.length} total)`);
        }
        else {
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
            const { testHyde } = await Promise.resolve().then(() => __importStar(require('@lethe/core/dist/hyde/index.js')));
            const hydeTest = await testHyde(db, 'async error handling');
            console.log(`  HyDE: ${hydeTest.success ? '✅' : '❌'} (${hydeTest.duration}ms)`);
            if (hydeTest.result) {
                console.log(`    Generated: ${hydeTest.result.queries.length} queries`);
            }
        }
        catch (error) {
            console.log(`  HyDE: ❌ ${error?.message || error}`);
        }
        // Test Summarization
        try {
            const { testSummarization } = await Promise.resolve().then(() => __importStar(require('@lethe/core/dist/summarize/index.js')));
            const sumTest = await testSummarization(db, 'async error handling');
            console.log(`  Summarization: ${sumTest.success ? '✅' : '❌'} (${sumTest.duration}ms)`);
            if (sumTest.result) {
                console.log(`    Extracted: ${sumTest.result.key_entities.length} entities, ${sumTest.result.claims.length} claims`);
            }
        }
        catch (error) {
            console.log(`  Summarization: ❌ ${error?.message || error}`);
        }
        console.log('\n✅ Diagnostics complete');
    }
    catch (error) {
        console.error(`Diagnostics failed: ${error}`);
        process.exit(1);
    }
    finally {
        db.close();
    }
})
    .command('state', 'Manage session state', (yargs) => {
    return yargs
        .command('list', 'List all sessions with state', {}, async () => {
        const dbPath = (0, path_1.join)(process.cwd(), '.ctx', 'lethe.db');
        const db = (0, sqlite_1.openDb)(dbPath);
        try {
            const { getStateManager } = await Promise.resolve().then(() => __importStar(require('@lethe/core/dist/state/index.js')));
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
        }
        finally {
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
        const dbPath = (0, path_1.join)(process.cwd(), '.ctx', 'lethe.db');
        const db = (0, sqlite_1.openDb)(dbPath);
        try {
            const { getStateManager } = await Promise.resolve().then(() => __importStar(require('@lethe/core/dist/state/index.js')));
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
        }
        finally {
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
        const dbPath = (0, path_1.join)(process.cwd(), '.ctx', 'lethe.db');
        const db = (0, sqlite_1.openDb)(dbPath);
        try {
            const { getStateManager } = await Promise.resolve().then(() => __importStar(require('@lethe/core/dist/state/index.js')));
            const stateManager = getStateManager(db);
            stateManager.clearSessionState(argv['session-id']);
            console.log(`Cleared state for session: ${argv['session-id']}`);
        }
        finally {
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
//# sourceMappingURL=index.js.map