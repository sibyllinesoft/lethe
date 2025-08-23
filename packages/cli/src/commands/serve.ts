import { join } from 'path';
import { spawn } from 'child_process';
import chalk from 'chalk';
import ora from 'ora';

export interface ServeOptions {
  port?: number;
  host?: string;
  open?: boolean;
}

export async function serveCommand(options: ServeOptions): Promise<void> {
  const port = options.port || 7071;
  const host = options.host || 'localhost';
  const ctxDir = join(process.cwd(), '.ctx');

  console.log(chalk.blue('🚀 Starting ctx-run development server...'));

  const spinner = ora('Checking workspace...').start();

  try {
    // Check if workspace exists
    const { promises: fs } = await import('fs');
    await fs.access(join(ctxDir, 'ctx.db'));
    
    spinner.text = 'Starting server...';
    
    // Start the dev server process
    const serverProcess = spawn('node', [
      '-e',
      `
      const express = require('express');
      const cors = require('cors');
      const { join } = require('path');
      const { openDb } = require('@ctx-run/sqlite');
      const { ContextOrchestrator } = require('@ctx-run/core');
      const { getDefaultProvider } = require('@ctx-run/embeddings');
      const { CrossEncoderReranker } = require('@ctx-run/reranker');
      
      const app = express();
      app.use(cors());
      app.use(express.json());
      
      const PORT = ${port};
      const HOST = '${host}';
      const DB_PATH = '${join(ctxDir, 'ctx.db')}';
      
      let db, orchestrator;
      
      async function initialize() {
        try {
          db = await openDb(DB_PATH);
          const config = db.getConfig('system') || ContextOrchestrator.getDefaultConfig();
          const embeddings = await getDefaultProvider();
          const reranker = new CrossEncoderReranker();
          
          orchestrator = new ContextOrchestrator(db, embeddings, reranker, config);
          
          console.log('✅ Database and AI models loaded');
        } catch (error) {
          console.error('❌ Failed to initialize:', error.message);
          process.exit(1);
        }
      }
      
      // API Routes
      app.get('/api/health', (req, res) => {
        res.json({ status: 'ok', timestamp: new Date().toISOString() });
      });
      
      app.get('/api/sessions', async (req, res) => {
        try {
          const sessions = await orchestrator.getSessions();
          res.json(sessions);
        } catch (error) {
          res.status(500).json({ error: error.message });
        }
      });
      
      app.get('/api/sessions/:sessionId', async (req, res) => {
        try {
          const { sessionId } = req.params;
          const stats = await orchestrator.getSessionStats(sessionId);
          const messages = db.getMessages(sessionId);
          const packs = db.getPacks(sessionId);
          
          res.json({ sessionId, stats, messages, packs });
        } catch (error) {
          res.status(500).json({ error: error.message });
        }
      });
      
      app.get('/api/sessions/:sessionId/messages', (req, res) => {
        try {
          const { sessionId } = req.params;
          const messages = db.getMessages(sessionId);
          res.json(messages);
        } catch (error) {
          res.status(500).json({ error: error.message });
        }
      });
      
      app.get('/api/sessions/:sessionId/chunks', (req, res) => {
        try {
          const { sessionId } = req.params;
          const chunks = db.getChunks(sessionId);
          res.json(chunks);
        } catch (error) {
          res.status(500).json({ error: error.message });
        }
      });
      
      app.post('/api/query', async (req, res) => {
        try {
          const { query, sessionId } = req.body;
          
          if (!query || !sessionId) {
            return res.status(400).json({ error: 'Query and sessionId are required' });
          }
          
          const result = await orchestrator.orchestrateTurn(sessionId, query);
          res.json(result);
        } catch (error) {
          res.status(500).json({ error: error.message });
        }
      });
      
      app.get('/api/config', (req, res) => {
        try {
          const config = orchestrator.getConfig();
          res.json(config);
        } catch (error) {
          res.status(500).json({ error: error.message });
        }
      });
      
      app.put('/api/config', (req, res) => {
        try {
          const updates = req.body;
          orchestrator.updateConfig(updates);
          res.json({ success: true });
        } catch (error) {
          res.status(500).json({ error: error.message });
        }
      });
      
      app.post('/api/annotate', (req, res) => {
        try {
          const { sessionId, messageId, annotation } = req.body;
          
          if (!sessionId || !messageId || !annotation) {
            return res.status(400).json({ error: 'sessionId, messageId, and annotation are required' });
          }
          
          // Store annotation in database
          const annotationId = Date.now().toString();
          db.createAnnotation(annotationId, sessionId, messageId, annotation);
          
          res.json({ success: true, annotationId });
        } catch (error) {
          res.status(500).json({ error: error.message });
        }
      });
      
      app.post('/api/checkpoint', (req, res) => {
        try {
          const { sessionId, label, description } = req.body;
          
          if (!sessionId || !label) {
            return res.status(400).json({ error: 'sessionId and label are required' });
          }
          
          // Create checkpoint
          const checkpointId = 'cp-' + Date.now();
          const stats = orchestrator.getSessionStats ? orchestrator.getSessionStats(sessionId) : { messageCount: 0, chunkCount: 0 };
          
          const checkpoint = {
            id: checkpointId,
            sessionId,
            label,
            description: description || '',
            createdAt: Date.now(),
            messageCount: stats.messageCount || 0,
            chunkCount: stats.chunkCount || 0
          };
          
          db.createCheckpoint(checkpoint);
          
          res.json({ success: true, checkpoint });
        } catch (error) {
          res.status(500).json({ error: error.message });
        }
      });
      
      // Serve static files (dev server will proxy to this)
      app.use(express.static(join(__dirname, '../devserver/dist')));
      
      // Start server
      initialize().then(() => {
        app.listen(PORT, HOST, () => {
          console.log('🌐 Server running at http://' + HOST + ':' + PORT);
          console.log('📊 API endpoints available at /api/*');
          console.log('🎨 UI available at http://' + HOST + ':' + PORT);
        });
      });
      
      process.on('SIGINT', () => {
        console.log('\\n🛑 Shutting down server...');
        if (db) db.close();
        process.exit(0);
      });
      `
    ], {
      stdio: 'pipe',
      env: {
        ...process.env,
        NODE_PATH: join(__dirname, '../../../node_modules')
      }
    });
    
    // Handle server output
    serverProcess.stdout?.on('data', (data) => {
      const output = data.toString().trim();
      if (output.includes('Server running')) {
        spinner.succeed(chalk.green('✅ Server started successfully!'));
        console.log();
        console.log(chalk.blue('🌐 Development server:'));
        console.log(chalk.white(`   http://${host}:${port}`));
        console.log();
        console.log(chalk.blue('📊 API endpoints:'));
        console.log(chalk.gray(`   GET  /api/health`));
        console.log(chalk.gray(`   GET  /api/sessions`));
        console.log(chalk.gray(`   GET  /api/sessions/:id`));
        console.log(chalk.gray(`   POST /api/query`));
        console.log(chalk.gray(`   GET  /api/config`));
        console.log();
        console.log(chalk.blue('💡 Usage:'));
        console.log(chalk.white('   • Browse sessions and conversation history'));
        console.log(chalk.white('   • Test queries with real-time results'));
        console.log(chalk.white('   • Adjust configuration parameters'));
        console.log(chalk.white('   • View search performance metrics'));
        console.log();
        console.log(chalk.gray('   Press Ctrl+C to stop the server'));
        
        if (options.open) {
          const open = await import('open');
          open.default(`http://${host}:${port}`);
        }
      } else {
        console.log(data.toString());
      }
    });
    
    serverProcess.stderr?.on('data', (data) => {
      const error = data.toString().trim();
      if (error) {
        console.error(chalk.red(error));
      }
    });
    
    serverProcess.on('error', (error) => {
      spinner.fail(chalk.red('❌ Failed to start server'));
      console.error(chalk.red('Error:'), error.message);
      process.exit(1);
    });
    
    serverProcess.on('exit', (code) => {
      if (code !== 0) {
        console.log(chalk.yellow(`\n📴 Server stopped (code: ${code})`));
      }
    });
    
    // Handle process termination
    process.on('SIGINT', () => {
      console.log(chalk.yellow('\n🛑 Stopping server...'));
      serverProcess.kill('SIGINT');
      process.exit(0);
    });
    
  } catch (error) {
    spinner.fail(chalk.red('❌ Failed to start server'));
    
    if ((error as any).code === 'ENOENT') {
      console.error(chalk.red('Error: No ctx-run workspace found.'));
      console.log(chalk.blue('Run: npx ctx-run init'));
    } else {
      console.error(chalk.red('Error:'), error instanceof Error ? error.message : error);
    }
    
    process.exit(1);
  }
}