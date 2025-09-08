import { promises as fs } from 'fs';
import { join } from 'path';
import chalk from 'chalk';
import ora from 'ora';
import { openDb, CtxDatabase } from '@ctx-run/sqlite';
import { ContextOrchestrator } from '@ctx-run/core';

export interface InitOptions {
  path?: string;
  force?: boolean;
}

export async function initCommand(options: InitOptions): Promise<void> {
  const targetPath = options.path || process.cwd();
  const ctxDir = join(targetPath, '.ctx');
  const dbPath = join(ctxDir, 'ctx.db');
  const lockPath = join(ctxDir, 'lock.json');

  console.log(chalk.blue('🚀 Initializing ctx-run workspace...'));

  // Check if already initialized
  try {
    await fs.access(ctxDir);
    if (!options.force) {
      console.log(chalk.yellow('⚠️  Context workspace already exists. Use --force to reinitialize.'));
      return;
    }
    console.log(chalk.yellow('🔄 Reinitializing existing workspace...'));
  } catch {
    // Directory doesn't exist, proceed
  }

  const spinner = ora('Creating workspace structure...').start();

  try {
    // Create .ctx directory
    await fs.mkdir(ctxDir, { recursive: true });

    // Create subdirectories
    await fs.mkdir(join(ctxDir, 'sessions'), { recursive: true });
    await fs.mkdir(join(ctxDir, 'checkpoints'), { recursive: true });
    await fs.mkdir(join(ctxDir, 'cache'), { recursive: true });

    spinner.text = 'Initializing database...';

    // Initialize database
    const db = await openDb(dbPath);
    
    spinner.text = 'Setting up default configuration...';

    // Store default configuration
    const defaultConfig = ContextOrchestrator.getDefaultConfig();
    db.setConfig('system', defaultConfig);

    // Create lock file with metadata
    const lockData = {
      version: '1.0.0',
      created: new Date().toISOString(),
      dbPath: 'ctx.db',
      vectorExtension: await db.loadVectorExtension(),
      config: defaultConfig
    };

    await fs.writeFile(lockPath, JSON.stringify(lockData, null, 2));

    // Create .gitignore
    const gitignorePath = join(ctxDir, '.gitignore');
    const gitignoreContent = `# ctx-run cache files
cache/
*.tmp
*.log

# Database files (optional - you might want to version control these)
# *.db
# *.db-wal
# *.db-shm
`;
    await fs.writeFile(gitignorePath, gitignoreContent);

    // Create README
    const readmePath = join(ctxDir, 'README.md');
    const readmeContent = `# ctx-run Workspace

This directory contains your ctx-run context management workspace.

## Structure

- \`ctx.db\` - SQLite database with your conversation history and indexes
- \`sessions/\` - Session-specific data and exports
- \`checkpoints/\` - Saved checkpoints and snapshots
- \`cache/\` - Temporary files and model cache
- \`lock.json\` - Workspace metadata and configuration

## Usage

\`\`\`bash
# Ingest conversation data
npx ctx-run ingest --session my-session --from conversation.json

# Build search indexes
npx ctx-run index --session my-session

# Query conversation history
npx ctx-run query "How do I set up authentication?" --session my-session

# Start development server
npx ctx-run serve
\`\`\`

## Configuration

Edit \`lock.json\` to modify search parameters, model settings, and other options.
`;
    await fs.writeFile(readmePath, readmeContent);

    db.close();
    
    spinner.succeed(chalk.green('✅ Workspace initialized successfully!'));
    
    console.log();
    console.log(chalk.blue('📁 Created workspace structure:'));
    console.log(chalk.gray(`   ${ctxDir}/`));
    console.log(chalk.gray(`   ├── ctx.db`));
    console.log(chalk.gray(`   ├── sessions/`));
    console.log(chalk.gray(`   ├── checkpoints/`));
    console.log(chalk.gray(`   ├── cache/`));
    console.log(chalk.gray(`   ├── lock.json`));
    console.log(chalk.gray(`   └── README.md`));
    
    console.log();
    console.log(chalk.blue('🔧 Vector extension:'), 
      lockData.vectorExtension ? 
        chalk.green('Loaded') : 
        chalk.yellow('Using WASM fallback')
    );
    
    console.log();
    console.log(chalk.blue('🚀 Next steps:'));
    console.log(chalk.gray('   1. Import conversation data:'));
    console.log(chalk.blue('      npx ctx-run ingest --session <name> --from <file>'));
    console.log(chalk.gray('   2. Build search indexes:'));
    console.log(chalk.blue('      npx ctx-run index --session <name>'));
    console.log(chalk.gray('   3. Query your conversations:'));
    console.log(chalk.blue('      npx ctx-run query "your question" --session <name>'));

  } catch (error) {
    spinner.fail(chalk.red('❌ Initialization failed'));
    console.error(chalk.red('Error:'), error instanceof Error ? error.message : error);
    process.exit(1);
  }
}