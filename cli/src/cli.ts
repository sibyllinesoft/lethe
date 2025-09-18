#!/usr/bin/env node

import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import chalk from 'chalk';

import { initCommand } from './commands/init.js';
import { ingestCommand } from './commands/ingest.js';
import { indexCommand } from './commands/index.js';
import { queryCommand } from './commands/query.js';
import { serveCommand } from './commands/serve.js';
import { evalCommand } from './commands/eval.js';

// CLI definition
yargs(hideBin(process.argv))
  .scriptName('ctx-run')
  .usage(chalk.blue('$0 <cmd> [args]'))
  .version('1.0.0')
  .help('help')
  .alias('help', 'h')
  .alias('version', 'v')
  
  // Global options
  .option('verbose', {
    alias: 'V',
    type: 'boolean',
    description: 'Enable verbose logging',
    global: true
  })
  
  // Commands
  .command(
    'init [path]',
    'Initialize a new ctx-run workspace',
    (yargs) => {
      return yargs
        .positional('path', {
          describe: 'Target directory for workspace',
          type: 'string',
          default: '.'
        })
        .option('force', {
          alias: 'f',
          type: 'boolean',
          description: 'Reinitialize existing workspace',
          default: false
        });
    },
    async (argv) => {
      try {
        await initCommand({
          path: argv.path,
          force: argv.force
        });
      } catch (error) {
        console.error(chalk.red('Command failed:'), error);
        process.exit(1);
      }
    }
  )
  
  .command(
    'ingest',
    'Import conversation data into a session',
    (yargs) => {
      return yargs
        .option('session', {
          alias: 's',
          type: 'string',
          description: 'Session identifier',
          demandOption: true
        })
        .option('from', {
          type: 'string',
          description: 'Input source (file path or "stdio")',
          demandOption: true
        })
        .option('format', {
          type: 'string',
          choices: ['auto', 'json', 'jsonl', 'claude-export'],
          description: 'Input format (auto-detected if not specified)',
          default: 'auto'
        });
    },
    async (argv) => {
      try {
        await ingestCommand({
          session: argv.session,
          from: argv.from,
          format: argv.format as any
        });
      } catch (error) {
        console.error(chalk.red('Command failed:'), error);
        process.exit(1);
      }
    }
  )
  
  .command(
    'index',
    'Build search indexes for a session',
    (yargs) => {
      return yargs
        .option('session', {
          alias: 's',
          type: 'string',
          description: 'Session identifier',
          demandOption: true
        })
        .option('force', {
          alias: 'f',
          type: 'boolean',
          description: 'Rebuild existing indexes',
          default: false
        });
    },
    async (argv) => {
      try {
        await indexCommand({
          session: argv.session,
          force: argv.force
        });
      } catch (error) {
        console.error(chalk.red('Command failed:'), error);
        process.exit(1);
      }
    }
  )
  
  .command(
    'query <query>',
    'Search conversation history',
    (yargs) => {
      return yargs
        .positional('query', {
          describe: 'Search query',
          type: 'string'
        })
        .option('session', {
          alias: 's',
          type: 'string',
          description: 'Session identifier',
          demandOption: true
        })
        .option('format', {
          type: 'string',
          choices: ['json', 'pretty', 'summary'],
          description: 'Output format',
          default: 'pretty'
        })
        .option('output', {
          alias: 'o',
          type: 'string',
          description: 'Output file path (for JSON format)'
        })
        .option('debug', {
          type: 'boolean',
          description: 'Include debug information',
          default: false
        });
    },
    async (argv) => {
      try {
        await queryCommand(argv.query, {
          session: argv.session,
          format: argv.format as any,
          output: argv.output,
          debug: argv.debug
        });
      } catch (error) {
        console.error(chalk.red('Command failed:'), error);
        process.exit(1);
      }
    }
  )
  
  .command(
    'serve',
    'Start development server with web UI',
    (yargs) => {
      return yargs
        .option('port', {
          alias: 'p',
          type: 'number',
          description: 'Server port',
          default: 7071
        })
        .option('host', {
          type: 'string',
          description: 'Server host',
          default: 'localhost'
        })
        .option('open', {
          type: 'boolean',
          description: 'Open browser automatically',
          default: false
        });
    },
    async (argv) => {
      try {
        await serveCommand({
          port: argv.port,
          host: argv.host,
          open: argv.open
        });
      } catch (error) {
        console.error(chalk.red('Command failed:'), error);
        process.exit(1);
      }
    }
  )
  
  .command(
    'checkpoint',
    'Create a checkpoint snapshot',
    (yargs) => {
      return yargs
        .option('session', {
          alias: 's',
          type: 'string',
          description: 'Session identifier',
          demandOption: true
        })
        .option('label', {
          alias: 'l',
          type: 'string',
          description: 'Checkpoint label',
          demandOption: true
        });
    },
    async (argv) => {
      console.log(chalk.yellow('🚧 Checkpoint functionality coming soon...'));
      console.log(chalk.gray(`   Session: ${argv.session}`));
      console.log(chalk.gray(`   Label: ${argv.label}`));
    }
  )
  
  .command(
    'eval',
    'Run evaluation suite',
    (yargs) => {
      return yargs
        .option('suite', {
          type: 'string',
          description: 'Evaluation suite path',
          demandOption: true
        })
        .option('output', {
          alias: 'o',
          type: 'string',
          description: 'Output file path for results'
        })
        .option('tune', {
          type: 'boolean',
          description: 'Enable parameter tuning',
          default: false
        })
        .option('iterations', {
          type: 'number',
          description: 'Number of parameter tuning iterations',
          default: 10
        });
    },
    async (argv) => {
      try {
        await evalCommand({
          suite: argv.suite,
          output: argv.output,
          tune: argv.tune,
          iterations: argv.iterations
        });
      } catch (error) {
        console.error(chalk.red('Command failed:'), error);
        process.exit(1);
      }
    }
  )
  
  // Examples and help
  .example('$0 init', 'Initialize workspace in current directory')
  .example('$0 ingest -s my-session --from chat.json', 'Import conversation from JSON file')
  .example('$0 index -s my-session', 'Build search indexes for session')
  .example('$0 query "how to setup auth" -s my-session', 'Search conversation history')
  .example('$0 serve -p 8080 --open', 'Start dev server on port 8080 and open browser')
  .example('$0 eval eval-suite.json --tune', 'Run evaluation with parameter tuning')
  
  .epilogue(chalk.blue('For more information, visit: https://github.com/your-org/ctx-run'))
  
  // Error handling
  .fail((msg, err, yargs) => {
    if (err) {
      console.error(chalk.red('❌ Error:'), err.message);
      if (argv.verbose) {
        console.error(err.stack);
      }
    } else {
      console.error(chalk.red('❌'), msg);
      console.log();
      console.log(yargs.help());
    }
    process.exit(1);
  })
  
  .demandCommand(1, chalk.yellow('Please specify a command. Use --help for available commands.'))
  .strict()
  .wrap(Math.min(120, process.stdout.columns || 80))
  .parse();