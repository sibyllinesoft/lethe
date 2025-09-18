#!/usr/bin/env bun
import yargs from 'yargs/yargs';
import type { CommandModule } from 'yargs';
import { hideBin } from 'yargs/helpers';
import { initCommand, type InitArgs } from './commands/init';
import { ingestCommand, type IngestArgs } from './commands/ingest';
import { queryCommand, type QueryArgs } from './commands/query';
import { serveCommand, type ServeArgs } from './commands/serve';

const initModule: CommandModule<unknown, InitArgs> = {
  command: 'init',
  describe: 'Create a local Lethe workspace',
  builder: (argv) =>
    argv.option('directory', {
      describe: 'Workspace root (defaults to current directory)',
      type: 'string',
    }),
  handler: async (argv) => {
    await initCommand({ directory: argv.directory });
  },
};

const ingestModule: CommandModule<unknown, IngestArgs & { session?: string }> = {
  command: 'ingest <file>',
  describe: 'Ingest messages from a JSON file',
  builder: (argv) =>
    argv
      .positional('file', {
        type: 'string',
        demandOption: true,
        describe: 'Path to JSON file containing messages',
      })
      .option('session', {
        type: 'string',
        describe: 'Session identifier',
      }),
  handler: async (argv) => {
    await ingestCommand({ file: argv.file, session: argv.session });
  },
};

const queryModule: CommandModule<unknown, QueryArgs> = {
  command: 'query <session> <query>',
  describe: 'Build a context pack for the provided query',
  builder: (argv) =>
    argv
      .positional('session', { type: 'string', demandOption: true })
      .positional('query', { type: 'string', demandOption: true })
      .option('topK', {
        type: 'number',
        describe: 'Limit the number of messages shown',
      })
      .option('config', {
        type: 'string',
        describe: 'Inline JSON fragment overriding config',
      }),
  handler: async (argv) => {
    await queryCommand({
      session: argv.session,
      query: argv.query,
      topK: argv.topK,
      config: argv.config,
    });
  },
};

const serveModule: CommandModule<unknown, ServeArgs> = {
  command: 'serve',
  describe: 'Start the local API server',
  builder: (argv) =>
    argv
      .option('port', { type: 'number', default: 3001 })
      .option('host', { type: 'string', default: '127.0.0.1' }),
  handler: async (argv) => {
    await serveCommand({ port: argv.port, host: argv.host });
  },
};

async function main() {
  await yargs(hideBin(process.argv))
    .scriptName('lethe')
    .usage('$0 <cmd> [args]')
    .command(initModule)
    .command(ingestModule)
    .command(queryModule)
    .command(serveModule)
    .demandCommand(1, 'Please provide a command')
    .help()
    .strict()
    .wrap(100)
    .parseAsync();
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
