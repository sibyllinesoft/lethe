import { startServer } from '@lethe/api-server';
import { Workspace } from '../workspace';

export interface ServeArgs {
  port?: number;
  host?: string;
}

export async function serveCommand(args: ServeArgs = {}): Promise<void> {
  const workspace = new Workspace();
  await workspace.ensure();

  const server = await startServer({
    port: args.port ?? 3001,
    host: args.host ?? '127.0.0.1',
  });

  console.log(`🌐 API server listening on http://${server.host}:${server.port}`);
}
