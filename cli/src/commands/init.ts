import { Workspace } from '../workspace';

export interface InitArgs {
  directory?: string;
}

export async function initCommand(args: InitArgs = {}): Promise<void> {
  const workspace = new Workspace(args.directory);
  await workspace.ensure();
  console.log('✅ Lethe workspace created at', workspace.path);
}
