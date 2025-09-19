import { afterEach, describe, expect, test } from 'bun:test';
import { mkdtemp, rm, writeFile } from 'fs/promises';
import { tmpdir } from 'os';
import { join } from 'path';
import { Workspace } from '@lethe/cli/workspace';
import { ingestCommand } from '@lethe/cli/commands/ingest';
import { initCommand } from '@lethe/cli/commands/init';
import type { MessageRole } from '@lethe/types';

let workspaceDir: string;
const originalCwd = process.cwd();

async function setupTempWorkspace() {
  workspaceDir = await mkdtemp(join(tmpdir(), 'lethe-cli-'));
  process.chdir(workspaceDir);
}

afterEach(async () => {
  if (workspaceDir) {
    await rm(workspaceDir, { recursive: true, force: true });
  }
  process.chdir(originalCwd);
});

describe('CLI workspace helpers', () => {
  test('init command creates workspace file', async () => {
    await setupTempWorkspace();
    await initCommand({ directory: workspaceDir });
    const workspace = new Workspace(workspaceDir);
    const data = await workspace.load();
    expect(Object.keys(data.sessions)).toHaveLength(0);
  });

  test('ingest command appends messages to workspace', async () => {
    await setupTempWorkspace();
    await initCommand({ directory: workspaceDir });

    const payload = [
      { role: 'user' as MessageRole, text: 'Any incidents overnight?' },
      { role: 'assistant' as MessageRole, text: 'No incidents recorded.' },
    ];

    const file = join(workspaceDir, 'messages.json');
    await writeFile(file, JSON.stringify(payload));

    await ingestCommand({ file, session: 'demo' });

    const workspace = new Workspace(workspaceDir);
    const data = await workspace.load();
    expect(data.sessions.demo.messages).toHaveLength(2);
  });
});
