import { describe, expect, test } from 'bun:test';
import { createApp } from '@lethe/api-server';

const app = createApp();

async function request(path: string, init?: RequestInit) {
  const url = new URL(path, 'http://localhost');
  return app.handle(new Request(url.toString(), init));
}

describe('API server', () => {
  test('responds to ping', async () => {
    const response = await request('/api/ping');
    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body.status).toBe('ok');
  });

  test('lists calls with metadata', async () => {
    const response = await request('/api/calls');
    const body = await response.json();
    expect(Array.isArray(body.calls)).toBe(true);
    expect(body.calls[0]).toHaveProperty('provider');
  });

  test('computes comparison diff', async () => {
    const callsResponse = await request('/api/calls');
    const calls = await callsResponse.json();
    const [first, second] = calls.calls;

    const diffResponse = await request(`/api/compare?call_id_a=${first.id}&call_id_b=${second.id}`);
    expect(diffResponse.status).toBe(200);
    const diff = await diffResponse.json();
    expect(diff).toHaveProperty('prompt_diff');
  });
});
