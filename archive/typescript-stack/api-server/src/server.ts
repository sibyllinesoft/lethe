import type { CallsFilters, CompareRequest } from '@lethe/types';
import {
  diffCalls,
  getCall,
  getRunComparison,
  getRuns,
  getStats,
  listCalls,
} from './data';
import { Elysia as StubElysia } from './elysia-fallback';

let ElysiaConstructor: typeof StubElysia;

try {
  const mod = await import('elysia');
  ElysiaConstructor = (mod as any).Elysia ?? (mod as any).default ?? StubElysia;
} catch {
  ElysiaConstructor = StubElysia;
}

export interface ServerConfig {
  port?: number;
  host?: string;
}

export function createApp() {
  return new ElysiaConstructor({ prefix: '/api' })
    .get('/ping', () => ({ status: 'ok', timestamp: Date.now() }))
    .get('/health', () => ({ status: 'healthy', timestamp: new Date().toISOString() }))
    .get('/calls', ({ query }) => listCalls(query as CallsFilters))
    .get('/calls/stats', () => getStats())
    .get('/calls/runs', () => ({ run_ids: getRuns() }))
    .get('/calls/:id', ({ params, set }) => {
      const call = getCall(params.id);
      if (!call) {
        set.status = 404;
        return { error: 'Call not found' };
      }
      return call;
    })
    .get('/calls/:id/pre-post-diff', ({ params, set }) => {
      const call = getCall(params.id);
      if (!call) {
        set.status = 404;
        return { error: 'Call not found' };
      }
      return {
        prompt: call.prompt,
        pre_context: call.pre_context,
        post_context: call.post_context,
      };
    })
    .get('/compare', ({ query, set }) => {
      const { call_id_a, call_id_b } = query as Partial<CompareRequest>;
      if (!call_id_a || !call_id_b) {
        set.status = 400;
        return { error: 'call_id_a and call_id_b are required' };
      }
      const callA = getCall(call_id_a);
      const callB = getCall(call_id_b);
      if (!callA || !callB) {
        set.status = 404;
        return { error: 'One or both calls not found' };
      }
      return diffCalls(callA, callB);
    })
    .post('/compare', ({ body, set }) => {
      const { call_id_a, call_id_b } = body as CompareRequest;
      if (!call_id_a || !call_id_b) {
        set.status = 400;
        return { error: 'call_id_a and call_id_b are required' };
      }
      const callA = getCall(call_id_a);
      const callB = getCall(call_id_b);
      if (!callA || !callB) {
        set.status = 404;
        return { error: 'One or both calls not found' };
      }
      return diffCalls(callA, callB);
    })
    .get('/compare/runs/:runId', ({ params, set }) => {
      const comparison = getRunComparison(params.runId);
      if (!comparison) {
        set.status = 404;
        return { error: 'Run not found' };
      }
      return comparison;
    })
    .post('/bundle/save', ({ body }) => {
      const payload = (body ?? {}) as { id?: string };
      return {
        success: true,
        id: payload.id ?? `bundle-${Date.now()}`,
      };
    })
    .post('/shutdown', () => ({ message: 'Use CLI to manage lifecycle.' }));
}

export async function startServer(config: ServerConfig = {}) {
  const app = createApp();
  const server = await app.listen({
    port: config.port ?? 3001,
    hostname: config.host ?? '127.0.0.1',
  });

  return {
    port: server.port,
    host: server.hostname,
    stop: () => server.stop(),
  };
}
