/**
 * Lethe API Server
 * Lightweight server for the LLM analyzer frontend
 */
import { Elysia } from 'elysia';
import type { LLMCall, CallPair } from '@lethe/types';

const app = new Elysia()
  .get('/api/ping', () => ({ status: 'ok', timestamp: Date.now() }))
  
  .post('/api/bundle/save', async ({ body }) => {
    // Handle bundle save requests from the frontend
    console.log('Bundle save request:', body);
    return { success: true, id: Date.now().toString() };
  })
  
  .post('/api/shutdown', () => {
    // Graceful shutdown endpoint
    setTimeout(() => process.exit(0), 1000);
    return { message: 'Server shutting down...' };
  })
  
  .listen(3001);

console.log(`🦊 Lethe API server is running at http://localhost:3001`);
