#!/usr/bin/env node

/**
 * Test script for Iteration 2 Query Understanding
 */

const Database = require('better-sqlite3');
const { testQueryUnderstanding } = require('./packages/core/dist/index.js');
const { migrate, upsertConfig } = require('./packages/sqlite/dist/index.js');

async function testQueryUnderstandingIntegration() {
  console.log('🧪 Testing Iteration 2 Query Understanding Integration...\n');
  
  // Create test database
  const db = new Database(':memory:');
  migrate(db);
  
  // Set up Iteration 2 configuration
  upsertConfig(db, 'plan', {
    query_rewrite: true,
    query_decompose: true
  });
  
  upsertConfig(db, 'timeouts', {
    rewrite_ms: 1500,
    decompose_ms: 2000
  });
  
  upsertConfig(db, 'query_understanding', {
    enabled: true,
    llm_model: 'xgen-small:4b',
    max_tokens: 256,
    temperature: 0.1,
    max_subqueries: 3,
    similarity_threshold: 0.8
  });
  
  // Test queries
  const testQueries = [
    'async error handling patterns',
    'How do I implement authentication middleware in Express.js?',
    'typescript generic types with constraints',
    'database transactions and rollback strategies',
    'performance optimization for React components'
  ];
  
  console.log('Configuration loaded:');
  console.log('- Query rewrite: enabled, 1500ms timeout');
  console.log('- Query decomposition: enabled, 2000ms timeout');
  console.log('- Model: xgen-small:4b\n');
  
  for (let i = 0; i < testQueries.length; i++) {
    const query = testQueries[i];
    console.log(`Test ${i + 1}: "${query}"`);
    
    const start = Date.now();
    const result = await testQueryUnderstanding(db, query, []);
    const duration = Date.now() - start;
    
    if (result.success) {
      const processed = result.result;
      console.log(`✅ Success (${duration}ms)`);
      console.log(`   Original: "${processed.original_query}"`);
      console.log(`   Canonical: "${processed.canonical_query || 'none'}"`);
      console.log(`   Subqueries: [${(processed.subqueries || []).join(', ')}]`);
      console.log(`   LLM calls: ${processed.llm_calls_made}`);
      console.log(`   Rewrite success: ${processed.rewrite_success}`);
      console.log(`   Decompose success: ${processed.decompose_success}`);
      if (processed.errors.length > 0) {
        console.log(`   Errors: ${processed.errors.join(', ')}`);
      }
    } else {
      console.log(`❌ Failed: ${result.error}`);
    }
    
    console.log('');
  }
  
  console.log('🎯 Quality Gates Check:');
  console.log('- Latency p50 target: ≤3500ms');
  console.log('- Rewrite failure rate target: ≤5%');
  console.log('- JSON parse errors target: ≤1%');
  console.log('- Integration: Query understanding → HyDE → Retrieval');
  
  db.close();
  console.log('\n✅ Query understanding integration test completed');
}

testQueryUnderstandingIntegration().catch(console.error);