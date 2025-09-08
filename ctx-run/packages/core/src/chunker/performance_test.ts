#!/usr/bin/env node
/**
 * Performance Test for Enhanced Tokenization
 * ==========================================
 * 
 * Validates that the enhanced tokenization improvements don't cause
 * latency regression while providing better accuracy.
 */

import { performance } from 'perf_hooks';
import { chunkMessage } from './index.js';
import { Message } from '@lethe/sqlite';

// Test data - various types of content that would be chunked
const testMessages: Message[] = [
  {
    id: 'test-1',
    session_id: 'perf-test',
    turn: 1,
    role: 'user',
    text: 'Simple test message that should be processed quickly.',
    ts: Date.now()
  },
  {
    id: 'test-2', 
    session_id: 'perf-test',
    turn: 2,
    role: 'assistant',
    text: `Here's a longer message with code:

\`\`\`javascript
function calculateMetrics(data) {
  return data.map(item => ({
    id: item.id,
    score: item.relevance * item.quality,
    normalized: item.score / item.max_score
  }));
}
\`\`\`

This function processes the evaluation metrics and normalizes them for comparison.`,
    ts: Date.now()
  },
  {
    id: 'test-3',
    session_id: 'perf-test', 
    turn: 3,
    role: 'user',
    text: 'Complex query with multiple parts: 1) How do I implement binary search? 2) What are the time complexity considerations? 3) Can you provide examples in TypeScript? 4) How does it compare to linear search for different dataset sizes? 5) Are there any edge cases I should consider?',
    ts: Date.now()
  },
  {
    id: 'test-4',
    session_id: 'perf-test',
    turn: 4, 
    role: 'assistant',
    text: 'Long response with mixed content: ' + Array(100).fill('This is a sentence with various words and punctuation! ').join('') +
          '\n\nAnd some code:\n```python\ndef search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```\n\n' +
          Array(50).fill('Additional explanatory text with technical terms like algorithms, optimization, and performance characteristics. ').join(''),
    ts: Date.now()
  }
];

function runPerformanceTest() {
  console.log('🚀 Running Enhanced Tokenization Performance Test');
  console.log('=' .repeat(60));
  
  const results = [];
  
  for (const [index, message] of testMessages.entries()) {
    console.log(`\n📝 Test ${index + 1}: ${message.text.substring(0, 50)}...`);
    
    // Warm up (run once to avoid cold start effects)
    chunkMessage(message);
    
    // Performance test - run multiple times
    const iterations = 100;
    const times = [];
    
    for (let i = 0; i < iterations; i++) {
      const startTime = performance.now();
      const chunks = chunkMessage(message);
      const endTime = performance.now();
      
      times.push(endTime - startTime);
      
      // Validate on first iteration
      if (i === 0) {
        console.log(`   📊 Chunks created: ${chunks.length}`);
        console.log(`   🔢 Total tokens: ${chunks.reduce((sum, c) => sum + c.tokens, 0)}`);
        console.log(`   📏 Text length: ${message.text.length} chars`);
        
        // Validate chunks have reasonable token counts
        for (const chunk of chunks) {
          if (chunk.tokens <= 0) {
            throw new Error(`Invalid token count: ${chunk.tokens} for chunk ${chunk.id}`);
          }
          if (chunk.tokens > 400) { // Should respect target of ~320
            console.warn(`   ⚠️  Large chunk detected: ${chunk.tokens} tokens`);
          }
        }
      }
    }
    
    // Calculate statistics
    const avgTime = times.reduce((a, b) => a + b) / times.length;
    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    const p95Time = times.sort((a, b) => a - b)[Math.floor(times.length * 0.95)];
    
    const result = {
      testIndex: index + 1,
      textLength: message.text.length,
      avgTime,
      minTime, 
      maxTime,
      p95Time,
      iterations
    };
    
    results.push(result);
    
    console.log(`   ⏱️  Average: ${avgTime.toFixed(2)}ms`);
    console.log(`   ⚡ Min: ${minTime.toFixed(2)}ms, Max: ${maxTime.toFixed(2)}ms`);
    console.log(`   📈 P95: ${p95Time.toFixed(2)}ms`);
    console.log(`   🎯 Throughput: ${(1000 / avgTime).toFixed(0)} msg/sec`);
  }
  
  // Overall summary
  console.log('\n' + '=' .repeat(60));
  console.log('📈 PERFORMANCE SUMMARY');
  console.log('=' .repeat(60));
  
  const totalAvgTime = results.reduce((sum, r) => sum + r.avgTime, 0) / results.length;
  const maxP95Time = Math.max(...results.map(r => r.p95Time));
  
  console.log(`📊 Overall average: ${totalAvgTime.toFixed(2)}ms`);
  console.log(`📊 Worst P95: ${maxP95Time.toFixed(2)}ms`);
  console.log(`📊 Overall throughput: ${(1000 / totalAvgTime).toFixed(0)} msg/sec`);
  
  // Performance validation criteria
  const PERFORMANCE_THRESHOLDS = {
    maxAvgTime: 10.0,     // Average should be under 10ms
    maxP95Time: 50.0,     // P95 should be under 50ms
    minThroughput: 100    // Should process at least 100 msg/sec average
  };
  
  console.log('\n🎯 PERFORMANCE VALIDATION');
  console.log('-' .repeat(40));
  
  let allPassed = true;
  
  // Check average time
  if (totalAvgTime <= PERFORMANCE_THRESHOLDS.maxAvgTime) {
    console.log(`✅ Average time: ${totalAvgTime.toFixed(2)}ms <= ${PERFORMANCE_THRESHOLDS.maxAvgTime}ms`);
  } else {
    console.log(`❌ Average time: ${totalAvgTime.toFixed(2)}ms > ${PERFORMANCE_THRESHOLDS.maxAvgTime}ms`);
    allPassed = false;
  }
  
  // Check P95 time
  if (maxP95Time <= PERFORMANCE_THRESHOLDS.maxP95Time) {
    console.log(`✅ P95 time: ${maxP95Time.toFixed(2)}ms <= ${PERFORMANCE_THRESHOLDS.maxP95Time}ms`);
  } else {
    console.log(`❌ P95 time: ${maxP95Time.toFixed(2)}ms > ${PERFORMANCE_THRESHOLDS.maxP95Time}ms`);
    allPassed = false;
  }
  
  // Check throughput
  const overallThroughput = 1000 / totalAvgTime;
  if (overallThroughput >= PERFORMANCE_THRESHOLDS.minThroughput) {
    console.log(`✅ Throughput: ${overallThroughput.toFixed(0)} msg/sec >= ${PERFORMANCE_THRESHOLDS.minThroughput} msg/sec`);
  } else {
    console.log(`❌ Throughput: ${overallThroughput.toFixed(0)} msg/sec < ${PERFORMANCE_THRESHOLDS.minThroughput} msg/sec`);
    allPassed = false;
  }
  
  console.log('\n' + '=' .repeat(60));
  if (allPassed) {
    console.log('🎉 ALL PERFORMANCE TESTS PASSED!');
    console.log('✨ Enhanced tokenization maintains excellent performance');
    console.log('🔥 No latency regression detected');
  } else {
    console.log('⚠️  PERFORMANCE ISSUES DETECTED');
    console.log('🔧 Consider optimizing tokenization algorithm');
  }
  console.log('=' .repeat(60));
  
  return allPassed;
}

function runAccuracyComparison() {
  console.log('\n🔬 TOKENIZATION ACCURACY COMPARISON');
  console.log('=' .repeat(60));
  
  const testCases = [
    'hello world',
    'The quick brown fox jumps over the lazy dog.',
    'console.log("Hello, world!");',
    'function calculateScore(data: any[]): number { return data.length; }',
    '@user How do you implement #algorithms with 100% efficiency?',
    'Mixed content with code: `const x = 42;` and normal text.',
  ];
  
  console.log('📊 Enhanced vs Simple Approximation:');
  console.log('-' .repeat(40));
  
  for (const text of testCases) {
    const message: Message = {
      id: 'accuracy-test',
      session_id: 'accuracy-test',
      turn: 1,
      role: 'user',
      text,
      ts: Date.now()
    };
    
    const chunks = chunkMessage(message);
    const enhancedTokens = chunks[0].tokens;
    const simpleApprox = Math.ceil(text.length / 4);
    const difference = enhancedTokens - simpleApprox;
    const diffPercent = ((difference / simpleApprox) * 100).toFixed(1);
    
    console.log(`📝 "${text.substring(0, 40)}..."`);
    console.log(`   Enhanced: ${enhancedTokens} tokens`);
    console.log(`   Simple: ${simpleApprox} tokens`);
    console.log(`   Difference: ${difference > 0 ? '+' : ''}${difference} (${diffPercent}%)`);
    console.log();
  }
}

// Main execution
if (require.main === module) {
  try {
    const performancePassed = runPerformanceTest();
    runAccuracyComparison();
    
    if (performancePassed) {
      process.exit(0);
    } else {
      process.exit(1);
    }
  } catch (error) {
    console.error('❌ Performance test failed:', error);
    process.exit(1);
  }
}