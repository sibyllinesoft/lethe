/**
 * Integration Test for Rust Hot Path
 * 
 * This test validates that the Rust hot path integration works correctly
 * and provides the expected performance improvements.
 */

import { selectOptimalContextRust, candidatesToRustAtoms, createTypeQuotas } from './ctx-run/packages/core/src/retrieval/rust-hotpath.js';

async function testRustIntegration() {
  console.log('🧪 Testing Rust Hot Path Integration...\n');
  
  // Mock candidates data (simulating retrieved search results)
  const mockCandidates = [
    { docId: 'chunk_1', score: 0.95, text: 'This is a Python function that handles user authentication using JWT tokens and secure hashing algorithms.', kind: 'code' },
    { docId: 'chunk_2', score: 0.88, text: 'Error handling in TypeScript requires careful consideration of async operations and proper error propagation patterns.', kind: 'text' },
    { docId: 'chunk_3', score: 0.82, text: 'import { useState, useEffect } from "react"; import { authenticate } from "./auth";', kind: 'import' },
    { docId: 'chunk_4', score: 0.79, text: 'Database connection pooling is essential for high-performance applications to avoid connection exhaustion.', kind: 'text' },
    { docId: 'chunk_5', score: 0.75, text: 'function calculateHash(input: string): string { return crypto.createHash("sha256").update(input).digest("hex"); }', kind: 'function' },
    { docId: 'chunk_6', score: 0.72, text: 'Configuration management best practices include environment-specific settings and secure credential storage.', kind: 'text' },
    { docId: 'chunk_7', score: 0.68, text: 'Runtime error: TypeError: Cannot read property "length" of undefined at line 42', kind: 'error' },
    { docId: 'chunk_8', score: 0.65, text: 'API documentation suggests using rate limiting to prevent abuse and ensure fair resource usage.', kind: 'text' }
  ];
  
  console.log(`📊 Input: ${mockCandidates.length} candidates`);
  
  // Test 1: Convert candidates to Rust format
  console.log('\n1️⃣ Testing candidate conversion...');
  const { atoms, textBuffer } = candidatesToRustAtoms(mockCandidates);
  console.log(`✅ Converted ${atoms.length} candidates to Rust atoms`);
  console.log(`📝 Text buffer size: ${textBuffer.length} bytes`);
  
  // Test 2: Create type quotas
  console.log('\n2️⃣ Testing type quota creation...');
  const mockConfig = {
    gamma_kind_boost: { code: 0.2, function: 0.15, error: 0.3 },
    k_final: 5
  };
  const quotas = createTypeQuotas(500, mockConfig); // 500 token budget
  console.log(`✅ Created ${quotas.length} type quotas:`);
  quotas.forEach(quota => {
    console.log(`   - ${quota.chunk_type}: min ${quota.min_tokens} tokens (${(quota.target_ratio * 100).toFixed(1)}% target)`);
  });
  
  // Test 3: Rust optimization
  console.log('\n3️⃣ Testing Rust context optimization...');
  const startTime = Date.now();
  
  const result = await selectOptimalContextRust(
    atoms,
    quotas, 
    500, // Token budget
    0.1, // Lambda threshold
    textBuffer
  );
  
  const endTime = Date.now();
  const totalTime = endTime - startTime;
  
  console.log(`✅ Rust optimization completed successfully!`);
  console.log(`⏱️  Total time: ${totalTime}ms`);
  console.log(`⚡ Processing time: ${(result.processing_time_ns / 1e6).toFixed(2)}ms`);
  console.log(`📦 Selected: ${result.selected_atoms.length} atoms`);
  console.log(`🎯 Total tokens: ${result.total_tokens}`);
  console.log(`📊 Coverage score: ${(result.coverage_score * 100).toFixed(1)}%`);
  console.log(`🎨 Diversity score: ${(result.diversity_score * 100).toFixed(1)}%`);
  
  // Test 4: Performance validation
  console.log('\n4️⃣ Performance validation...');
  const expectedMaxTime = 50; // 50ms maximum expected time
  const actualTime = result.processing_time_ns / 1e6;
  
  if (actualTime < expectedMaxTime) {
    console.log(`✅ Performance EXCELLENT: ${actualTime.toFixed(2)}ms < ${expectedMaxTime}ms target`);
  } else {
    console.log(`⚠️  Performance acceptable but slower than target: ${actualTime.toFixed(2)}ms > ${expectedMaxTime}ms`);
  }
  
  // Test 5: Quality validation
  console.log('\n5️⃣ Quality validation...');
  const minCoverage = 0.8; // 80% minimum coverage
  const minDiversity = 0.7; // 70% minimum diversity
  
  const coveragePass = result.coverage_score >= minCoverage;
  const diversityPass = result.diversity_score >= minDiversity;
  
  console.log(`📊 Coverage: ${coveragePass ? '✅' : '❌'} ${(result.coverage_score * 100).toFixed(1)}% (target: ${(minCoverage * 100).toFixed(1)}%)`);
  console.log(`🎨 Diversity: ${diversityPass ? '✅' : '❌'} ${(result.diversity_score * 100).toFixed(1)}% (target: ${(minDiversity * 100).toFixed(1)}%)`);
  
  // Test 6: Selection validation
  console.log('\n6️⃣ Selection validation...');
  console.log('Selected chunks:');
  const selectedCandidates = mockCandidates.filter(c => result.selected_atoms.includes(c.docId));
  selectedCandidates.forEach((candidate, i) => {
    console.log(`   ${i + 1}. [${candidate.kind}] ${candidate.docId} (score: ${candidate.score.toFixed(2)})`);
  });
  
  // Summary
  console.log('\n🎉 Integration Test Summary:');
  console.log('─'.repeat(50));
  console.log(`Performance: ${actualTime.toFixed(2)}ms processing time`);
  console.log(`Quality: ${(result.coverage_score * 100).toFixed(1)}% coverage, ${(result.diversity_score * 100).toFixed(1)}% diversity`);
  console.log(`Selection: ${result.selected_atoms.length}/${mockCandidates.length} atoms selected`);
  console.log(`Token efficiency: ${result.total_tokens}/500 tokens used`);
  
  const allTestsPass = actualTime < expectedMaxTime * 2 && coveragePass && diversityPass && result.selected_atoms.length > 0;
  console.log(`\n${allTestsPass ? '✅ ALL TESTS PASSED' : '❌ Some tests need attention'}`);
  
  if (allTestsPass) {
    console.log('\n🚀 Rust hot path integration is ready for production!');
  }
  
  return {
    success: allTestsPass,
    processingTimeMs: actualTime,
    coverage: result.coverage_score,
    diversity: result.diversity_score,
    selectedCount: result.selected_atoms.length
  };
}

// Run the test
testRustIntegration().catch(console.error);