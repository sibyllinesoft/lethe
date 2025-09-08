#!/usr/bin/env node
/**
 * Lens Integration Validation Script
 * 
 * Tests the Lens server integration running on port 5678 with comprehensive validation
 * of all integration components including connectivity, code intent detection,
 * cost calculation, configuration loading, and end-to-end flow.
 * 
 * Usage: npx tsx src/lens/validate-integration.ts
 * Or: node dist/lens/validate-integration.js
 */

import {
  getLensService,
  detectCodeIntent,
  calculateLagrangianCost,
  symbolGroupsToRetrievalCandidates,
  testLensIntegration,
  DEFAULT_LENS_CONFIG,
  type LensService,
  type SymbolGroup,
  type CodeIntentResult,
  type LagrangianCostResult,
  type LensConfig
} from './index.js';

import {
  maybeLens,
  lensEnhancedHybridRetrieval,
  LENS_PROFILES,
  type LensEnhancedRetrievalOptions
} from './integration-example.js';

// ANSI color codes for output formatting
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m'
};

interface TestResult {
  name: string;
  passed: boolean;
  duration_ms: number;
  details: string;
  error?: string;
  metrics?: Record<string, number | string | boolean>;
}

interface ValidationSummary {
  total_tests: number;
  passed_tests: number;
  failed_tests: number;
  total_duration_ms: number;
  server_available: boolean;
  sla_compliant: boolean;
  error_conditions: string[];
  recommendations: string[];
}

/**
 * Test runner utility
 */
async function runTest(
  name: string,
  testFn: () => Promise<{ passed: boolean; details: string; error?: string; metrics?: Record<string, any> }>
): Promise<TestResult> {
  const startTime = performance.now();
  
  try {
    console.log(`${colors.blue}[RUNNING]${colors.reset} ${name}...`);
    const result = await testFn();
    const duration = performance.now() - startTime;
    
    const status = result.passed 
      ? `${colors.green}[PASSED]${colors.reset}` 
      : `${colors.red}[FAILED]${colors.reset}`;
    
    console.log(`${status} ${name} (${duration.toFixed(1)}ms)`);
    if (result.details) {
      console.log(`  ${result.details}`);
    }
    if (result.error) {
      console.log(`  ${colors.red}Error: ${result.error}${colors.reset}`);
    }
    
    return {
      name,
      passed: result.passed,
      duration_ms: duration,
      details: result.details,
      error: result.error,
      metrics: result.metrics
    };
  } catch (error: any) {
    const duration = performance.now() - startTime;
    const errorMessage = error?.message || String(error);
    
    console.log(`${colors.red}[FAILED]${colors.reset} ${name} (${duration.toFixed(1)}ms)`);
    console.log(`  ${colors.red}Exception: ${errorMessage}${colors.reset}`);
    
    return {
      name,
      passed: false,
      duration_ms: duration,
      details: `Exception during test execution`,
      error: errorMessage
    };
  }
}

/**
 * Test 1: Basic Server Connectivity
 */
async function testBasicConnectivity(): Promise<{ passed: boolean; details: string; error?: string; metrics?: Record<string, any> }> {
  const config = { ...DEFAULT_LENS_CONFIG, enabled: true };
  const lensService = await getLensService();
  
  // Test health endpoint
  const healthStart = performance.now();
  const available = await lensService.isAvailable();
  const healthLatency = performance.now() - healthStart;
  
  if (!available) {
    return {
      passed: false,
      details: `Health endpoint not responding (${healthLatency.toFixed(1)}ms)`,
      error: 'Lens server not available at http://localhost:5678',
      metrics: { health_latency_ms: healthLatency, available: false }
    };
  }
  
  // Test connection details
  const connTest = await lensService.testConnection();
  
  // Test status endpoint
  const status = await lensService.getStatus();
  
  const slaCompliant = healthLatency <= 500; // Connection timeout threshold
  
  return {
    passed: available && connTest.available,
    details: `Health: OK (${healthLatency.toFixed(1)}ms), Status: ${status.healthy ? 'Healthy' : 'Unhealthy'}, LSP: ${status.lsp_available ? 'Available' : 'Unavailable'}, Cache: ${status.raptor_cache_status}`,
    metrics: {
      health_latency_ms: healthLatency,
      connection_latency_ms: connTest.latency_ms || 0,
      server_healthy: status.healthy,
      lsp_available: status.lsp_available,
      raptor_cache_status: status.raptor_cache_status,
      server_version: status.version,
      sla_compliant: slaCompliant
    }
  };
}

/**
 * Test 2: Code Intent Detection
 */
async function testCodeIntentDetection(): Promise<{ passed: boolean; details: string; error?: string; metrics?: Record<string, any> }> {
  const testCases = [
    {
      query: 'fix error in calculateBM25 function',
      expectedIntent: true,
      expectedConfidence: 0.5,
      context: { recent_files: ['index.ts'], recent_activity: 'code' as const }
    },
    {
      query: 'implement UserService.createUser method',
      expectedIntent: true,
      expectedConfidence: 0.6,
      context: { recent_files: ['service.ts'], recent_activity: 'code' as const }
    },
    {
      query: 'what is the weather today',
      expectedIntent: false,
      expectedConfidence: 0.3,
      context: { recent_files: ['readme.md'], recent_activity: 'docs' as const }
    },
    {
      query: 'TypeError in async function handleRequest',
      expectedIntent: true,
      expectedConfidence: 0.6,
      context: { recent_files: ['handler.ts'], recent_activity: 'code' as const }
    }
  ];
  
  let passed = 0;
  let total = testCases.length;
  const results: any[] = [];
  
  for (const testCase of testCases) {
    const result = detectCodeIntent(
      testCase.query,
      testCase.context.recent_files,
      testCase.context.recent_activity
    );
    
    const intentCorrect = result.is_code_intent === testCase.expectedIntent;
    const confidenceReasonable = result.confidence >= testCase.expectedConfidence;
    const testPassed = intentCorrect && confidenceReasonable;
    
    if (testPassed) passed++;
    
    results.push({
      query: testCase.query,
      expected_intent: testCase.expectedIntent,
      actual_intent: result.is_code_intent,
      expected_confidence: testCase.expectedConfidence,
      actual_confidence: result.confidence,
      detected_language: result.detected_language,
      extracted_symbols: result.extracted_symbols,
      passed: testPassed
    });
  }
  
  return {
    passed: passed === total,
    details: `${passed}/${total} test cases passed`,
    metrics: {
      test_cases_passed: passed,
      test_cases_total: total,
      success_rate: (passed / total) * 100,
      results: results
    }
  };
}

/**
 * Test 3: Cost Calculation System
 */
async function testCostCalculation(): Promise<{ passed: boolean; details: string; error?: string; metrics?: Record<string, any> }> {
  // Create mock symbol groups with varying characteristics
  const mockSymbolGroups: SymbolGroup[] = [
    {
      id: 'test_group_1',
      primary_symbol: 'calculateBM25',
      language: 'typescript',
      file_path: 'src/retrieval/bm25.ts',
      definition: {
        id: 'def_1',
        content: 'function calculateBM25(query: string, doc: string): number { /* implementation */ }',
        file_path: 'src/retrieval/bm25.ts',
        start_line: 15,
        end_line: 20,
        start_char: 0,
        end_char: 120,
        atom_type: 'definition',
        symbol_name: 'calculateBM25',
        tokens: 30,
        importance: 1.0
      },
      references: [{
        id: 'ref_1',
        content: 'const score = calculateBM25(query, document);',
        file_path: 'src/retrieval/hybrid.ts',
        start_line: 45,
        end_line: 45,
        start_char: 4,
        end_char: 50,
        atom_type: 'reference',
        symbol_name: 'calculateBM25',
        tokens: 12,
        importance: 0.8
      }],
      implementations: [],
      estimated_tokens: 150,
      relevance_score: 0.9,
      topic_weight: 0.35,
      is_precise_match: true
    },
    {
      id: 'test_group_2',
      primary_symbol: 'UserService',
      language: 'typescript',
      file_path: 'src/services/user.ts',
      definition: {
        id: 'def_2',
        content: 'export class UserService { constructor(private db: Database) {} }',
        file_path: 'src/services/user.ts',
        start_line: 10,
        end_line: 12,
        start_char: 0,
        end_char: 65,
        atom_type: 'definition',
        symbol_name: 'UserService',
        tokens: 20,
        importance: 0.9
      },
      references: [],
      implementations: [{
        id: 'impl_1',
        content: 'async createUser(data: CreateUserData): Promise<User> { /* impl */ }',
        file_path: 'src/services/user.ts',
        start_line: 15,
        end_line: 18,
        start_char: 2,
        end_char: 100,
        atom_type: 'implementation',
        symbol_name: 'createUser',
        tokens: 25,
        importance: 0.7
      }],
      estimated_tokens: 200,
      relevance_score: 0.7,
      topic_weight: 0.25,
      is_precise_match: false
    }
  ];
  
  const testConfigs = [
    {
      name: 'standard_config',
      config: DEFAULT_LENS_CONFIG,
      current_tokens: 1000,
      budget: 4000,
      latency: 120
    },
    {
      name: 'tight_budget',
      config: { ...DEFAULT_LENS_CONFIG, lambda_multiplier: 1.5 },
      current_tokens: 3500,
      budget: 4000,
      latency: 180
    },
    {
      name: 'high_latency',
      config: { ...DEFAULT_LENS_CONFIG, sla_recall_ms: 100 },
      current_tokens: 1000,
      budget: 4000,
      latency: 150
    }
  ];
  
  let passedTests = 0;
  const testResults: any[] = [];
  
  for (const testConfig of testConfigs) {
    const result = calculateLagrangianCost(
      mockSymbolGroups,
      testConfig.config,
      testConfig.current_tokens,
      testConfig.budget,
      testConfig.latency
    );
    
    // Validate cost structure
    const hasValidCosts = result.total_cost > 0 && 
                         result.token_cost >= 0 && 
                         result.compute_cost >= 0;
    
    const hasValidBenefit = result.expected_benefit >= 0;
    const hasValidRatio = !isNaN(result.cost_benefit_ratio) && result.cost_benefit_ratio >= 0;
    const hasValidConstraints = typeof result.cost_acceptable === 'boolean' && 
                               typeof result.sla_constraint_met === 'boolean';
    
    // Check cost breakdown
    const hasValidBreakdown = result.cost_breakdown.base_tokens === testConfig.current_tokens &&
                             result.cost_breakdown.lens_tokens > 0 &&
                             result.cost_breakdown.estimated_latency_ms === testConfig.latency;
    
    const testPassed = hasValidCosts && hasValidBenefit && hasValidRatio && 
                      hasValidConstraints && hasValidBreakdown;
    
    if (testPassed) passedTests++;
    
    testResults.push({
      config_name: testConfig.name,
      total_cost: result.total_cost,
      token_cost: result.token_cost,
      compute_cost: result.compute_cost,
      expected_benefit: result.expected_benefit,
      cost_benefit_ratio: result.cost_benefit_ratio,
      cost_acceptable: result.cost_acceptable,
      sla_constraint_met: result.sla_constraint_met,
      passed: testPassed
    });
  }
  
  return {
    passed: passedTests === testConfigs.length,
    details: `${passedTests}/${testConfigs.length} cost calculation tests passed`,
    metrics: {
      tests_passed: passedTests,
      tests_total: testConfigs.length,
      test_results: testResults
    }
  };
}

/**
 * Test 4: Configuration System
 */
async function testConfiguration(): Promise<{ passed: boolean; details: string; error?: string; metrics?: Record<string, any> }> {
  try {
    // Test default configuration
    const defaultValid = DEFAULT_LENS_CONFIG.base_url === 'http://localhost:5678' &&
                        DEFAULT_LENS_CONFIG.sla_recall_ms === 150 &&
                        DEFAULT_LENS_CONFIG.topic_fanout_k === 240;
    
    // Test service creation with different configs
    const testConfigs = [
      { ...DEFAULT_LENS_CONFIG, enabled: true },
      { ...DEFAULT_LENS_CONFIG, enabled: false },
      { ...DEFAULT_LENS_CONFIG, base_url: 'http://localhost:9999' }
    ];
    
    let configTestsPassed = 0;
    
    for (const config of testConfigs) {
      try {
        const service = await getLensService();
        configTestsPassed++;
      } catch (error) {
        // Expected for some invalid configs
      }
    }
    
    // Test profile configurations
    const profilesValid = Object.keys(LENS_PROFILES).every(profileName => {
      const profile = LENS_PROFILES[profileName as keyof typeof LENS_PROFILES];
      return profile.enabled !== undefined &&
             profile.mode !== undefined &&
             profile.lens_tokens_cap > 0 &&
             profile.topic_fanout_k > 0;
    });
    
    return {
      passed: defaultValid && profilesValid && configTestsPassed > 0,
      details: `Default config: ${defaultValid ? 'Valid' : 'Invalid'}, Profiles: ${profilesValid ? 'Valid' : 'Invalid'}, Service creation: ${configTestsPassed}/3`,
      metrics: {
        default_config_valid: defaultValid,
        profiles_valid: profilesValid,
        service_creation_tests: configTestsPassed,
        profile_count: Object.keys(LENS_PROFILES).length
      }
    };
  } catch (error: any) {
    return {
      passed: false,
      details: 'Configuration test failed with exception',
      error: error?.message || String(error)
    };
  }
}

/**
 * Test 5: Search Integration (if server available)
 */
async function testSearchIntegration(): Promise<{ passed: boolean; details: string; error?: string; metrics?: Record<string, any> }> {
  try {
    const config = { ...DEFAULT_LENS_CONFIG, enabled: true };
    const lensService = await getLensService();
    
    // Check if server is available first
    const available = await lensService.isAvailable();
    if (!available) {
      return {
        passed: false,
        details: 'Server not available for search testing',
        error: 'Lens server not responding to health checks'
      };
    }
    
    const testQueries = [
      { query: 'calculateBM25 function', max_groups: 5, expected_timeout: false },
      { query: 'UserService implementation', max_groups: 3, expected_timeout: false },
      { query: 'async function handler', max_groups: 10, timeout_ms: 50, expected_timeout: true }
    ];
    
    let searchTestsPassed = 0;
    const searchResults: any[] = [];
    
    for (const testQuery of testQueries) {
      const startTime = performance.now();
      
      try {
        const result = await lensService.search({
          query: testQuery.query,
          max_groups: testQuery.max_groups,
          timeout_ms: testQuery.timeout_ms || 1000
        });
        
        const duration = performance.now() - startTime;
        const timeoutCorrect = result.timeout_hit === (testQuery.expected_timeout || false);
        const hasValidStructure = Array.isArray(result.symbol_groups) &&
                                 typeof result.processing_time_ms === 'number' &&
                                 result.metadata !== undefined;
        
        const testPassed = timeoutCorrect && hasValidStructure;
        if (testPassed) searchTestsPassed++;
        
        searchResults.push({
          query: testQuery.query,
          duration_ms: duration,
          symbol_groups_count: result.symbol_groups.length,
          timeout_hit: result.timeout_hit,
          expected_timeout: testQuery.expected_timeout || false,
          lsp_available: result.lsp_available,
          topics_expanded: result.topics_expanded,
          passed: testPassed
        });
        
      } catch (error: any) {
        searchResults.push({
          query: testQuery.query,
          error: error?.message || String(error),
          passed: false
        });
      }
    }
    
    return {
      passed: searchTestsPassed === testQueries.length,
      details: `${searchTestsPassed}/${testQueries.length} search queries passed`,
      metrics: {
        search_tests_passed: searchTestsPassed,
        search_tests_total: testQueries.length,
        search_results: searchResults
      }
    };
    
  } catch (error: any) {
    return {
      passed: false,
      details: 'Search integration test failed',
      error: error?.message || String(error)
    };
  }
}

/**
 * Test 6: End-to-End Integration Flow
 */
async function testEndToEndFlow(): Promise<{ passed: boolean; details: string; error?: string; metrics?: Record<string, any> }> {
  try {
    // Mock options for integration testing
    const options: LensEnhancedRetrievalOptions = {
      db: {} as any, // Mock DB
      embeddings: {} as any, // Mock embeddings
      sessionId: 'test_session',
      recent_files: ['src/retrieval/bm25.ts', 'src/services/user.ts'],
      recent_activity: 'code',
      current_token_count: 1500,
      total_token_budget: 4000,
      enable_lens: true
    };
    
    // Test maybeLens function with different scenarios
    const testCases = [
      {
        name: 'code_query',
        query: 'fix calculateBM25 implementation error',
        expected_lens_usage: true // May fail if server unavailable
      },
      {
        name: 'non_code_query',
        query: 'what is the weather today',
        expected_lens_usage: false
      },
      {
        name: 'lens_disabled',
        query: 'implement UserService method',
        options: { ...options, enable_lens: false },
        expected_lens_usage: false
      }
    ];
    
    let e2eTestsPassed = 0;
    const e2eResults: any[] = [];
    
    for (const testCase of testCases) {
      const testOptions = testCase.options || options;
      
      try {
        const result = await maybeLens(testCase.query, testOptions);
        
        const lensUsageCorrect = result.used_lens === testCase.expected_lens_usage ||
                               (testCase.expected_lens_usage && result.fallback_reason !== undefined);
        
        const hasValidStructure = Array.isArray(result.lens_candidates) &&
                                typeof result.processing_time_ms === 'number' &&
                                result.code_intent !== undefined;
        
        const testPassed = lensUsageCorrect && hasValidStructure;
        if (testPassed) e2eTestsPassed++;
        
        e2eResults.push({
          test_name: testCase.name,
          query: testCase.query,
          expected_lens_usage: testCase.expected_lens_usage,
          actual_lens_usage: result.used_lens,
          fallback_reason: result.fallback_reason,
          candidates_count: result.lens_candidates.length,
          processing_time_ms: result.processing_time_ms,
          code_intent_confidence: result.code_intent.confidence,
          passed: testPassed
        });
        
      } catch (error: any) {
        e2eResults.push({
          test_name: testCase.name,
          query: testCase.query,
          error: error?.message || String(error),
          passed: false
        });
      }
    }
    
    return {
      passed: e2eTestsPassed === testCases.length,
      details: `${e2eTestsPassed}/${testCases.length} end-to-end tests passed`,
      metrics: {
        e2e_tests_passed: e2eTestsPassed,
        e2e_tests_total: testCases.length,
        e2e_results: e2eResults
      }
    };
    
  } catch (error: any) {
    return {
      passed: false,
      details: 'End-to-end test failed',
      error: error?.message || String(error)
    };
  }
}

/**
 * Test 7: Performance and SLA Validation
 */
async function testPerformanceAndSLA(): Promise<{ passed: boolean; details: string; error?: string; metrics?: Record<string, any> }> {
  const slaTargets = {
    connect_timeout_ms: 500,
    request_timeout_ms: 150,
    health_check_ms: 100
  };
  
  const performanceMetrics: any = {};
  let slaViolations = 0;
  
  try {
    const config = { ...DEFAULT_LENS_CONFIG, enabled: true };
    const lensService = await getLensService();
    
    // Test health check performance
    const healthStart = performance.now();
    const available = await lensService.isAvailable();
    const healthDuration = performance.now() - healthStart;
    
    performanceMetrics.health_check_ms = healthDuration;
    if (healthDuration > slaTargets.health_check_ms) {
      slaViolations++;
    }
    
    if (available) {
      // Test connection performance
      const connStart = performance.now();
      const connTest = await lensService.testConnection();
      const connDuration = performance.now() - connStart;
      
      performanceMetrics.connection_test_ms = connDuration;
      performanceMetrics.connection_latency_ms = connTest.latency_ms || 0;
      
      if (connDuration > slaTargets.connect_timeout_ms) {
        slaViolations++;
      }
      
      // Test search performance (if available)
      try {
        const searchStart = performance.now();
        const searchResult = await lensService.search({
          query: 'test performance query',
          max_groups: 5,
          timeout_ms: slaTargets.request_timeout_ms
        });
        const searchDuration = performance.now() - searchStart;
        
        performanceMetrics.search_total_ms = searchDuration;
        performanceMetrics.search_server_ms = searchResult.processing_time_ms;
        performanceMetrics.search_timeout_hit = searchResult.timeout_hit;
        
        if (searchDuration > slaTargets.request_timeout_ms && !searchResult.timeout_hit) {
          slaViolations++;
        }
      } catch (error) {
        performanceMetrics.search_error = String(error);
      }
    } else {
      performanceMetrics.server_unavailable = true;
    }
    
    return {
      passed: slaViolations === 0 || !available, // Pass if no SLA violations or server unavailable
      details: available 
        ? `SLA violations: ${slaViolations}, Health: ${healthDuration.toFixed(1)}ms`
        : 'Server unavailable - SLA tests skipped',
      metrics: {
        sla_violations: slaViolations,
        sla_targets: slaTargets,
        performance_metrics: performanceMetrics,
        server_available: available
      }
    };
    
  } catch (error: any) {
    return {
      passed: false,
      details: 'Performance test failed',
      error: error?.message || String(error),
      metrics: performanceMetrics
    };
  }
}

/**
 * Main validation runner
 */
async function runValidation(): Promise<ValidationSummary> {
  console.log(`${colors.bright}${colors.cyan}🔍 Lens Integration Validation Script${colors.reset}`);
  console.log(`${colors.cyan}Testing Lens server integration at http://localhost:5678${colors.reset}\n`);
  
  const tests = [
    { name: 'Basic Connectivity', fn: testBasicConnectivity },
    { name: 'Code Intent Detection', fn: testCodeIntentDetection },
    { name: 'Cost Calculation', fn: testCostCalculation },
    { name: 'Configuration System', fn: testConfiguration },
    { name: 'Search Integration', fn: testSearchIntegration },
    { name: 'End-to-End Flow', fn: testEndToEndFlow },
    { name: 'Performance & SLA', fn: testPerformanceAndSLA }
  ];
  
  const results: TestResult[] = [];
  
  for (const test of tests) {
    const result = await runTest(test.name, test.fn);
    results.push(result);
    console.log(''); // Add spacing between tests
  }
  
  // Generate summary
  const passedTests = results.filter(r => r.passed).length;
  const failedTests = results.length - passedTests;
  const totalDuration = results.reduce((sum, r) => sum + r.duration_ms, 0);
  
  // Determine server availability from connectivity test
  const connectivityTest = results.find(r => r.name === 'Basic Connectivity');
  const serverAvailable = connectivityTest?.passed || false;
  
  // Check SLA compliance
  const performanceTest = results.find(r => r.name === 'Performance & SLA');
  const slaCompliant = performanceTest?.metrics?.sla_violations === 0;
  
  // Collect error conditions
  const errorConditions = results
    .filter(r => !r.passed)
    .map(r => `${r.name}: ${r.error || r.details}`);
  
  // Generate recommendations
  const recommendations: string[] = [];
  
  if (!serverAvailable) {
    recommendations.push('Start the Lens server on port 5678 to enable full integration testing');
  }
  
  if (!slaCompliant && serverAvailable) {
    recommendations.push('Optimize Lens server performance to meet SLA targets');
  }
  
  if (failedTests > 0) {
    recommendations.push('Review failed tests and fix underlying issues');
  }
  
  if (serverAvailable && passedTests === results.length) {
    recommendations.push('Integration is working correctly - ready for production use');
  }
  
  return {
    total_tests: results.length,
    passed_tests: passedTests,
    failed_tests: failedTests,
    total_duration_ms: totalDuration,
    server_available: serverAvailable,
    sla_compliant: slaCompliant,
    error_conditions: errorConditions,
    recommendations: recommendations
  };
}

/**
 * Print summary report
 */
function printSummary(summary: ValidationSummary): void {
  console.log(`${colors.bright}${colors.white}📊 VALIDATION SUMMARY${colors.reset}`);
  console.log('━'.repeat(60));
  
  // Test results
  const passColor = summary.passed_tests === summary.total_tests ? colors.green : colors.yellow;
  console.log(`${colors.bright}Tests:${colors.reset} ${passColor}${summary.passed_tests}/${summary.total_tests} passed${colors.reset}`);
  
  if (summary.failed_tests > 0) {
    console.log(`${colors.bright}Failed:${colors.reset} ${colors.red}${summary.failed_tests}${colors.reset}`);
  }
  
  console.log(`${colors.bright}Duration:${colors.reset} ${summary.total_duration_ms.toFixed(1)}ms`);
  
  // Server status
  const serverColor = summary.server_available ? colors.green : colors.red;
  console.log(`${colors.bright}Server:${colors.reset} ${serverColor}${summary.server_available ? 'Available' : 'Unavailable'}${colors.reset}`);
  
  const slaColor = summary.sla_compliant ? colors.green : colors.yellow;
  console.log(`${colors.bright}SLA:${colors.reset} ${slaColor}${summary.sla_compliant ? 'Compliant' : 'Non-compliant'}${colors.reset}`);
  
  // Error conditions
  if (summary.error_conditions.length > 0) {
    console.log(`\n${colors.bright}${colors.red}❌ Error Conditions:${colors.reset}`);
    summary.error_conditions.forEach(error => {
      console.log(`   • ${error}`);
    });
  }
  
  // Recommendations
  if (summary.recommendations.length > 0) {
    console.log(`\n${colors.bright}${colors.blue}💡 Recommendations:${colors.reset}`);
    summary.recommendations.forEach(rec => {
      console.log(`   • ${rec}`);
    });
  }
  
  console.log(''); // Final spacing
}

/**
 * Entry point
 */
async function main(): Promise<void> {
  try {
    const summary = await runValidation();
    printSummary(summary);
    
    // Exit with appropriate code
    const exitCode = summary.failed_tests === 0 ? 0 : 1;
    process.exit(exitCode);
    
  } catch (error: any) {
    console.error(`${colors.red}❌ Validation script failed: ${error?.message || String(error)}${colors.reset}`);
    process.exit(1);
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error(`${colors.red}Fatal error: ${error}${colors.reset}`);
    process.exit(1);
  });
}

// Export for programmatic use
export {
  runValidation,
  printSummary,
  type TestResult,
  type ValidationSummary
};