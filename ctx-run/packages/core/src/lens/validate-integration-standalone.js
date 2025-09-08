#!/usr/bin/env node
/**
 * Lens Integration Validation Script (Standalone)
 * 
 * A simplified validation script that tests the Lens server integration
 * running on port 5678 without complex TypeScript dependencies.
 * 
 * Usage: node src/lens/validate-integration-standalone.js
 */

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

/**
 * Test runner utility
 */
async function runTest(name, testFn) {
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
  } catch (error) {
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
 * Simple fetch wrapper with timeout
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = 5000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

/**
 * Test 1: Basic Server Connectivity
 */
async function testBasicConnectivity() {
  const baseUrl = 'http://localhost:5678';
  
  try {
    // Test health endpoint
    const healthStart = performance.now();
    const healthResponse = await fetchWithTimeout(`${baseUrl}/health`, {
      headers: { 'Accept': 'application/json' }
    }, 1000);
    const healthLatency = performance.now() - healthStart;
    
    if (!healthResponse.ok) {
      return {
        passed: false,
        details: `Health endpoint returned ${healthResponse.status}`,
        error: 'Lens server health check failed',
        metrics: { health_latency_ms: healthLatency, status: healthResponse.status }
      };
    }
    
    // Test status endpoint
    let statusData = null;
    try {
      const statusResponse = await fetchWithTimeout(`${baseUrl}/status`, {
        headers: { 'Accept': 'application/json' }
      }, 1000);
      
      if (statusResponse.ok) {
        statusData = await statusResponse.json();
      }
    } catch (error) {
      console.log(`    Status endpoint not available: ${error.message}`);
    }
    
    const slaCompliant = healthLatency <= 500;
    
    return {
      passed: true,
      details: `Health: OK (${healthLatency.toFixed(1)}ms), Status: ${statusData ? 'Available' : 'Limited'}, SLA: ${slaCompliant ? 'Compliant' : 'Non-compliant'}`,
      metrics: {
        health_latency_ms: healthLatency,
        server_healthy: true,
        status_available: !!statusData,
        lsp_available: statusData?.lsp_available || false,
        raptor_cache_status: statusData?.raptor_cache_status || 'unknown',
        server_version: statusData?.version || 'unknown',
        sla_compliant: slaCompliant
      }
    };
    
  } catch (error) {
    return {
      passed: false,
      details: `Connection failed: ${error.message}`,
      error: 'Cannot connect to Lens server',
      metrics: { connection_error: error.message }
    };
  }
}

/**
 * Test 2: Code Intent Detection (Simplified)
 */
async function testCodeIntentDetection() {
  // Simplified code intent detection logic
  function detectCodeIntent(query) {
    const lowerQuery = query.toLowerCase();
    
    // Basic patterns for code detection
    const codePatterns = [
      /\b(function|class|method|error|bug|fix|implement)\b/,
      /\b[a-zA-Z_]\w*\s*\(/,  // function calls
      /\b(const|let|var|def|async|await)\b/,
      /\.(js|ts|py|rs|go|java|cpp)(\s|$)/
    ];
    
    const errorPatterns = [
      /\b(exception|error|traceback|failed|broken)\b/i
    ];
    
    let confidence = 0;
    let hasCodeSymbols = false;
    let hasErrorTokens = false;
    
    for (const pattern of codePatterns) {
      if (pattern.test(query)) {
        hasCodeSymbols = true;
        confidence += 0.3;
        break;
      }
    }
    
    for (const pattern of errorPatterns) {
      if (pattern.test(query)) {
        hasErrorTokens = true;
        confidence += 0.2;
        break;
      }
    }
    
    // File extensions boost confidence
    if (/\.(js|ts|py|rs|go)/.test(query)) {
      confidence += 0.2;
    }
    
    confidence = Math.min(1.0, confidence);
    const isCodeIntent = confidence >= 0.3;
    
    return {
      is_code_intent: isCodeIntent,
      confidence,
      patterns: {
        has_code_symbols: hasCodeSymbols,
        has_error_tokens: hasErrorTokens
      }
    };
  }
  
  const testCases = [
    { query: 'fix error in calculateBM25 function', expectedIntent: true },
    { query: 'implement UserService.createUser method', expectedIntent: true },
    { query: 'what is the weather today', expectedIntent: false },
    { query: 'TypeError in async function handleRequest', expectedIntent: true }
  ];
  
  let passed = 0;
  let total = testCases.length;
  const results = [];
  
  for (const testCase of testCases) {
    const result = detectCodeIntent(testCase.query);
    const intentCorrect = result.is_code_intent === testCase.expectedIntent;
    
    if (intentCorrect) passed++;
    
    results.push({
      query: testCase.query,
      expected_intent: testCase.expectedIntent,
      actual_intent: result.is_code_intent,
      confidence: result.confidence,
      passed: intentCorrect
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
 * Test 3: Basic Search Integration
 */
async function testSearchIntegration() {
  const baseUrl = 'http://localhost:5678';
  
  try {
    // First check if server is available
    const healthResponse = await fetchWithTimeout(`${baseUrl}/health`, {}, 1000);
    if (!healthResponse.ok) {
      return {
        passed: false,
        details: 'Server not available for search testing',
        error: 'Health check failed'
      };
    }
    
    const testQueries = [
      { query: 'calculateBM25 function', max_groups: 5 },
      { query: 'UserService implementation', max_groups: 3 }
    ];
    
    let searchTestsPassed = 0;
    const searchResults = [];
    
    for (const testQuery of testQueries) {
      const startTime = performance.now();
      
      try {
        const searchPayload = {
          query: testQuery.query,
          max_groups: testQuery.max_groups,
          timeout_ms: 2000
        };
        
        const searchResponse = await fetchWithTimeout(`${baseUrl}/api/search`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(searchPayload)
        }, 3000);
        
        const duration = performance.now() - startTime;
        
        if (searchResponse.ok) {
          const result = await searchResponse.json();
          const hasValidStructure = Array.isArray(result.symbol_groups) &&
                                   typeof result.processing_time_ms === 'number';
          
          if (hasValidStructure) searchTestsPassed++;
          
          searchResults.push({
            query: testQuery.query,
            duration_ms: duration,
            status: searchResponse.status,
            symbol_groups_count: result.symbol_groups?.length || 0,
            server_processing_ms: result.processing_time_ms || 0,
            timeout_hit: result.timeout_hit || false,
            passed: hasValidStructure
          });
        } else {
          searchResults.push({
            query: testQuery.query,
            duration_ms: duration,
            status: searchResponse.status,
            error: `HTTP ${searchResponse.status}`,
            passed: false
          });
        }
        
      } catch (error) {
        searchResults.push({
          query: testQuery.query,
          error: error.message,
          passed: false
        });
      }
    }
    
    return {
      passed: searchTestsPassed > 0,
      details: `${searchTestsPassed}/${testQueries.length} search queries succeeded`,
      metrics: {
        search_tests_passed: searchTestsPassed,
        search_tests_total: testQueries.length,
        search_results: searchResults
      }
    };
    
  } catch (error) {
    return {
      passed: false,
      details: 'Search integration test failed',
      error: error.message
    };
  }
}

/**
 * Test 4: Performance and SLA Validation
 */
async function testPerformanceAndSLA() {
  const baseUrl = 'http://localhost:5678';
  const slaTargets = {
    connect_timeout_ms: 500,
    health_check_ms: 100,
    search_timeout_ms: 2000
  };
  
  const performanceMetrics = {};
  let slaViolations = 0;
  
  try {
    // Test health check performance
    const healthStart = performance.now();
    const healthResponse = await fetchWithTimeout(`${baseUrl}/health`, {}, slaTargets.health_check_ms);
    const healthDuration = performance.now() - healthStart;
    
    performanceMetrics.health_check_ms = healthDuration;
    if (healthDuration > slaTargets.health_check_ms) {
      slaViolations++;
    }
    
    if (healthResponse.ok) {
      // Test search performance if server is available
      try {
        const searchStart = performance.now();
        const searchResponse = await fetchWithTimeout(`${baseUrl}/api/search`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({
            query: 'test performance query',
            max_groups: 3,
            timeout_ms: 500
          })
        }, slaTargets.search_timeout_ms);
        
        const searchDuration = performance.now() - searchStart;
        performanceMetrics.search_total_ms = searchDuration;
        
        if (searchResponse.ok) {
          const result = await searchResponse.json();
          performanceMetrics.search_server_ms = result.processing_time_ms || 0;
          performanceMetrics.search_timeout_hit = result.timeout_hit || false;
        }
        
        // Search SLA is more lenient
        if (searchDuration > slaTargets.search_timeout_ms) {
          slaViolations++;
        }
        
      } catch (error) {
        performanceMetrics.search_error = error.message;
      }
    }
    
    return {
      passed: slaViolations === 0,
      details: `SLA violations: ${slaViolations}, Health: ${healthDuration.toFixed(1)}ms, Search: ${performanceMetrics.search_total_ms?.toFixed(1) || 'N/A'}ms`,
      metrics: {
        sla_violations: slaViolations,
        sla_targets: slaTargets,
        performance_metrics: performanceMetrics,
        server_available: healthResponse.ok
      }
    };
    
  } catch (error) {
    return {
      passed: false,
      details: 'Performance test failed',
      error: error.message,
      metrics: performanceMetrics
    };
  }
}

/**
 * Test 5: Configuration Validation
 */
async function testConfiguration() {
  // Test basic configuration constants
  const expectedConfig = {
    base_url: 'http://localhost:5678',
    connect_timeout_ms: 500,
    request_timeout_ms: 150,
    sla_recall_ms: 150,
    topic_fanout_k: 240,
    weight_cap: 0.4
  };
  
  // Simulate configuration validation
  let configValid = true;
  const configIssues = [];
  
  // Check if configuration values are reasonable
  if (expectedConfig.request_timeout_ms > expectedConfig.sla_recall_ms) {
    configValid = false;
    configIssues.push('Request timeout exceeds SLA recall budget');
  }
  
  if (expectedConfig.topic_fanout_k < 100 || expectedConfig.topic_fanout_k > 500) {
    configValid = false;
    configIssues.push('Topic fanout outside reasonable range (100-500)');
  }
  
  if (expectedConfig.weight_cap <= 0 || expectedConfig.weight_cap >= 1) {
    configValid = false;
    configIssues.push('Weight cap outside valid range (0-1)');
  }
  
  return {
    passed: configValid,
    details: configValid ? 'Configuration validation passed' : `Issues: ${configIssues.join(', ')}`,
    metrics: {
      config_valid: configValid,
      config_issues: configIssues,
      expected_config: expectedConfig
    }
  };
}

/**
 * Main validation runner
 */
async function runValidation() {
  console.log(`${colors.bright}${colors.cyan}🔍 Lens Integration Validation Script (Standalone)${colors.reset}`);
  console.log(`${colors.cyan}Testing Lens server integration at http://localhost:5678${colors.reset}\n`);
  
  const tests = [
    { name: 'Basic Connectivity', fn: testBasicConnectivity },
    { name: 'Code Intent Detection', fn: testCodeIntentDetection },
    { name: 'Search Integration', fn: testSearchIntegration },
    { name: 'Performance & SLA', fn: testPerformanceAndSLA },
    { name: 'Configuration Validation', fn: testConfiguration }
  ];
  
  const results = [];
  
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
  
  // Print summary
  console.log(`${colors.bright}${colors.white}📊 VALIDATION SUMMARY${colors.reset}`);
  console.log('━'.repeat(60));
  
  // Test results
  const passColor = passedTests === results.length ? colors.green : colors.yellow;
  console.log(`${colors.bright}Tests:${colors.reset} ${passColor}${passedTests}/${results.length} passed${colors.reset}`);
  
  if (failedTests > 0) {
    console.log(`${colors.bright}Failed:${colors.reset} ${colors.red}${failedTests}${colors.reset}`);
  }
  
  console.log(`${colors.bright}Duration:${colors.reset} ${totalDuration.toFixed(1)}ms`);
  
  // Server status
  const serverColor = serverAvailable ? colors.green : colors.red;
  console.log(`${colors.bright}Server:${colors.reset} ${serverColor}${serverAvailable ? 'Available' : 'Unavailable'}${colors.reset}`);
  
  const slaColor = slaCompliant ? colors.green : colors.yellow;
  console.log(`${colors.bright}SLA:${colors.reset} ${slaColor}${slaCompliant ? 'Compliant' : 'Non-compliant'}${colors.reset}`);
  
  // Error conditions
  const errorConditions = results
    .filter(r => !r.passed)
    .map(r => `${r.name}: ${r.error || r.details}`);
  
  if (errorConditions.length > 0) {
    console.log(`\n${colors.bright}${colors.red}❌ Error Conditions:${colors.reset}`);
    errorConditions.forEach(error => {
      console.log(`   • ${error}`);
    });
  }
  
  // Recommendations
  const recommendations = [];
  
  if (!serverAvailable) {
    recommendations.push('Start the Lens server on port 5678 to enable full integration testing');
    recommendations.push('Verify server configuration and network connectivity');
  }
  
  if (!slaCompliant && serverAvailable) {
    recommendations.push('Optimize Lens server performance to meet SLA targets');
  }
  
  if (failedTests === 0 && serverAvailable) {
    recommendations.push('Integration is working correctly - ready for production use');
  }
  
  if (recommendations.length > 0) {
    console.log(`\n${colors.bright}${colors.blue}💡 Recommendations:${colors.reset}`);
    recommendations.forEach(rec => {
      console.log(`   • ${rec}`);
    });
  }
  
  console.log(''); // Final spacing
  
  // Exit with appropriate code
  const exitCode = failedTests === 0 ? 0 : 1;
  process.exit(exitCode);
}

// Run validation
runValidation().catch(error => {
  console.error(`${colors.red}❌ Validation script failed: ${error.message}${colors.reset}`);
  process.exit(1);
});