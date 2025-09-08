# Lens Integration Validation Scripts

This directory contains validation scripts to test the Lens server integration running on port 5678.

## Available Scripts

### 1. Full TypeScript Validation (`validate-integration.ts`)

Comprehensive validation script with full TypeScript integration testing.

**Features:**
- Complete integration with existing Lens service classes
- Detailed cost calculation testing with mock symbol groups  
- End-to-end flow testing with maybeLens integration
- Advanced performance metrics and SLA validation
- Full test coverage of all integration components

**Usage:**
```bash
# From the core package directory
npx tsx src/lens/validate-integration.ts

# Or if tsx is installed globally
tsx src/lens/validate-integration.ts
```

**Requirements:**
- TypeScript environment with tsx
- All package dependencies installed
- Access to Lens service modules

### 2. Standalone JavaScript Validation (`validate-integration-standalone.js`)

Simplified validation script that runs independently without complex dependencies.

**Features:**
- Direct HTTP testing of Lens server endpoints
- Simplified code intent detection logic
- Basic search integration testing
- Performance and SLA validation
- Configuration validation
- No external dependencies except Node.js fetch API

**Usage:**
```bash
# From the core package directory  
node src/lens/validate-integration-standalone.js

# Or make it executable and run directly
chmod +x src/lens/validate-integration-standalone.js
./src/lens/validate-integration-standalone.js
```

**Requirements:**
- Node.js 18+ (for fetch API)
- No additional dependencies

## Test Categories

Both scripts test the following categories:

### 1. **Basic Connectivity**
- Health endpoint response (`/api/health`)
- Status endpoint response (`/api/status`) 
- Connection latency measurement
- Server availability verification

### 2. **Code Intent Detection**
- Pattern matching for code-related queries
- Confidence scoring algorithms
- Test cases for various query types
- Validation of intent detection accuracy

### 3. **Search Integration** 
- Search endpoint functionality (`/api/search`)
- Request/response structure validation
- Search performance measurement
- Timeout handling verification

### 4. **Performance & SLA Validation**
- Response time measurements
- SLA compliance checking
- Timeout threshold validation
- Performance regression detection

### 5. **Configuration System**
- Default configuration validation
- Parameter range checking  
- Configuration consistency verification
- Profile settings validation

### 6. **Cost Calculation** (Full version only)
- Lagrangian cost model testing
- Token and compute cost calculation
- Cost-benefit ratio analysis
- SLA constraint validation

### 7. **End-to-End Flow** (Full version only)
- Complete integration workflow testing
- maybeLens function validation
- Fallback scenario testing
- Integration pipeline verification

## Expected Output

### When Lens Server is Available
```
🔍 Lens Integration Validation Script
Testing Lens server integration at http://localhost:5678

[PASSED] Basic Connectivity (45.2ms)
  Health: OK (42.1ms), Status: Available, SLA: Compliant

[PASSED] Code Intent Detection (2.1ms)  
  4/4 test cases passed

[PASSED] Search Integration (156.7ms)
  2/2 search queries succeeded

[PASSED] Performance & SLA (123.4ms)
  SLA violations: 0, Health: 45.2ms, Search: 156.7ms

[PASSED] Configuration Validation (0.8ms)
  Configuration validation passed

📊 VALIDATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests: 5/5 passed
Duration: 328.2ms
Server: Available  
SLA: Compliant

💡 Recommendations:
   • Integration is working correctly - ready for production use
```

### When Lens Server is Unavailable
```
🔍 Lens Integration Validation Script  
Testing Lens server integration at http://localhost:5678

[FAILED] Basic Connectivity (83.0ms)
  Health endpoint returned 404
  Error: Lens server health check failed

[PASSED] Code Intent Detection (1.9ms)
  4/4 test cases passed

[FAILED] Search Integration (5.9ms) 
  Server not available for search testing
  Error: Health check failed

[PASSED] Performance & SLA (3.6ms)
  SLA violations: 0, Health: 3.3ms, Search: N/Ams

[PASSED] Configuration Validation (0.3ms)
  Configuration validation passed

📊 VALIDATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests: 3/5 passed
Failed: 2
Duration: 94.8ms  
Server: Unavailable
SLA: Compliant

❌ Error Conditions:
   • Basic Connectivity: Lens server health check failed
   • Search Integration: Health check failed

💡 Recommendations:
   • Start the Lens server on port 5678 to enable full integration testing
   • Verify server configuration and network connectivity
```

## Exit Codes

- **0**: All tests passed successfully
- **1**: One or more tests failed

## Integration with CI/CD

The validation scripts can be integrated into CI/CD pipelines:

```bash
# Example CI script
if node src/lens/validate-integration-standalone.js; then
  echo "Lens integration validation passed"
else
  echo "Lens integration validation failed"
  exit 1
fi
```

## Troubleshooting

### Common Issues

1. **"Connection refused" errors**
   - Ensure Lens server is running on port 5678
   - Check firewall settings
   - Verify server startup logs

2. **"404 Not Found" on endpoints**  
   - Check Lens server version compatibility
   - Verify API endpoint paths
   - Review server routing configuration

3. **Timeout errors**
   - Adjust timeout settings in validation script
   - Check server performance and load
   - Verify network latency

4. **TypeScript compilation errors** (Full version)
   - Ensure all dependencies are installed
   - Check TypeScript configuration
   - Verify import paths and module resolution

### Server Startup

To start the Lens server for testing:

```bash
# Navigate to Lens server directory
cd /path/to/lens-server

# Install dependencies  
npm install

# Start development server
npm run dev

# Or start production server
npm start
```

The server should be accessible at `http://localhost:5678` when running correctly.