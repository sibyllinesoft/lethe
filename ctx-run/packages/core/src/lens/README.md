# Lens Search Server Integration

This module implements the Lens search server integration for Lethe, providing symbol-aware code search capabilities as described in the project TODO.md. Lens acts as an optional "S± stage" between traditional retrieval stages, offering precise symbol matching with semantic reachability.

## Overview

The Lens integration provides:

- **Symbol-centric search**: LSP-precise symbol definitions, references, and implementations
- **Code intent detection**: Automatic detection of code-focused queries
- **Lagrangian cost control**: Token and compute cost optimization with SLA constraints
- **Fault-tolerant design**: Graceful fallbacks when Lens server is unavailable
- **Non-breaking integration**: Can be enabled/disabled without affecting existing functionality

## Key Components

### Core Interfaces

- **`SymbolGroup`**: Represents a cohesive set of code symbols with definitions, references, and implementations
- **`CodeAtom`**: Individual code pieces with precise location information
- **`LensService`**: Main service interface for interacting with the Lens search server
- **`LensConfig`**: Configuration options for Lens integration

### Main Functions

- **`detectCodeIntent()`**: Analyzes queries to determine if they are code-focused
- **`calculateLagrangianCost()`**: Performs cost-benefit analysis with λ/μ multipliers
- **`getLensService()`**: Factory function for creating Lens service instances
- **`maybeLens()`**: Main integration function implementing the "MAYBE_LENS" concept

## Configuration

Add Lens configuration to your `ctx.config.json`:

```json
{
  "lens": {
    "enabled": true,
    "base_url": "http://localhost:8081",
    "sla_recall_ms": 150,
    "topic_fanout_k": 240,
    "weight_cap": 0.4,
    "mode": "auto",
    "lambda_multiplier": 1.2,
    "mu_multiplier": 1.0,
    "lens_tokens_cap": 4000
  }
}
```

### Configuration Profiles

The integration supports different profiles based on context size:

- **Small context** (≤4k tokens): `auto` mode with aggressive Lens usage
- **Medium context**: Balanced approach with moderate token caps
- **Large context** (~100k tokens): `earn-its-place` mode with higher quality bar

## Usage

### Basic Integration

```typescript
import { getLensService, detectCodeIntent, maybeLens } from '@lethe/core';

// Check if query is code-focused
const codeIntent = detectCodeIntent('fix error in calculateBM25 function');
if (codeIntent.is_code_intent) {
  console.log(`Code intent detected with ${(codeIntent.confidence * 100).toFixed(1)}% confidence`);
}

// Perform Lens-enhanced retrieval
const lensResult = await maybeLens(query, {
  db,
  embeddings,
  sessionId: 'my_session',
  recent_files: ['src/retrieval/index.ts'],
  recent_activity: 'code',
  current_token_count: 1000,
  total_token_budget: 4000
});

if (lensResult.used_lens) {
  console.log(`Lens found ${lensResult.lens_candidates.length} symbol candidates`);
} else {
  console.log(`Lens skipped: ${lensResult.fallback_reason}`);
}
```

### Integration with Existing Retrieval

```typescript
import { lensEnhancedHybridRetrieval } from '@lethe/core';

const result = await lensEnhancedHybridRetrieval(queries, {
  db,
  embeddings, 
  sessionId: 'session_id',
  enable_lens: true,
  recent_files: ['src/main.ts'],
  recent_activity: 'code'
});

// Results combine traditional retrieval + Lens symbol groups
console.log(`Total candidates: ${result.candidates.length}`);
console.log(`Lens contributed: ${result.lens_contribution.candidates_count}`);
```

## Code Intent Detection

The system automatically detects code-focused queries using pattern matching:

### Detected Patterns

- **Function calls**: `functionName()`, `Class::method()`
- **Error patterns**: `Exception`, `Error`, `E404`, `stack trace`
- **File paths**: `/src/file.ts`, `C:\\path\\file.js`
- **Language keywords**: `import`, `class`, `function`, `async`
- **Library patterns**: `std::`, `np.`, `torch.`

### Example Classifications

```typescript
detectCodeIntent('fix bug in getUserData function');
// → { is_code_intent: true, confidence: 0.85, detected_language: 'javascript' }

detectCodeIntent('what is the weather today');
// → { is_code_intent: false, confidence: 0.1 }

detectCodeIntent('TypeError in src/utils.ts line 42');
// → { is_code_intent: true, confidence: 0.9, patterns: { has_error_tokens: true } }
```

## Cost Analysis

The Lagrangian cost controller enforces token and compute budgets:

### Cost Components

- **Token cost**: `λ × lens_tokens` (with lambda multiplier)
- **Compute cost**: `μ × (ce_cost + dpp_cost)` (with mu multiplier)
- **SLA constraint**: Hard limit at 150ms for SLA-Recall requirement

### Example Analysis

```typescript
const costResult = calculateLagrangianCost(
  symbolGroups,
  config,
  currentTokens: 1000,
  totalBudget: 4000,
  estimatedLatencyMs: 120
);

if (costResult.cost_acceptable && costResult.sla_constraint_met) {
  console.log('✅ Cost analysis passed, proceeding with Lens integration');
} else {
  console.log('❌ Cost constraints not met, skipping Lens');
}
```

## Architecture

### S± Stage Integration

Lens integrates as an optional stage between traditional retrieval phases:

```
Query → Code Intent Detection → MAYBE_LENS → Traditional Retrieval → Fusion
         ↓                      ↓
    Not code-intent         LSP + RAPTOR Search
         ↓                      ↓
      Skip Lens             Symbol Groups
                                ↓
                         Cost Analysis
                                ↓
                        Retrieval Candidates
```

### Fault Tolerance

The integration includes multiple fallback mechanisms:

1. **Server unavailability**: Skip Lens, continue with traditional retrieval
2. **Timeout exceeded**: Return partial results or fallback
3. **Cost constraints**: Reject Lens results if they don't meet budget
4. **Quality degradation**: Reduce topic fanout and DPP rank

## Performance Characteristics

- **Target latency**: p95 ≤ 87ms (production baseline)
- **SLA constraint**: Hard limit at 150ms
- **Token efficiency**: 2-4k token packs for small context
- **Quality targets**: ≥98% answer span preservation, +10% nDCG@10

## Testing

The module includes comprehensive test suites:

- **Basic tests**: Core functionality, code intent detection, cost analysis
- **Comprehensive tests**: HTTP client, configuration, production scenarios
- **Integration examples**: Real-world usage patterns

Run tests:
```bash
npm test lens-basic.test.ts
npm test lens-comprehensive.test.ts
```

## Monitoring and Observability

The integration provides detailed metrics:

- **Processing time**: Lens search latency tracking
- **Cost analysis**: Token and compute cost breakdown  
- **Quality metrics**: LSP availability, topic expansion count
- **Fallback tracking**: Reasons for Lens bypassing

## Security Considerations

- **Input validation**: All queries are validated and sanitized
- **Timeout enforcement**: Hard timeouts prevent resource exhaustion
- **Error isolation**: Lens failures don't affect traditional retrieval
- **Configuration validation**: All config parameters are bounds-checked

## Future Enhancements

Planned improvements include:

- **Caching**: Symbol group caching for repeated queries
- **Load balancing**: Multiple Lens server support
- **Metrics aggregation**: Enhanced monitoring and alerting
- **A/B testing**: Canary deployment support for Lens features

## Troubleshooting

Common issues and solutions:

### Lens Server Not Available
```
Error: Lens server not available
Solution: Check server status at http://localhost:8081/api/health
```

### SLA Budget Exhausted
```
Warning: Insufficient SLA budget remaining
Solution: Increase sla_recall_ms or optimize query processing
```

### High Cost Rejection
```
Warning: Cost constraints not met
Solution: Adjust lambda_multiplier or increase token budget
```

### Low Code Intent Confidence
```
Info: Not code-intent query (confidence: 0.2)
Solution: Verify query contains programming-related terms
```

## References

- **TODO.md**: Original specification and requirements
- **Lens paper**: LSP precision spine + RAPTOR semantic bridging
- **Production metrics**: p95≈87ms, SLA-Recall@150ms constraints