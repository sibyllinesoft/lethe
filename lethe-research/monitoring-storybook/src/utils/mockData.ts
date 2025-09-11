import { 
  TransformChangeV2, 
  TokenMetrics, 
  KVMetrics, 
  DifficultyGateMetrics,
  ChangeType
} from '../types/transform';

/**
 * Mock data generator for Transform Diff Panel stories
 * Generates realistic data that matches the TransformChangeV2 schema
 */

// Helper function to generate random dates within a range
function randomDate(start: Date, end: Date): Date {
  return new Date(start.getTime() + Math.random() * (end.getTime() - start.getTime()));
}

// Helper function to generate random request IDs
function generateRequestId(): string {
  return `req_${Math.random().toString(36).substr(2, 8)}`;
}

// Generate sample change types with realistic data
function generateChangeType(type: 'system' | 'user' | 'legacy' | 'validation' | 'no_change'): ChangeType {
  switch (type) {
    case 'system':
      return {
        system_prelude_added: {
          position: Math.random() > 0.5 ? 'beginning' : 'end',
          content_hash: `hash_${Math.random().toString(36).substr(2, 8)}`
        }
      };
    
    case 'user':
      return {
        user_content_rewritten: {
          rewrite_type: ['language_normalization', 'format_standardization', 'content_expansion'][
            Math.floor(Math.random() * 3)
          ] as any,
          affected_messages: Math.floor(Math.random() * 5) + 1,
          content_size_change: Math.floor((Math.random() - 0.5) * 200)
        }
      };
    
    case 'legacy':
      return {
        legacy_prompt_rewritten: {
          conversion_type: ['prompt_to_messages', 'completion_to_chat'][
            Math.floor(Math.random() * 2)
          ] as any,
          prompt_length: Math.floor(Math.random() * 1000) + 100
        }
      };
    
    case 'validation':
      return {
        validation_applied: {
          validation_type: ['schema_validation', 'content_validation', 'security_validation'][
            Math.floor(Math.random() * 3)
          ] as any,
          validation_result: Math.random() > 0.8 ? 
            { failed: ['Validation error occurred'] } :
            { passed: null }
        }
      };
    
    case 'no_change':
    default:
      return {
        no_changes_applied: {
          reason: ['already_optimal', 'rewriting_disabled', 'insufficient_data'][
            Math.floor(Math.random() * 3)
          ] as any
        }
      };
  }
}

// Generate a single transform change
export function generateTransformChange(
  timestamp?: Date,
  provider?: string,
  requestId?: string
): TransformChangeV2 {
  const changeTypes: Array<'system' | 'user' | 'legacy' | 'validation' | 'no_change'> = [
    'system', 'user', 'legacy', 'validation', 'no_change'
  ];
  
  const randomType = changeTypes[Math.floor(Math.random() * changeTypes.length)];
  
  return {
    schema_version: 2,
    timestamp: (timestamp || randomDate(new Date(Date.now() - 24 * 60 * 60 * 1000), new Date())).toISOString(),
    change_type: generateChangeType(randomType),
    metadata: {
      performance: {
        duration_us: Math.floor(Math.random() * 50000) + 1000, // 1-50ms
        memory_used_bytes: Math.floor(Math.random() * 1000000) + 100000, // 100KB-1MB
        cpu_cycles: Math.floor(Math.random() * 1000000),
        input_size_bytes: Math.floor(Math.random() * 5000) + 500, // 500B-5KB
        output_size_bytes: Math.floor(Math.random() * 5500) + 400 // 400B-5.5KB
      },
      config_version: '1.0.0',
      config_hash: `config_${Math.random().toString(36).substr(2, 8)}`,
      confidence_score: Math.random() > 0.1 ? Math.random() * 0.3 + 0.7 : Math.random() * 0.5, // Mostly high confidence
      quality_score: Math.random() * 0.4 + 0.6, // 0.6-1.0
      reversible: Math.random() > 0.3,
      rollback_data: Math.random() > 0.7 ? `rollback_${Math.random().toString(36).substr(2, 8)}` : undefined
    },
    context: {
      request_id: requestId || generateRequestId(),
      target_provider: provider || ['openai', 'anthropic', 'google', 'cohere'][Math.floor(Math.random() * 4)],
      target_model: ['gpt-4', 'claude-3', 'gemini-pro', 'command-r'][Math.floor(Math.random() * 4)],
      user_context: `user_${Math.random().toString(36).substr(2, 6)}`,
      feature_flags: Math.random() > 0.5 ? ['enhanced_rewriting', 'fast_mode'] : [],
      experiment_bucket: Math.random() > 0.7 ? `bucket_${Math.random().toString(36).substr(2, 2)}` : undefined
    }
  };
}

// Generate multiple transform changes
export function generateTransformChanges(count: number, timeSpanHours = 24): TransformChangeV2[] {
  const endTime = new Date();
  const startTime = new Date(endTime.getTime() - timeSpanHours * 60 * 60 * 1000);
  
  const providers = ['openai', 'anthropic', 'google', 'cohere'];
  const requestIds = Array.from({ length: Math.min(count / 3, 20) }, () => generateRequestId());
  
  return Array.from({ length: count }, () => {
    const timestamp = randomDate(startTime, endTime);
    const provider = providers[Math.floor(Math.random() * providers.length)];
    const requestId = requestIds[Math.floor(Math.random() * requestIds.length)];
    
    return generateTransformChange(timestamp, provider, requestId);
  });
}

// Generate token metrics
export function generateTokenMetrics(changes: TransformChangeV2[]): TokenMetrics {
  if (changes.length === 0) {
    return {
      before: 1000,
      after: 950,
      delta: -50,
      efficiency: 0.95
    };
  }

  // Calculate from actual changes
  const totalInputBytes = changes.reduce((sum, change) => 
    sum + (change.metadata.performance?.input_size_bytes || 0), 0);
  const totalOutputBytes = changes.reduce((sum, change) => 
    sum + (change.metadata.performance?.output_size_bytes || 0), 0);

  // Estimate tokens (roughly 4 bytes per token)
  const before = Math.floor(totalInputBytes / 4) || 1000;
  const after = Math.floor(totalOutputBytes / 4) || 950;
  const delta = after - before;
  const efficiency = before > 0 ? after / before : 1;

  return {
    before,
    after,
    delta,
    efficiency
  };
}

// Generate KV metrics
export function generateKVMetrics(changes: TransformChangeV2[]): KVMetrics {
  // Simulate Jaccard similarity based on change diversity
  const changeTypes = new Set(changes.map(c => Object.keys(c.change_type)[0]));
  const diversity = changeTypes.size / Math.max(changes.length / 10, 1);
  const prefixJaccard = Math.max(0, Math.min(1, 1 - diversity * 0.3));

  // Simulate volatility based on performance variance
  const durations = changes.map(c => c.metadata.performance?.duration_us || 0);
  const avgDuration = durations.reduce((sum, d) => sum + d, 0) / durations.length || 0;
  const variance = durations.reduce((sum, d) => sum + Math.pow(d - avgDuration, 2), 0) / durations.length || 0;
  const volatility = Math.min(1, Math.sqrt(variance) / 50000); // Normalize to 0-1

  // Count head and tail edits
  const headEdits = changes.filter(c => 
    Object.keys(c.change_type)[0].includes('system') || 
    Object.keys(c.change_type)[0].includes('prelude')
  ).length;

  const tailEdits = changes.filter(c => 
    Object.keys(c.change_type)[0].includes('content') || 
    Object.keys(c.change_type)[0].includes('enhanced')
  ).length;

  return {
    prefixJaccard,
    volatility,
    headEdits,
    tailEdits
  };
}

// Generate difficulty gate metrics
export function generateDifficultyGateMetrics(changes: TransformChangeV2[]): DifficultyGateMetrics {
  if (changes.length === 0) {
    return {
      changeEntropy: 1.5,
      rollbackFrequency: 0.05,
      editDepth: 0.2,
      complexityScore: 0.4,
      recommendedK2Cap: 2048,
      recommendedDimension: 256
    };
  }

  // Calculate Shannon entropy of change types
  const changeTypeFreq = new Map<string, number>();
  changes.forEach(change => {
    const changeType = Object.keys(change.change_type)[0];
    changeTypeFreq.set(changeType, (changeTypeFreq.get(changeType) || 0) + 1);
  });

  let entropy = 0;
  const totalChanges = changes.length;
  changeTypeFreq.forEach(freq => {
    const probability = freq / totalChanges;
    entropy -= probability * Math.log2(probability);
  });

  // Calculate rollback frequency (changes with low confidence)
  const lowConfidenceChanges = changes.filter(c => 
    c.metadata.confidence_score !== undefined && c.metadata.confidence_score < 0.5
  ).length;
  const rollbackFrequency = lowConfidenceChanges / totalChanges;

  // Calculate average edit depth
  const editDepths = changes
    .map(change => {
      const perf = change.metadata.performance;
      if (!perf || perf.input_size_bytes === 0) return 0;
      return Math.abs(perf.output_size_bytes - perf.input_size_bytes) / perf.input_size_bytes;
    })
    .filter(depth => depth > 0);

  const editDepth = editDepths.length > 0 
    ? editDepths.reduce((sum, depth) => sum + depth, 0) / editDepths.length 
    : 0;

  // Calculate composite complexity score
  const complexityScore = (entropy * 0.4) + (rollbackFrequency * 0.3) + (editDepth * 0.3);

  // Generate recommendations based on complexity
  let recommendedK2Cap: number;
  let recommendedDimension: 256 | 768;

  if (complexityScore > 0.7) {
    recommendedK2Cap = 1024;
    recommendedDimension = 768;
  } else if (complexityScore > 0.5) {
    recommendedK2Cap = 2048;
    recommendedDimension = 768;
  } else if (complexityScore > 0.3) {
    recommendedK2Cap = 2048;
    recommendedDimension = 256;
  } else {
    recommendedK2Cap = 4096;
    recommendedDimension = 256;
  }

  return {
    changeEntropy: entropy,
    rollbackFrequency,
    editDepth,
    complexityScore,
    recommendedK2Cap,
    recommendedDimension
  };
}

// Preset data scenarios for stories
export const mockDataScenarios = {
  // Minimal dataset for basic testing
  minimal: {
    changes: generateTransformChanges(5, 1),
    get tokenMetrics() { return generateTokenMetrics(this.changes); },
    get kvMetrics() { return generateKVMetrics(this.changes); },
    get difficultyMetrics() { return generateDifficultyGateMetrics(this.changes); }
  },

  // Standard dataset for typical usage
  standard: {
    changes: generateTransformChanges(50, 4),
    get tokenMetrics() { return generateTokenMetrics(this.changes); },
    get kvMetrics() { return generateKVMetrics(this.changes); },
    get difficultyMetrics() { return generateDifficultyGateMetrics(this.changes); }
  },

  // Large dataset for performance testing
  large: {
    changes: generateTransformChanges(500, 24),
    get tokenMetrics() { return generateTokenMetrics(this.changes); },
    get kvMetrics() { return generateKVMetrics(this.changes); },
    get difficultyMetrics() { return generateDifficultyGateMetrics(this.changes); }
  },

  // High complexity scenario
  highComplexity: {
    changes: (() => {
      const changes = generateTransformChanges(100, 2);
      // Inject some high complexity scenarios
      changes.forEach((change, index) => {
        if (index % 10 === 0) {
          // Add some failures and low confidence scores
          change.metadata.confidence_score = Math.random() * 0.3;
        }
        if (index % 7 === 0) {
          // Add some high variance in performance
          change.metadata.performance!.duration_us = Math.random() * 100000 + 50000;
        }
      });
      return changes;
    })(),
    get tokenMetrics() { return generateTokenMetrics(this.changes); },
    get kvMetrics() { return generateKVMetrics(this.changes); },
    get difficultyMetrics() { return generateDifficultyGateMetrics(this.changes); }
  },

  // Real-time simulation data
  realTime: {
    changes: generateTransformChanges(20, 0.5),
    get tokenMetrics() { return generateTokenMetrics(this.changes); },
    get kvMetrics() { return generateKVMetrics(this.changes); },
    get difficultyMetrics() { return generateDifficultyGateMetrics(this.changes); }
  }
};

// Function to add new changes for real-time simulation
export function addRealtimeChange(existing: TransformChangeV2[]): TransformChangeV2[] {
  const newChange = generateTransformChange();
  return [...existing, newChange];
}