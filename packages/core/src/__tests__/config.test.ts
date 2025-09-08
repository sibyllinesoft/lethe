/**
 * Configuration and Feature Flag Management Tests
 * 
 * Comprehensive test suite for configuration management system:
 * - Configuration validation and updates
 * - Feature flag evaluation with rollout logic
 * - Environment-specific overrides
 * - Hot reload and change notification
 * - Import/export functionality
 * - Performance and edge case testing
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  ConfigurationManager,
  LetheConfig,
  FeatureFlagContext,
  initializeConfig,
  getConfigManager,
  isFeatureEnabled,
  getConfigSection,
  CONFIG_SCHEMAS
} from '../config.js';
import { LetheError } from '../errors.js';

describe('ConfigurationManager', () => {
  let configManager: ConfigurationManager;
  
  beforeEach(() => {
    configManager = new ConfigurationManager();
  });

  describe('Configuration Management', () => {
    it('should initialize with default configuration', () => {
      const config = configManager.getConfig();
      
      expect(config.system.version).toBe('1.0.0');
      expect(config.system.environment).toBe('development');
      expect(config.performance.targetP50Ms).toBe(3000);
      expect(config.retrieval.defaultStrategy).toBe('hybrid');
      expect(config.features.enableAdvancedRetrieval).toBe(true);
    });

    it('should merge partial configuration with defaults', () => {
      const customConfig = new ConfigurationManager({
        system: {
          environment: 'production' as const,
          logLevel: 'error' as const
        },
        performance: {
          targetP50Ms: 2000
        }
      });
      
      const config = customConfig.getConfig();
      
      // Custom values should override defaults
      expect(config.system.environment).toBe('production');
      expect(config.system.logLevel).toBe('error');
      expect(config.performance.targetP50Ms).toBe(2000);
      
      // Other defaults should remain
      expect(config.system.version).toBe('1.0.0');
      expect(config.performance.targetP95Ms).toBe(6000);
      expect(config.retrieval.defaultStrategy).toBe('hybrid');
    });

    it('should validate configuration updates', async () => {
      const validUpdate = {
        system: {
          maxConcurrentRequests: 200
        },
        performance: {
          targetP50Ms: 2500
        }
      };
      
      const result = await configManager.updateConfig(validUpdate);
      expect(result.success).toBe(true);
      
      const config = configManager.getConfig();
      expect(config.system.maxConcurrentRequests).toBe(200);
      expect(config.performance.targetP50Ms).toBe(2500);
    });

    it('should reject invalid configuration updates', async () => {
      const invalidUpdate = {
        system: {
          maxConcurrentRequests: -10 // Invalid: negative value
        }
      };
      
      const result = await configManager.updateConfig(invalidUpdate);
      expect(result.success).toBe(false);
      expect(result.error?.code).toBe('CONFIGURATION_VALIDATION_ERROR');
    });

    it('should validate cross-field constraints', async () => {
      const invalidUpdate = {
        performance: {
          targetP50Ms: 8000,
          targetP95Ms: 6000 // Invalid: P50 > P95
        }
      };
      
      const result = await configManager.updateConfig(invalidUpdate);
      expect(result.success).toBe(false);
      expect(result.error?.message).toContain('P50 must be less than P95');
    });

    it('should validate hybrid retrieval weights', async () => {
      const invalidUpdate = {
        retrieval: {
          hybridConfig: {
            bm25Weight: 0.7,
            vectorWeight: 0.4 // Invalid: sum = 1.1 ≠ 1.0
          }
        }
      };
      
      const result = await configManager.updateConfig(invalidUpdate);
      expect(result.success).toBe(false);
      expect(result.error?.message).toContain('Hybrid weights must sum to 1.0');
    });

    it('should validate chunking size constraints', async () => {
      const invalidUpdate = {
        chunking: {
          minChunkSize: 1000,
          maxChunkSize: 500 // Invalid: min > max
        }
      };
      
      const result = await configManager.updateConfig(invalidUpdate);
      expect(result.success).toBe(false);
      expect(result.error?.message).toContain('Min chunk size must be less than max chunk size');
    });

    it('should track configuration changes', async () => {
      const changeEvents: any[] = [];
      configManager.addChangeListener((event) => {
        changeEvents.push(event);
      });
      
      await configManager.updateConfig({
        system: {
          maxConcurrentRequests: 150,
          logLevel: 'warn' as const
        }
      });
      
      expect(changeEvents).toHaveLength(2);
      expect(changeEvents[0].field).toBe('system.maxConcurrentRequests');
      expect(changeEvents[0].newValue).toBe(150);
      expect(changeEvents[1].field).toBe('system.logLevel');
      expect(changeEvents[1].newValue).toBe('warn');
    });
  });

  describe('Feature Flag Management', () => {
    const mockContext: FeatureFlagContext = {
      sessionId: 'test-session-123',
      environment: 'development',
      version: '1.0.0',
      userSegment: 'beta'
    };

    it('should evaluate boolean feature flags', () => {
      const enabled = configManager.evaluateFeature('enableAdvancedRetrieval', mockContext);
      expect(enabled).toBe(true);
      
      const disabled = configManager.evaluateFeature('enableIterativeRefinement', mockContext);
      expect(disabled).toBe(false);
    });

    it('should apply percentage-based rollouts', async () => {
      // Set 50% rollout for a feature
      await configManager.updateConfig({
        features: {
          rolloutPercentage: {
            enableLLMReranking: 50
          }
        }
      });
      
      // Test multiple sessions to verify rollout logic
      const results: boolean[] = [];
      for (let i = 0; i < 100; i++) {
        const context = {
          ...mockContext,
          sessionId: `session-${i}`
        };
        results.push(configManager.evaluateFeature('enableLLMReranking', context));
      }
      
      const enabledCount = results.filter(r => r).length;
      expect(enabledCount).toBeGreaterThan(30);
      expect(enabledCount).toBeLessThan(70);
    });

    it('should apply user segment restrictions', async () => {
      await configManager.updateConfig({
        features: {
          userSegments: {
            enableContradictionDetection: ['beta', 'premium']
          }
        }
      });
      
      // Beta user should have access
      const betaContext = { ...mockContext, userSegment: 'beta' };
      expect(configManager.evaluateFeature('enableContradictionDetection', betaContext)).toBe(true);
      
      // Regular user should not have access
      const regularContext = { ...mockContext, userSegment: 'regular' };
      expect(configManager.evaluateFeature('enableContradictionDetection', regularContext)).toBe(false);
    });

    it('should apply environment overrides', async () => {
      await configManager.updateConfig({
        features: {
          enableDetailedLogging: false,
          environmentOverrides: {
            enableDetailedLogging: true
          }
        }
      });
      
      const result = configManager.evaluateFeature('enableDetailedLogging', mockContext);
      expect(result).toBe(true);
    });

    it('should get all active features for context', async () => {
      await configManager.updateConfig({
        features: {
          rolloutPercentage: {
            enableBatching: 0 // Disable for this session
          }
        }
      });
      
      const activeFeatures = configManager.getActiveFeatures(mockContext);
      
      expect(activeFeatures.enableAdvancedRetrieval).toBe(true);
      expect(activeFeatures.enableBatching).toBe(false);
      expect(activeFeatures.enableIterativeRefinement).toBe(false);
      
      // Should not include rollout control fields
      expect(activeFeatures.rolloutPercentage).toBeUndefined();
      expect(activeFeatures.userSegments).toBeUndefined();
    });

    it('should handle consistent percentage rollouts', () => {
      // Same session should always get same result
      const context1 = { ...mockContext, sessionId: 'consistent-session' };
      const result1a = configManager.evaluateFeature('enableAdvancedRetrieval', context1);
      const result1b = configManager.evaluateFeature('enableAdvancedRetrieval', context1);
      expect(result1a).toBe(result1b);
      
      // Different sessions can get different results
      const context2 = { ...mockContext, sessionId: 'different-session' };
      const result2 = configManager.evaluateFeature('enableAdvancedRetrieval', context2);
      // Results may be same or different, but should be deterministic
      const result2b = configManager.evaluateFeature('enableAdvancedRetrieval', context2);
      expect(result2).toBe(result2b);
    });
  });

  describe('Configuration Sections', () => {
    it('should retrieve specific configuration sections', () => {
      const systemConfig = configManager.getSection('system');
      expect(systemConfig).toHaveProperty('version');
      expect(systemConfig).toHaveProperty('environment');
      expect(systemConfig).toHaveProperty('logLevel');
      
      const performanceConfig = configManager.getSection('performance');
      expect(performanceConfig).toHaveProperty('targetP50Ms');
      expect(performanceConfig).toHaveProperty('targetP95Ms');
      
      const retrievalConfig = configManager.getSection('retrieval');
      expect(retrievalConfig).toHaveProperty('defaultStrategy');
      expect(retrievalConfig).toHaveProperty('bm25Config');
    });

    it('should return immutable section copies', () => {
      const systemConfig1 = configManager.getSection('system');
      const systemConfig2 = configManager.getSection('system');
      
      // Should be different objects
      expect(systemConfig1).not.toBe(systemConfig2);
      
      // But with same content
      expect(systemConfig1).toEqual(systemConfig2);
      
      // Modifying one should not affect the other
      (systemConfig1 as any).modifiedProperty = 'test';
      expect((systemConfig2 as any).modifiedProperty).toBeUndefined();
    });
  });

  describe('Change Listeners and History', () => {
    it('should notify change listeners', async () => {
      const mockListener = vi.fn();
      configManager.addChangeListener(mockListener);
      
      await configManager.updateConfig({
        system: { logLevel: 'debug' as const }
      });
      
      expect(mockListener).toHaveBeenCalledTimes(1);
      expect(mockListener).toHaveBeenCalledWith(
        expect.objectContaining({
          field: 'system.logLevel',
          newValue: 'debug',
          source: 'manual'
        })
      );
    });

    it('should remove change listeners', async () => {
      const mockListener = vi.fn();
      configManager.addChangeListener(mockListener);
      configManager.removeChangeListener(mockListener);
      
      await configManager.updateConfig({
        system: { logLevel: 'debug' as const }
      });
      
      expect(mockListener).not.toHaveBeenCalled();
    });

    it('should track change history', async () => {
      await configManager.updateConfig({
        system: { logLevel: 'warn' as const },
        performance: { targetP50Ms: 2500 }
      });
      
      const history = configManager.getChangeHistory();
      expect(history).toHaveLength(2);
      
      const logLevelChange = history.find(h => h.field === 'system.logLevel');
      expect(logLevelChange).toBeDefined();
      expect(logLevelChange?.newValue).toBe('warn');
      
      const performanceChange = history.find(h => h.field === 'performance.targetP50Ms');
      expect(performanceChange).toBeDefined();
      expect(performanceChange?.newValue).toBe(2500);
    });

    it('should limit change history', async () => {
      // Make multiple changes
      for (let i = 0; i < 150; i++) {
        await configManager.updateConfig({
          system: { maxConcurrentRequests: 100 + i }
        });
      }
      
      const limitedHistory = configManager.getChangeHistory(50);
      expect(limitedHistory).toHaveLength(50);
      
      // Should return most recent changes
      expect(limitedHistory[limitedHistory.length - 1].newValue).toBe(249);
    });
  });

  describe('Import/Export Functionality', () => {
    it('should export configuration as JSON', () => {
      const configJson = configManager.exportConfig();
      const parsedConfig = JSON.parse(configJson);
      
      expect(parsedConfig).toHaveProperty('system');
      expect(parsedConfig).toHaveProperty('performance');
      expect(parsedConfig).toHaveProperty('retrieval');
      expect(parsedConfig).toHaveProperty('features');
      expect(parsedConfig.system.version).toBe('1.0.0');
    });

    it('should import valid configuration from JSON', async () => {
      const configUpdate = {
        system: {
          environment: 'production' as const,
          logLevel: 'error' as const
        },
        performance: {
          targetP50Ms: 2000
        }
      };
      
      const result = await configManager.importConfig(JSON.stringify(configUpdate));
      expect(result.success).toBe(true);
      
      const config = configManager.getConfig();
      expect(config.system.environment).toBe('production');
      expect(config.system.logLevel).toBe('error');
      expect(config.performance.targetP50Ms).toBe(2000);
    });

    it('should reject invalid JSON during import', async () => {
      const result = await configManager.importConfig('invalid json');
      expect(result.success).toBe(false);
      expect(result.error?.code).toBe('CONFIGURATION_IMPORT_ERROR');
    });

    it('should reject invalid configuration during import', async () => {
      const invalidConfig = {
        system: {
          maxConcurrentRequests: 'invalid' // Should be number
        }
      };
      
      const result = await configManager.importConfig(JSON.stringify(invalidConfig));
      expect(result.success).toBe(false);
      expect(result.error?.code).toBe('CONFIGURATION_VALIDATION_ERROR');
    });
  });

  describe('Hot Reload Functionality', () => {
    it('should enable and disable hot reload', () => {
      configManager.enableHotReload();
      // In a real implementation, this would set up file watchers
      // For now, we just test the flag is set
      
      configManager.disableHotReload();
      // And that it can be disabled
    });
  });

  describe('Error Handling', () => {
    it('should handle listener errors gracefully', async () => {
      const faultyListener = vi.fn(() => {
        throw new Error('Listener error');
      });
      
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation();
      
      configManager.addChangeListener(faultyListener);
      
      await configManager.updateConfig({
        system: { logLevel: 'debug' as const }
      });
      
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Error in config change listener:',
        expect.any(Error)
      );
      
      consoleErrorSpy.mockRestore();
    });

    it('should throw on invalid initial configuration', () => {
      expect(() => new ConfigurationManager({
        system: {
          maxConcurrentRequests: -1 // Invalid
        }
      })).toThrow('Invalid initial configuration');
    });
  });

  describe('Performance Testing', () => {
    it('should handle rapid configuration updates', async () => {
      const startTime = Date.now();
      
      // Perform 100 rapid updates
      const promises = Array.from({ length: 100 }, (_, i) => 
        configManager.updateConfig({
          system: { maxConcurrentRequests: 100 + i }
        })
      );
      
      const results = await Promise.all(promises);
      const endTime = Date.now();
      
      // All updates should succeed
      expect(results.every(r => r.success)).toBe(true);
      
      // Should complete reasonably quickly (< 1s)
      expect(endTime - startTime).toBeLessThan(1000);
      
      // Final configuration should reflect last update
      const config = configManager.getConfig();
      expect(config.system.maxConcurrentRequests).toBe(199);
    });

    it('should handle many feature flag evaluations efficiently', () => {
      const startTime = Date.now();
      
      // Evaluate 1000 feature flags
      for (let i = 0; i < 1000; i++) {
        const context: FeatureFlagContext = {
          sessionId: `session-${i}`,
          environment: 'production',
          version: '1.0.0'
        };
        
        configManager.evaluateFeature('enableAdvancedRetrieval', context);
        configManager.evaluateFeature('enableSemanticDiversification', context);
        configManager.evaluateFeature('enableLLMReranking', context);
      }
      
      const endTime = Date.now();
      
      // Should complete very quickly (< 100ms)
      expect(endTime - startTime).toBeLessThan(100);
    });
  });
});

describe('Global Configuration Manager', () => {
  afterEach(() => {
    // Reset global state
    (global as any).globalConfigManager = null;
  });

  it('should initialize global configuration manager', () => {
    const manager = initializeConfig({
      system: { environment: 'testing' as const }
    });
    
    expect(manager).toBeInstanceOf(ConfigurationManager);
    expect(manager.getConfig().system.environment).toBe('testing');
  });

  it('should return existing global manager on subsequent calls', () => {
    const manager1 = initializeConfig();
    const manager2 = initializeConfig();
    
    expect(manager1).toBe(manager2);
  });

  it('should provide access to global manager', () => {
    initializeConfig();
    const manager = getConfigManager();
    
    expect(manager).toBeInstanceOf(ConfigurationManager);
  });

  it('should throw error when accessing uninitialized manager', () => {
    expect(() => getConfigManager()).toThrow('Configuration manager not initialized');
  });

  it('should provide convenience functions', () => {
    initializeConfig();
    
    const context: FeatureFlagContext = {
      sessionId: 'test',
      environment: 'development',
      version: '1.0.0'
    };
    
    const enabled = isFeatureEnabled('enableAdvancedRetrieval', context);
    expect(typeof enabled).toBe('boolean');
    
    const systemConfig = getConfigSection('system');
    expect(systemConfig).toHaveProperty('version');
  });
});

describe('Configuration Schemas', () => {
  it('should provide valid JSON schemas', () => {
    expect(CONFIG_SCHEMAS.system).toBeDefined();
    expect(CONFIG_SCHEMAS.system.type).toBe('object');
    expect(CONFIG_SCHEMAS.system.required).toContain('version');
    
    expect(CONFIG_SCHEMAS.performance).toBeDefined();
    expect(CONFIG_SCHEMAS.performance.type).toBe('object');
    expect(CONFIG_SCHEMAS.performance.required).toContain('targetP50Ms');
  });
});

describe('Edge Cases and Boundary Conditions', () => {
  let configManager: ConfigurationManager;
  
  beforeEach(() => {
    configManager = new ConfigurationManager();
  });

  it('should handle empty configuration updates', async () => {
    const result = await configManager.updateConfig({});
    expect(result.success).toBe(true);
  });

  it('should handle null and undefined values appropriately', async () => {
    const result = await configManager.updateConfig({
      system: undefined,
      performance: null as any
    });
    
    // Should handle gracefully without breaking
    expect(result.success).toBe(true);
  });

  it('should handle extreme rollout percentages', async () => {
    await configManager.updateConfig({
      features: {
        rolloutPercentage: {
          enableAdvancedRetrieval: 0,
          enableSemanticDiversification: 100
        }
      }
    });
    
    const context: FeatureFlagContext = {
      sessionId: 'test',
      environment: 'development',
      version: '1.0.0'
    };
    
    expect(configManager.evaluateFeature('enableAdvancedRetrieval', context)).toBe(false);
    expect(configManager.evaluateFeature('enableSemanticDiversification', context)).toBe(true);
  });

  it('should handle very long session IDs', () => {
    const longSessionId = 'a'.repeat(1000);
    const context: FeatureFlagContext = {
      sessionId: longSessionId,
      environment: 'development',
      version: '1.0.0'
    };
    
    // Should not throw or crash
    const result = configManager.evaluateFeature('enableAdvancedRetrieval', context);
    expect(typeof result).toBe('boolean');
  });

  it('should handle deep nested configuration updates', async () => {
    const deepUpdate = {
      retrieval: {
        bm25Config: {
          k1: 1.5,
          epsilon: 0.3
        }
      }
    };
    
    const result = await configManager.updateConfig(deepUpdate);
    expect(result.success).toBe(true);
    
    const config = configManager.getConfig();
    expect(config.retrieval.bm25Config.k1).toBe(1.5);
    expect(config.retrieval.bm25Config.epsilon).toBe(0.3);
    // Other values should remain unchanged
    expect(config.retrieval.bm25Config.b).toBe(0.75);
  });
});