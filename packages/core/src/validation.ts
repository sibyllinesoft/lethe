import { Result, LetheError } from './types.js';

/**
 * JSON Schema validation utilities with comprehensive error handling
 * for the Lethe retrieval system. Implements RFC 7807 error formats
 * and provides detailed validation feedback.
 */

export interface ValidationError {
  field: string;
  message: string;
  value?: unknown;
  constraint?: string;
  code: string;
}

export interface SchemaValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
  schema: JsonSchema;
  validatedData?: unknown;
}

export interface JsonSchema {
  type: 'object' | 'array' | 'string' | 'number' | 'boolean' | 'null';
  properties?: Record<string, JsonSchema>;
  items?: JsonSchema;
  required?: string[];
  additionalProperties?: boolean | JsonSchema;
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
  enum?: unknown[];
  format?: 'email' | 'date-time' | 'uri' | 'uuid' | 'ipv4' | 'ipv6';
  default?: unknown;
  description?: string;
  title?: string;
}

/**
 * Configuration schemas for Lethe system components
 */
export const LETHE_SCHEMAS = {
  config: {
    type: 'object',
    required: ['retrieval', 'chunking', 'ranking', 'performance'],
    properties: {
      retrieval: {
        type: 'object',
        required: ['method', 'topK'],
        properties: {
          method: {
            type: 'string',
            enum: ['window', 'bm25', 'vector', 'hybrid']
          },
          topK: {
            type: 'number',
            minimum: 1,
            maximum: 1000
          },
          windowSize: {
            type: 'number',
            minimum: 1,
            maximum: 10000
          },
          hybridWeights: {
            type: 'array',
            items: {
              type: 'number',
              minimum: 0,
              maximum: 1
            },
            minLength: 2,
            maxLength: 4
          }
        },
        additionalProperties: false
      },
      chunking: {
        type: 'object',
        required: ['strategy', 'maxSize'],
        properties: {
          strategy: {
            type: 'string',
            enum: ['ast', 'hierarchical', 'propositional']
          },
          maxSize: {
            type: 'number',
            minimum: 100,
            maximum: 10000
          },
          overlap: {
            type: 'number',
            minimum: 0,
            maximum: 0.5
          },
          languages: {
            type: 'array',
            items: {
              type: 'string',
              enum: ['typescript', 'javascript', 'python', 'rust', 'go', 'java', 'cpp']
            }
          }
        },
        additionalProperties: false
      },
      ranking: {
        type: 'object',
        properties: {
          enableMetadataBoost: {
            type: 'boolean'
          },
          diversificationMethod: {
            type: 'string',
            enum: ['entity', 'semantic', 'none']
          },
          fusionWeights: {
            type: 'array',
            items: {
              type: 'number',
              minimum: 0,
              maximum: 1
            }
          }
        },
        additionalProperties: false
      },
      performance: {
        type: 'object',
        required: ['budgetTracking', 'telemetry'],
        properties: {
          budgetTracking: {
            type: 'object',
            required: ['totalBudget'],
            properties: {
              totalBudget: {
                type: 'number',
                minimum: 0.1,
                maximum: 1000
              },
              alertThreshold: {
                type: 'number',
                minimum: 0.1,
                maximum: 1.0
              }
            },
            additionalProperties: false
          },
          telemetry: {
            type: 'object',
            required: ['enabled'],
            properties: {
              enabled: {
                type: 'boolean'
              },
              endpoint: {
                type: 'string',
                format: 'uri'
              },
              batchSize: {
                type: 'number',
                minimum: 1,
                maximum: 1000
              }
            },
            additionalProperties: false
          }
        },
        additionalProperties: false
      },
      llm: {
        type: 'object',
        properties: {
          model: {
            type: 'string',
            minLength: 1
          },
          apiKey: {
            type: 'string',
            minLength: 1
          },
          enableReranking: {
            type: 'boolean'
          },
          enableContradictionDetection: {
            type: 'boolean'
          },
          rerankerModel: {
            type: 'string',
            minLength: 1
          },
          batchSize: {
            type: 'number',
            minimum: 1,
            maximum: 32
          },
          maxTokens: {
            type: 'number',
            minimum: 1,
            maximum: 32000
          },
          temperature: {
            type: 'number',
            minimum: 0,
            maximum: 2
          }
        },
        additionalProperties: false
      }
    },
    additionalProperties: false
  } as JsonSchema,

  candidate: {
    type: 'object',
    required: ['id', 'content', 'score'],
    properties: {
      id: {
        type: 'string',
        minLength: 1
      },
      content: {
        type: 'string',
        minLength: 1
      },
      score: {
        type: 'number',
        minimum: 0,
        maximum: 1
      },
      llmScore: {
        type: 'number',
        minimum: 0,
        maximum: 1
      },
      metadata: {
        type: 'object',
        additionalProperties: true
      }
    },
    additionalProperties: false
  } as JsonSchema,

  message: {
    type: 'object',
    required: ['id', 'content', 'timestamp'],
    properties: {
      id: {
        type: 'string',
        minLength: 1
      },
      content: {
        type: 'string',
        minLength: 1
      },
      timestamp: {
        type: 'string',
        format: 'date-time'
      },
      author: {
        type: 'string'
      },
      metadata: {
        type: 'object',
        additionalProperties: true
      }
    },
    additionalProperties: false
  } as JsonSchema,

  telemetryEvent: {
    type: 'object',
    required: ['timestamp', 'event', 'sessionId'],
    properties: {
      timestamp: {
        type: 'string',
        format: 'date-time'
      },
      event: {
        type: 'string',
        enum: [
          'orchestration_start',
          'orchestration_complete',
          'retrieval_start',
          'retrieval_complete',
          'ranking_start',
          'ranking_complete',
          'chunking_start',
          'chunking_complete',
          'llm_rerank_start',
          'llm_rerank_complete',
          'contradiction_detection',
          'error'
        ]
      },
      sessionId: {
        type: 'string',
        minLength: 1
      },
      variant: {
        type: 'string',
        enum: ['V1', 'V2', 'V3', 'V4', 'V5']
      },
      performance: {
        type: 'object',
        properties: {
          totalTime: {
            type: 'number',
            minimum: 0
          },
          budgetUsed: {
            type: 'number',
            minimum: 0,
            maximum: 1
          },
          candidateCount: {
            type: 'number',
            minimum: 0
          }
        },
        additionalProperties: false
      },
      error: {
        type: 'object',
        properties: {
          code: {
            type: 'string'
          },
          message: {
            type: 'string'
          },
          timestamp: {
            type: 'string',
            format: 'date-time'
          }
        },
        additionalProperties: true
      }
    },
    additionalProperties: true
  } as JsonSchema
};

/**
 * Main JSON Schema validator with comprehensive error handling
 */
export class JsonSchemaValidator {
  private formatValidators: Map<string, (value: string) => boolean>;

  constructor() {
    this.formatValidators = new Map([
      ['email', this.validateEmail.bind(this)],
      ['date-time', this.validateDateTime.bind(this)],
      ['uri', this.validateUri.bind(this)],
      ['uuid', this.validateUuid.bind(this)],
      ['ipv4', this.validateIPv4.bind(this)],
      ['ipv6', this.validateIPv6.bind(this)]
    ]);
  }

  /**
   * Validate data against a JSON schema with detailed error reporting
   */
  validate(data: unknown, schema: JsonSchema, path: string = '$'): Result<SchemaValidationResult, LetheError> {
    try {
      const errors: ValidationError[] = [];
      const warnings: ValidationError[] = [];
      
      // Recursive validation
      const isValid = this.validateRecursive(data, schema, path, errors, warnings);
      
      const result: SchemaValidationResult = {
        valid: isValid && errors.length === 0,
        errors,
        warnings,
        schema,
        validatedData: isValid ? data : undefined
      };
      
      return { success: true, data: result };
      
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'SCHEMA_VALIDATION_ERROR',
          message: `Schema validation failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date().toISOString(),
          context: { schema, path }
        }
      };
    }
  }

  /**
   * Validate Lethe configuration object
   */
  validateConfig(config: unknown): Result<SchemaValidationResult, LetheError> {
    return this.validate(config, LETHE_SCHEMAS.config, '$.config');
  }

  /**
   * Validate enhanced candidate object
   */
  validateCandidate(candidate: unknown): Result<SchemaValidationResult, LetheError> {
    return this.validate(candidate, LETHE_SCHEMAS.candidate, '$.candidate');
  }

  /**
   * Validate message object
   */
  validateMessage(message: unknown): Result<SchemaValidationResult, LetheError> {
    return this.validate(message, LETHE_SCHEMAS.message, '$.message');
  }

  /**
   * Validate telemetry event object
   */
  validateTelemetryEvent(event: unknown): Result<SchemaValidationResult, LetheError> {
    return this.validate(event, LETHE_SCHEMAS.telemetryEvent, '$.telemetryEvent');
  }

  /**
   * Batch validate an array of objects against a schema
   */
  validateBatch<T>(
    items: T[], 
    schema: JsonSchema,
    options: {
      stopOnFirstError?: boolean;
      maxErrors?: number;
    } = {}
  ): Result<BatchValidationResult<T>, LetheError> {
    try {
      const results: Array<{ item: T; valid: boolean; errors: ValidationError[] }> = [];
      const allErrors: ValidationError[] = [];
      let totalValid = 0;
      
      const maxErrors = options.maxErrors || 1000;
      const stopOnFirstError = options.stopOnFirstError || false;
      
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        const validationResult = this.validate(item, schema, `$[${i}]`);
        
        if (!validationResult.success) {
          const error: ValidationError = {
            field: `$[${i}]`,
            message: validationResult.error.message,
            value: item,
            code: validationResult.error.code
          };
          
          results.push({ item, valid: false, errors: [error] });
          allErrors.push(error);
          
          if (stopOnFirstError || allErrors.length >= maxErrors) {
            break;
          }
        } else {
          const itemValid = validationResult.data.valid;
          results.push({ 
            item, 
            valid: itemValid, 
            errors: validationResult.data.errors 
          });
          
          allErrors.push(...validationResult.data.errors);
          
          if (itemValid) {
            totalValid++;
          }
          
          if (!itemValid && stopOnFirstError) {
            break;
          }
        }
      }
      
      return {
        success: true,
        data: {
          totalItems: items.length,
          validItems: totalValid,
          invalidItems: items.length - totalValid,
          results,
          allErrors,
          overallValid: allErrors.length === 0
        }
      };
      
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'BATCH_VALIDATION_ERROR',
          message: `Batch validation failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date().toISOString(),
          context: { itemCount: items.length, schema }
        }
      };
    }
  }

  /**
   * Recursive validation implementation
   */
  private validateRecursive(
    data: unknown, 
    schema: JsonSchema, 
    path: string, 
    errors: ValidationError[], 
    warnings: ValidationError[]
  ): boolean {
    // Type validation
    if (!this.validateType(data, schema.type, path, errors)) {
      return false;
    }

    // Null check
    if (data === null) {
      return schema.type === 'null';
    }

    let isValid = true;

    // Type-specific validation
    switch (schema.type) {
      case 'object':
        isValid = this.validateObject(data as Record<string, unknown>, schema, path, errors, warnings);
        break;
      case 'array':
        isValid = this.validateArray(data as unknown[], schema, path, errors, warnings);
        break;
      case 'string':
        isValid = this.validateString(data as string, schema, path, errors);
        break;
      case 'number':
        isValid = this.validateNumber(data as number, schema, path, errors);
        break;
    }

    // Enum validation
    if (schema.enum && !schema.enum.includes(data)) {
      errors.push({
        field: path,
        message: `Value must be one of: ${schema.enum.map(v => JSON.stringify(v)).join(', ')}`,
        value: data,
        code: 'ENUM_VIOLATION',
        constraint: `enum: [${schema.enum.map(v => JSON.stringify(v)).join(', ')}]`
      });
      isValid = false;
    }

    return isValid;
  }

  /**
   * Validate data type
   */
  private validateType(data: unknown, type: JsonSchema['type'], path: string, errors: ValidationError[]): boolean {
    const actualType = this.getJsType(data);
    
    if (actualType !== type) {
      errors.push({
        field: path,
        message: `Expected type ${type}, got ${actualType}`,
        value: data,
        code: 'TYPE_MISMATCH',
        constraint: `type: ${type}`
      });
      return false;
    }
    
    return true;
  }

  /**
   * Validate object properties
   */
  private validateObject(
    obj: Record<string, unknown>, 
    schema: JsonSchema, 
    path: string, 
    errors: ValidationError[], 
    warnings: ValidationError[]
  ): boolean {
    let isValid = true;

    // Required properties
    if (schema.required) {
      for (const requiredProp of schema.required) {
        if (!(requiredProp in obj)) {
          errors.push({
            field: `${path}.${requiredProp}`,
            message: `Missing required property: ${requiredProp}`,
            code: 'MISSING_REQUIRED_PROPERTY',
            constraint: `required: [${schema.required.join(', ')}]`
          });
          isValid = false;
        }
      }
    }

    // Property validation
    if (schema.properties) {
      for (const [propName, propSchema] of Object.entries(schema.properties)) {
        if (propName in obj) {
          const propValid = this.validateRecursive(
            obj[propName], 
            propSchema, 
            `${path}.${propName}`, 
            errors, 
            warnings
          );
          if (!propValid) {
            isValid = false;
          }
        }
      }
    }

    // Additional properties
    if (schema.additionalProperties === false) {
      const allowedProps = new Set(Object.keys(schema.properties || {}));
      for (const propName of Object.keys(obj)) {
        if (!allowedProps.has(propName)) {
          warnings.push({
            field: `${path}.${propName}`,
            message: `Additional property not allowed: ${propName}`,
            value: obj[propName],
            code: 'ADDITIONAL_PROPERTY',
            constraint: 'additionalProperties: false'
          });
        }
      }
    } else if (typeof schema.additionalProperties === 'object') {
      // Validate additional properties against schema
      const allowedProps = new Set(Object.keys(schema.properties || {}));
      for (const [propName, propValue] of Object.entries(obj)) {
        if (!allowedProps.has(propName)) {
          const propValid = this.validateRecursive(
            propValue, 
            schema.additionalProperties, 
            `${path}.${propName}`, 
            errors, 
            warnings
          );
          if (!propValid) {
            isValid = false;
          }
        }
      }
    }

    return isValid;
  }

  /**
   * Validate array items
   */
  private validateArray(
    arr: unknown[], 
    schema: JsonSchema, 
    path: string, 
    errors: ValidationError[], 
    warnings: ValidationError[]
  ): boolean {
    let isValid = true;

    if (schema.items) {
      for (let i = 0; i < arr.length; i++) {
        const itemValid = this.validateRecursive(
          arr[i], 
          schema.items, 
          `${path}[${i}]`, 
          errors, 
          warnings
        );
        if (!itemValid) {
          isValid = false;
        }
      }
    }

    return isValid;
  }

  /**
   * Validate string constraints
   */
  private validateString(str: string, schema: JsonSchema, path: string, errors: ValidationError[]): boolean {
    let isValid = true;

    // Length constraints
    if (schema.minLength !== undefined && str.length < schema.minLength) {
      errors.push({
        field: path,
        message: `String length ${str.length} is below minimum ${schema.minLength}`,
        value: str,
        code: 'MIN_LENGTH_VIOLATION',
        constraint: `minLength: ${schema.minLength}`
      });
      isValid = false;
    }

    if (schema.maxLength !== undefined && str.length > schema.maxLength) {
      errors.push({
        field: path,
        message: `String length ${str.length} exceeds maximum ${schema.maxLength}`,
        value: str,
        code: 'MAX_LENGTH_VIOLATION',
        constraint: `maxLength: ${schema.maxLength}`
      });
      isValid = false;
    }

    // Pattern validation
    if (schema.pattern) {
      try {
        const regex = new RegExp(schema.pattern);
        if (!regex.test(str)) {
          errors.push({
            field: path,
            message: `String does not match pattern: ${schema.pattern}`,
            value: str,
            code: 'PATTERN_VIOLATION',
            constraint: `pattern: ${schema.pattern}`
          });
          isValid = false;
        }
      } catch (regexError) {
        errors.push({
          field: path,
          message: `Invalid pattern: ${schema.pattern}`,
          value: str,
          code: 'INVALID_PATTERN',
          constraint: `pattern: ${schema.pattern}`
        });
        isValid = false;
      }
    }

    // Format validation
    if (schema.format) {
      const formatValidator = this.formatValidators.get(schema.format);
      if (formatValidator && !formatValidator(str)) {
        errors.push({
          field: path,
          message: `String does not match format: ${schema.format}`,
          value: str,
          code: 'FORMAT_VIOLATION',
          constraint: `format: ${schema.format}`
        });
        isValid = false;
      }
    }

    return isValid;
  }

  /**
   * Validate number constraints
   */
  private validateNumber(num: number, schema: JsonSchema, path: string, errors: ValidationError[]): boolean {
    let isValid = true;

    if (schema.minimum !== undefined && num < schema.minimum) {
      errors.push({
        field: path,
        message: `Number ${num} is below minimum ${schema.minimum}`,
        value: num,
        code: 'MINIMUM_VIOLATION',
        constraint: `minimum: ${schema.minimum}`
      });
      isValid = false;
    }

    if (schema.maximum !== undefined && num > schema.maximum) {
      errors.push({
        field: path,
        message: `Number ${num} exceeds maximum ${schema.maximum}`,
        value: num,
        code: 'MAXIMUM_VIOLATION',
        constraint: `maximum: ${schema.maximum}`
      });
      isValid = false;
    }

    return isValid;
  }

  /**
   * Get JavaScript type string for validation
   */
  private getJsType(data: unknown): string {
    if (data === null) return 'null';
    if (Array.isArray(data)) return 'array';
    return typeof data;
  }

  // Format validators
  private validateEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  private validateDateTime(dateTime: string): boolean {
    try {
      const date = new Date(dateTime);
      return !isNaN(date.getTime()) && dateTime.includes('T');
    } catch {
      return false;
    }
  }

  private validateUri(uri: string): boolean {
    try {
      new URL(uri);
      return true;
    } catch {
      return false;
    }
  }

  private validateUuid(uuid: string): boolean {
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    return uuidRegex.test(uuid);
  }

  private validateIPv4(ip: string): boolean {
    const ipv4Regex = /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
    return ipv4Regex.test(ip);
  }

  private validateIPv6(ip: string): boolean {
    const ipv6Regex = /^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$/;
    return ipv6Regex.test(ip);
  }
}

/**
 * Batch validation result interface
 */
export interface BatchValidationResult<T> {
  totalItems: number;
  validItems: number;
  invalidItems: number;
  results: Array<{ item: T; valid: boolean; errors: ValidationError[] }>;
  allErrors: ValidationError[];
  overallValid: boolean;
}

/**
 * Configuration validator with feature flag support
 */
export class ConfigurationValidator extends JsonSchemaValidator {
  private featureFlags: Map<string, boolean>;

  constructor(featureFlags: Record<string, boolean> = {}) {
    super();
    this.featureFlags = new Map(Object.entries(featureFlags));
  }

  /**
   * Validate configuration with feature flag considerations
   */
  validateConfigWithFeatures(config: unknown): Result<EnhancedValidationResult, LetheError> {
    const baseResult = this.validateConfig(config);
    
    if (!baseResult.success) {
      return baseResult;
    }

    try {
      const configObj = config as any;
      const additionalErrors: ValidationError[] = [];
      const warnings: ValidationError[] = [...baseResult.data.warnings];

      // Feature-specific validations
      if (this.featureFlags.get('llmReranking') && !configObj.llm?.enableReranking) {
        warnings.push({
          field: '$.config.llm.enableReranking',
          message: 'LLM reranking feature flag is enabled but configuration is disabled',
          code: 'FEATURE_FLAG_MISMATCH'
        });
      }

      if (configObj.llm?.enableReranking && !configObj.llm?.rerankerModel) {
        additionalErrors.push({
          field: '$.config.llm.rerankerModel',
          message: 'Reranker model must be specified when LLM reranking is enabled',
          code: 'CONDITIONAL_REQUIRED_MISSING'
        });
      }

      if (configObj.llm?.enableContradictionDetection && !this.featureFlags.get('contradictionDetection')) {
        warnings.push({
          field: '$.config.llm.enableContradictionDetection',
          message: 'Contradiction detection is enabled but feature flag is disabled',
          code: 'FEATURE_FLAG_MISMATCH'
        });
      }

      // Performance target validation
      if (configObj.performance?.budgetTracking?.totalBudget) {
        const budget = configObj.performance.budgetTracking.totalBudget;
        const method = configObj.retrieval?.method;
        
        // Different methods have different performance characteristics
        const minBudgets = {
          'window': 0.1,
          'bm25': 0.2,
          'vector': 0.5,
          'hybrid': 0.8
        };
        
        const minBudget = minBudgets[method as keyof typeof minBudgets] || 0.1;
        
        if (budget < minBudget) {
          warnings.push({
            field: '$.config.performance.budgetTracking.totalBudget',
            message: `Budget ${budget} may be insufficient for ${method} retrieval method (recommended: ${minBudget})`,
            value: budget,
            code: 'PERFORMANCE_WARNING',
            constraint: `minRecommended: ${minBudget}`
          });
        }
      }

      const enhancedResult: EnhancedValidationResult = {
        ...baseResult.data,
        valid: baseResult.data.valid && additionalErrors.length === 0,
        errors: [...baseResult.data.errors, ...additionalErrors],
        warnings,
        featureFlags: Object.fromEntries(this.featureFlags),
        performanceEstimates: this.calculatePerformanceEstimates(configObj)
      };

      return { success: true, data: enhancedResult };
      
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'FEATURE_VALIDATION_ERROR',
          message: `Feature validation failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date().toISOString(),
          context: { config }
        }
      };
    }
  }

  /**
   * Calculate performance estimates based on configuration
   */
  private calculatePerformanceEstimates(config: any): PerformanceEstimates {
    const estimates: PerformanceEstimates = {
      expectedLatencyMs: 1000, // Base latency
      memoryUsageMB: 100,      // Base memory
      cpuUsagePercent: 20      // Base CPU
    };

    // Adjust based on retrieval method
    switch (config.retrieval?.method) {
      case 'window':
        estimates.expectedLatencyMs += 500;
        estimates.memoryUsageMB += 50;
        break;
      case 'bm25':
        estimates.expectedLatencyMs += 800;
        estimates.memoryUsageMB += 100;
        estimates.cpuUsagePercent += 10;
        break;
      case 'vector':
        estimates.expectedLatencyMs += 1500;
        estimates.memoryUsageMB += 300;
        estimates.cpuUsagePercent += 30;
        break;
      case 'hybrid':
        estimates.expectedLatencyMs += 2000;
        estimates.memoryUsageMB += 400;
        estimates.cpuUsagePercent += 40;
        break;
    }

    // Adjust for LLM features
    if (config.llm?.enableReranking) {
      estimates.expectedLatencyMs += 2000;
      estimates.memoryUsageMB += 200;
      estimates.cpuUsagePercent += 20;
    }

    if (config.llm?.enableContradictionDetection) {
      estimates.expectedLatencyMs += 1500;
      estimates.memoryUsageMB += 150;
      estimates.cpuUsagePercent += 15;
    }

    // Adjust for chunking strategy
    switch (config.chunking?.strategy) {
      case 'ast':
        estimates.expectedLatencyMs += 800;
        estimates.memoryUsageMB += 100;
        estimates.cpuUsagePercent += 25;
        break;
      case 'propositional':
        estimates.expectedLatencyMs += 1200;
        estimates.memoryUsageMB += 150;
        estimates.cpuUsagePercent += 35;
        break;
    }

    return estimates;
  }

  /**
   * Update feature flags
   */
  setFeatureFlags(flags: Record<string, boolean>): void {
    for (const [key, value] of Object.entries(flags)) {
      this.featureFlags.set(key, value);
    }
  }

  /**
   * Get current feature flags
   */
  getFeatureFlags(): Record<string, boolean> {
    return Object.fromEntries(this.featureFlags);
  }
}

/**
 * Enhanced validation result with feature flag and performance information
 */
export interface EnhancedValidationResult extends SchemaValidationResult {
  featureFlags: Record<string, boolean>;
  performanceEstimates: PerformanceEstimates;
}

/**
 * Performance estimates based on configuration
 */
export interface PerformanceEstimates {
  expectedLatencyMs: number;
  memoryUsageMB: number;
  cpuUsagePercent: number;
}

/**
 * Singleton validator instances
 */
export const jsonValidator = new JsonSchemaValidator();
export const configValidator = new ConfigurationValidator();

/**
 * Utility functions for common validation tasks
 */
export const ValidationUtils = {
  /**
   * Create RFC 7807 compliant error from validation result
   */
  toRFC7807Error(validationResult: SchemaValidationResult, title: string = 'Validation Error'): LetheError {
    const firstError = validationResult.errors[0];
    return {
      code: firstError?.code || 'VALIDATION_ERROR',
      message: `${title}: ${firstError?.message || 'Unknown validation error'}`,
      timestamp: new Date().toISOString(),
      context: {
        field: firstError?.field,
        value: firstError?.value,
        constraint: firstError?.constraint,
        totalErrors: validationResult.errors.length,
        allErrors: validationResult.errors
      }
    };
  },

  /**
   * Check if validation result has specific error types
   */
  hasErrorType(validationResult: SchemaValidationResult, errorCode: string): boolean {
    return validationResult.errors.some(error => error.code === errorCode);
  },

  /**
   * Filter validation errors by severity
   */
  getErrorsBySeverity(validationResult: SchemaValidationResult): {
    critical: ValidationError[];
    warning: ValidationError[];
    info: ValidationError[];
  } {
    const critical = validationResult.errors.filter(error => 
      ['TYPE_MISMATCH', 'MISSING_REQUIRED_PROPERTY', 'ENUM_VIOLATION'].includes(error.code)
    );
    
    const warning = validationResult.errors.filter(error => 
      ['FORMAT_VIOLATION', 'PATTERN_VIOLATION'].includes(error.code)
    );
    
    const info = validationResult.errors.filter(error => 
      !critical.includes(error) && !warning.includes(error)
    );
    
    return { critical, warning, info };
  }
};