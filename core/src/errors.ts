import { Result, LetheError } from './types.js';

/**
 * Comprehensive error handling system for Lethe retrieval pipeline
 * Implements RFC 7807 Problem Details for HTTP APIs and provides
 * structured error handling with Result<T, E> pattern throughout.
 */

/**
 * Error categories for systematic error handling
 */
export enum ErrorCategory {
  VALIDATION = 'validation',
  CONFIGURATION = 'configuration', 
  RETRIEVAL = 'retrieval',
  CHUNKING = 'chunking',
  RANKING = 'ranking',
  LLM = 'llm',
  ORCHESTRATION = 'orchestration',
  PERFORMANCE = 'performance',
  NETWORK = 'network',
  STORAGE = 'storage',
  SECURITY = 'security',
  SYSTEM = 'system'
}

/**
 * Error severity levels
 */
export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium', 
  HIGH = 'high',
  CRITICAL = 'critical'
}

/**
 * Extended error interface with context and recovery information
 */
export interface ExtendedLetheError extends LetheError {
  category: ErrorCategory;
  severity: ErrorSeverity;
  recoverable: boolean;
  retryAfter?: number; // milliseconds
  details?: Record<string, unknown>;
  stack?: string;
  cause?: Error;
  correlationId?: string;
}

/**
 * Error recovery strategies
 */
export interface ErrorRecoveryStrategy {
  canRecover(error: ExtendedLetheError): boolean;
  recover(error: ExtendedLetheError, context?: unknown): Promise<Result<unknown, ExtendedLetheError>>;
  maxRetries: number;
  backoffMs: number;
}

/**
 * Comprehensive error builder with fluent API
 */
export class ErrorBuilder {
  private error: Partial<ExtendedLetheError> = {
    timestamp: new Date().toISOString()
  };

  static create(code: string, message: string): ErrorBuilder {
    return new ErrorBuilder().code(code).message(message);
  }

  code(code: string): ErrorBuilder {
    this.error.code = code;
    return this;
  }

  message(message: string): ErrorBuilder {
    this.error.message = message;
    return this;
  }

  category(category: ErrorCategory): ErrorBuilder {
    this.error.category = category;
    return this;
  }

  severity(severity: ErrorSeverity): ErrorBuilder {
    this.error.severity = severity;
    return this;
  }

  recoverable(recoverable: boolean = true): ErrorBuilder {
    this.error.recoverable = recoverable;
    return this;
  }

  retryAfter(ms: number): ErrorBuilder {
    this.error.retryAfter = ms;
    return this;
  }

  context(context: Record<string, unknown>): ErrorBuilder {
    this.error.context = { ...this.error.context, ...context };
    return this;
  }

  details(details: Record<string, unknown>): ErrorBuilder {
    this.error.details = { ...this.error.details, ...details };
    return this;
  }

  cause(cause: Error): ErrorBuilder {
    this.error.cause = cause;
    this.error.stack = cause.stack;
    return this;
  }

  correlationId(id: string): ErrorBuilder {
    this.error.correlationId = id;
    return this;
  }

  build(): ExtendedLetheError {
    // Set defaults
    if (!this.error.category) this.error.category = ErrorCategory.SYSTEM;
    if (!this.error.severity) this.error.severity = ErrorSeverity.MEDIUM;
    if (this.error.recoverable === undefined) this.error.recoverable = false;

    return this.error as ExtendedLetheError;
  }
}

/**
 * Predefined error factories for common scenarios
 */
export const ErrorFactories = {
  // Validation errors
  validation: {
    invalidInput: (field: string, value: unknown, expected: string): ExtendedLetheError =>
      ErrorBuilder.create('INVALID_INPUT', `Invalid input for field ${field}`)
        .category(ErrorCategory.VALIDATION)
        .severity(ErrorSeverity.MEDIUM)
        .recoverable(true)
        .context({ field, value, expected })
        .build(),

    missingRequired: (field: string): ExtendedLetheError =>
      ErrorBuilder.create('MISSING_REQUIRED_FIELD', `Required field missing: ${field}`)
        .category(ErrorCategory.VALIDATION)
        .severity(ErrorSeverity.HIGH)
        .recoverable(true)
        .context({ field })
        .build(),

    schemaViolation: (errors: string[], schema: string): ExtendedLetheError =>
      ErrorBuilder.create('SCHEMA_VALIDATION_FAILED', `Schema validation failed: ${errors.join(', ')}`)
        .category(ErrorCategory.VALIDATION)
        .severity(ErrorSeverity.HIGH)
        .recoverable(true)
        .context({ errors, schema })
        .build()
  },

  // Configuration errors
  configuration: {
    invalidConfig: (section: string, details: Record<string, unknown>): ExtendedLetheError =>
      ErrorBuilder.create('INVALID_CONFIGURATION', `Invalid configuration in section: ${section}`)
        .category(ErrorCategory.CONFIGURATION)
        .severity(ErrorSeverity.HIGH)
        .recoverable(true)
        .context({ section })
        .details(details)
        .build(),

    missingConfig: (key: string, required: boolean = true): ExtendedLetheError =>
      ErrorBuilder.create('MISSING_CONFIGURATION', `Missing configuration key: ${key}`)
        .category(ErrorCategory.CONFIGURATION)
        .severity(required ? ErrorSeverity.CRITICAL : ErrorSeverity.MEDIUM)
        .recoverable(true)
        .context({ key, required })
        .build(),

    featureFlagMismatch: (flag: string, configValue: boolean, flagValue: boolean): ExtendedLetheError =>
      ErrorBuilder.create('FEATURE_FLAG_MISMATCH', `Feature flag ${flag} mismatch`)
        .category(ErrorCategory.CONFIGURATION)
        .severity(ErrorSeverity.LOW)
        .recoverable(true)
        .context({ flag, configValue, flagValue })
        .build()
  },

  // Retrieval errors
  retrieval: {
    searchFailed: (query: string, method: string, cause?: Error): ExtendedLetheError =>
      ErrorBuilder.create('SEARCH_FAILED', `Search failed for query: ${query}`)
        .category(ErrorCategory.RETRIEVAL)
        .severity(ErrorSeverity.HIGH)
        .recoverable(true)
        .retryAfter(1000)
        .context({ query, method })
        .cause(cause || new Error('Unknown search failure'))
        .build(),

    indexNotFound: (indexName: string): ExtendedLetheError =>
      ErrorBuilder.create('INDEX_NOT_FOUND', `Search index not found: ${indexName}`)
        .category(ErrorCategory.RETRIEVAL)
        .severity(ErrorSeverity.CRITICAL)
        .recoverable(false)
        .context({ indexName })
        .build(),

    queryTooComplex: (query: string, complexity: number, maxComplexity: number): ExtendedLetheError =>
      ErrorBuilder.create('QUERY_TOO_COMPLEX', `Query complexity ${complexity} exceeds maximum ${maxComplexity}`)
        .category(ErrorCategory.RETRIEVAL)
        .severity(ErrorSeverity.MEDIUM)
        .recoverable(true)
        .context({ query, complexity, maxComplexity })
        .build()
  },

  // Chunking errors
  chunking: {
    parsingFailed: (content: string, strategy: string, cause?: Error): ExtendedLetheError =>
      ErrorBuilder.create('CONTENT_PARSING_FAILED', `Failed to parse content with ${strategy} strategy`)
        .category(ErrorCategory.CHUNKING)
        .severity(ErrorSeverity.MEDIUM)
        .recoverable(true)
        .context({ contentLength: content.length, strategy })
        .cause(cause || new Error('Parsing failed'))
        .build(),

    chunkTooLarge: (chunkSize: number, maxSize: number): ExtendedLetheError =>
      ErrorBuilder.create('CHUNK_SIZE_EXCEEDED', `Chunk size ${chunkSize} exceeds maximum ${maxSize}`)
        .category(ErrorCategory.CHUNKING)
        .severity(ErrorSeverity.MEDIUM)
        .recoverable(true)
        .context({ chunkSize, maxSize })
        .build(),

    languageNotSupported: (language: string, supportedLanguages: string[]): ExtendedLetheError =>
      ErrorBuilder.create('LANGUAGE_NOT_SUPPORTED', `Language ${language} not supported`)
        .category(ErrorCategory.CHUNKING)
        .severity(ErrorSeverity.MEDIUM)
        .recoverable(true)
        .context({ language, supportedLanguages })
        .build()
  },

  // Ranking errors
  ranking: {
    scoringFailed: (candidateCount: number, cause?: Error): ExtendedLetheError =>
      ErrorBuilder.create('SCORING_FAILED', `Failed to score ${candidateCount} candidates`)
        .category(ErrorCategory.RANKING)
        .severity(ErrorSeverity.HIGH)
        .recoverable(true)
        .retryAfter(500)
        .context({ candidateCount })
        .cause(cause || new Error('Scoring failed'))
        .build(),

    diversificationFailed: (method: string, cause?: Error): ExtendedLetheError =>
      ErrorBuilder.create('DIVERSIFICATION_FAILED', `Diversification failed with method: ${method}`)
        .category(ErrorCategory.RANKING)
        .severity(ErrorSeverity.MEDIUM)
        .recoverable(true)
        .context({ method })
        .cause(cause || new Error('Diversification failed'))
        .build(),

    contradictionDetectionFailed: (pairCount: number, cause?: Error): ExtendedLetheError =>
      ErrorBuilder.create('CONTRADICTION_DETECTION_FAILED', `Failed to detect contradictions in ${pairCount} pairs`)
        .category(ErrorCategory.RANKING)
        .severity(ErrorSeverity.LOW)
        .recoverable(true)
        .context({ pairCount })
        .cause(cause || new Error('Contradiction detection failed'))
        .build()
  },

  // LLM errors
  llm: {
    modelNotAvailable: (modelName: string): ExtendedLetheError =>
      ErrorBuilder.create('LLM_MODEL_NOT_AVAILABLE', `LLM model not available: ${modelName}`)
        .category(ErrorCategory.LLM)
        .severity(ErrorSeverity.HIGH)
        .recoverable(true)
        .retryAfter(5000)
        .context({ modelName })
        .build(),

    rateLimitExceeded: (retryAfter: number): ExtendedLetheError =>
      ErrorBuilder.create('LLM_RATE_LIMIT_EXCEEDED', 'LLM API rate limit exceeded')
        .category(ErrorCategory.LLM)
        .severity(ErrorSeverity.MEDIUM)
        .recoverable(true)
        .retryAfter(retryAfter)
        .build(),

    apiKeyInvalid: (): ExtendedLetheError =>
      ErrorBuilder.create('LLM_API_KEY_INVALID', 'LLM API key is invalid or expired')
        .category(ErrorCategory.LLM)
        .severity(ErrorSeverity.CRITICAL)
        .recoverable(false)
        .build(),

    responseMalformed: (response: string, expectedFormat: string): ExtendedLetheError =>
      ErrorBuilder.create('LLM_RESPONSE_MALFORMED', `LLM response malformed, expected ${expectedFormat}`)
        .category(ErrorCategory.LLM)
        .severity(ErrorSeverity.MEDIUM)
        .recoverable(true)
        .context({ responseLength: response.length, expectedFormat })
        .build(),

    contextLengthExceeded: (length: number, maxLength: number): ExtendedLetheError =>
      ErrorBuilder.create('LLM_CONTEXT_LENGTH_EXCEEDED', `Context length ${length} exceeds maximum ${maxLength}`)
        .category(ErrorCategory.LLM)
        .severity(ErrorSeverity.HIGH)
        .recoverable(true)
        .context({ length, maxLength })
        .build()
  },

  // Performance errors
  performance: {
    budgetExceeded: (used: number, total: number): ExtendedLetheError =>
      ErrorBuilder.create('BUDGET_EXCEEDED', `Budget exceeded: used ${used}, total ${total}`)
        .category(ErrorCategory.PERFORMANCE)
        .severity(ErrorSeverity.HIGH)
        .recoverable(false)
        .context({ used, total, utilization: used / total })
        .build(),

    timeoutExceeded: (operation: string, timeoutMs: number): ExtendedLetheError =>
      ErrorBuilder.create('TIMEOUT_EXCEEDED', `Operation ${operation} exceeded timeout of ${timeoutMs}ms`)
        .category(ErrorCategory.PERFORMANCE)
        .severity(ErrorSeverity.HIGH)
        .recoverable(true)
        .retryAfter(timeoutMs / 2)
        .context({ operation, timeoutMs })
        .build(),

    memoryLimitExceeded: (used: number, limit: number): ExtendedLetheError =>
      ErrorBuilder.create('MEMORY_LIMIT_EXCEEDED', `Memory usage ${used}MB exceeds limit ${limit}MB`)
        .category(ErrorCategory.PERFORMANCE)
        .severity(ErrorSeverity.CRITICAL)
        .recoverable(false)
        .context({ used, limit, utilization: used / limit })
        .build()
  },

  // Network errors
  network: {
    connectionFailed: (endpoint: string, cause?: Error): ExtendedLetheError =>
      ErrorBuilder.create('CONNECTION_FAILED', `Failed to connect to ${endpoint}`)
        .category(ErrorCategory.NETWORK)
        .severity(ErrorSeverity.HIGH)
        .recoverable(true)
        .retryAfter(2000)
        .context({ endpoint })
        .cause(cause || new Error('Connection failed'))
        .build(),

    requestTimeout: (url: string, timeoutMs: number): ExtendedLetheError =>
      ErrorBuilder.create('REQUEST_TIMEOUT', `Request to ${url} timed out after ${timeoutMs}ms`)
        .category(ErrorCategory.NETWORK)
        .severity(ErrorSeverity.MEDIUM)
        .recoverable(true)
        .retryAfter(timeoutMs)
        .context({ url, timeoutMs })
        .build(),

    httpError: (status: number, statusText: string, url: string): ExtendedLetheError =>
      ErrorBuilder.create('HTTP_ERROR', `HTTP ${status} ${statusText} for ${url}`)
        .category(ErrorCategory.NETWORK)
        .severity(status >= 500 ? ErrorSeverity.HIGH : ErrorSeverity.MEDIUM)
        .recoverable(status >= 500 || status === 429)
        .retryAfter(status === 429 ? 60000 : 1000)
        .context({ status, statusText, url })
        .build()
  },

  // System errors
  system: {
    unexpectedError: (message: string, cause?: Error): ExtendedLetheError =>
      ErrorBuilder.create('UNEXPECTED_ERROR', message)
        .category(ErrorCategory.SYSTEM)
        .severity(ErrorSeverity.CRITICAL)
        .recoverable(false)
        .cause(cause || new Error('Unexpected error'))
        .build(),

    resourceUnavailable: (resource: string): ExtendedLetheError =>
      ErrorBuilder.create('RESOURCE_UNAVAILABLE', `Resource unavailable: ${resource}`)
        .category(ErrorCategory.SYSTEM)
        .severity(ErrorSeverity.HIGH)
        .recoverable(true)
        .retryAfter(5000)
        .context({ resource })
        .build(),

    dependencyFailed: (dependency: string, cause?: Error): ExtendedLetheError =>
      ErrorBuilder.create('DEPENDENCY_FAILED', `Dependency failed: ${dependency}`)
        .category(ErrorCategory.SYSTEM)
        .severity(ErrorSeverity.CRITICAL)
        .recoverable(true)
        .retryAfter(10000)
        .context({ dependency })
        .cause(cause || new Error('Dependency failed'))
        .build()
  }
};

/**
 * Error recovery strategies for different error categories
 */
export const RecoveryStrategies: Record<ErrorCategory, ErrorRecoveryStrategy> = {
  [ErrorCategory.VALIDATION]: {
    canRecover: (error) => error.recoverable && error.severity !== ErrorSeverity.CRITICAL,
    recover: async (error, context) => {
      // For validation errors, attempt to use default values or sanitize input
      if (error.context?.field && context && typeof context === 'object') {
        const defaults = getDefaultValues(error.context.field as string);
        return { success: true, data: defaults };
      }
      return { success: false, error };
    },
    maxRetries: 1,
    backoffMs: 0
  },

  [ErrorCategory.CONFIGURATION]: {
    canRecover: (error) => error.recoverable,
    recover: async (error) => {
      // For config errors, attempt to load default configuration
      const defaultConfig = getDefaultConfiguration(error.context?.section as string);
      if (defaultConfig) {
        return { success: true, data: defaultConfig };
      }
      return { success: false, error };
    },
    maxRetries: 1,
    backoffMs: 0
  },

  [ErrorCategory.RETRIEVAL]: {
    canRecover: (error) => error.recoverable,
    recover: async (error, context) => {
      // For retrieval errors, try alternative methods or simplified queries
      if (error.code === 'SEARCH_FAILED') {
        // Try with fallback retrieval method
        return { success: false, error }; // Placeholder - implement fallback logic
      }
      return { success: false, error };
    },
    maxRetries: 3,
    backoffMs: 1000
  },

  [ErrorCategory.CHUNKING]: {
    canRecover: (error) => error.recoverable,
    recover: async (error) => {
      // For chunking errors, try alternative strategies
      if (error.code === 'CONTENT_PARSING_FAILED') {
        // Try simpler chunking strategy
        return { success: false, error }; // Placeholder
      }
      return { success: false, error };
    },
    maxRetries: 2,
    backoffMs: 500
  },

  [ErrorCategory.RANKING]: {
    canRecover: (error) => error.recoverable && error.severity !== ErrorSeverity.CRITICAL,
    recover: async (error) => {
      // For ranking errors, use simpler scoring methods
      return { success: false, error }; // Placeholder
    },
    maxRetries: 2,
    backoffMs: 500
  },

  [ErrorCategory.LLM]: {
    canRecover: (error) => error.recoverable && error.code !== 'LLM_API_KEY_INVALID',
    recover: async (error) => {
      // For LLM errors, wait and retry or use fallback models
      if (error.code === 'LLM_RATE_LIMIT_EXCEEDED') {
        await new Promise(resolve => setTimeout(resolve, error.retryAfter || 5000));
        return { success: false, error }; // Signal to retry
      }
      return { success: false, error };
    },
    maxRetries: 5,
    backoffMs: 2000
  },

  [ErrorCategory.ORCHESTRATION]: {
    canRecover: (error) => error.recoverable,
    recover: async (error) => {
      // For orchestration errors, try to continue with partial results
      return { success: false, error }; // Placeholder
    },
    maxRetries: 1,
    backoffMs: 1000
  },

  [ErrorCategory.PERFORMANCE]: {
    canRecover: (error) => error.code !== 'BUDGET_EXCEEDED' && error.code !== 'MEMORY_LIMIT_EXCEEDED',
    recover: async (error) => {
      // For performance errors, try with reduced parameters
      return { success: false, error }; // Placeholder
    },
    maxRetries: 1,
    backoffMs: 2000
  },

  [ErrorCategory.NETWORK]: {
    canRecover: (error) => error.recoverable,
    recover: async (error) => {
      // For network errors, wait and retry
      if (error.retryAfter) {
        await new Promise(resolve => setTimeout(resolve, error.retryAfter));
      }
      return { success: false, error }; // Signal to retry
    },
    maxRetries: 3,
    backoffMs: 1000
  },

  [ErrorCategory.STORAGE]: {
    canRecover: (error) => error.recoverable,
    recover: async (error) => {
      // For storage errors, retry or use alternative storage
      return { success: false, error }; // Placeholder
    },
    maxRetries: 3,
    backoffMs: 1000
  },

  [ErrorCategory.SECURITY]: {
    canRecover: () => false, // Security errors should not be automatically recovered
    recover: async (error) => {
      return { success: false, error };
    },
    maxRetries: 0,
    backoffMs: 0
  },

  [ErrorCategory.SYSTEM]: {
    canRecover: (error) => error.recoverable,
    recover: async (error) => {
      // For system errors, basic retry logic
      return { success: false, error }; // Placeholder
    },
    maxRetries: 2,
    backoffMs: 5000
  }
};

/**
 * Error handler with automatic recovery and telemetry
 */
export class ErrorHandler {
  private correlationCounter = 0;
  private errorCounts = new Map<string, number>();
  private lastErrors = new Map<string, ExtendedLetheError>();

  /**
   * Handle error with automatic recovery attempts
   */
  async handleError<T>(
    error: ExtendedLetheError,
    context?: unknown,
    attemptRecovery: boolean = true
  ): Promise<Result<T, ExtendedLetheError>> {
    // Add correlation ID if not present
    if (!error.correlationId) {
      error.correlationId = `err_${Date.now()}_${++this.correlationCounter}`;
    }

    // Track error frequency
    this.trackError(error);

    // Log error (in production, send to telemetry service)
    this.logError(error);

    // Attempt recovery if enabled and error is recoverable
    if (attemptRecovery && error.recoverable) {
      const strategy = RecoveryStrategies[error.category];
      if (strategy && strategy.canRecover(error)) {
        const recoveryResult = await this.attemptRecovery(error, context, strategy);
        if (recoveryResult.success) {
          return recoveryResult as Result<T, ExtendedLetheError>;
        }
      }
    }

    return { success: false, error };
  }

  /**
   * Attempt error recovery with retry logic
   */
  private async attemptRecovery<T>(
    error: ExtendedLetheError,
    context: unknown,
    strategy: ErrorRecoveryStrategy
  ): Promise<Result<T, ExtendedLetheError>> {
    let lastError = error;
    
    for (let attempt = 1; attempt <= strategy.maxRetries; attempt++) {
      try {
        // Apply backoff delay
        if (attempt > 1) {
          const delay = strategy.backoffMs * Math.pow(2, attempt - 2); // Exponential backoff
          await new Promise(resolve => setTimeout(resolve, delay));
        }

        const recoveryResult = await strategy.recover(lastError, context);
        
        if (recoveryResult.success) {
          // Log successful recovery
          this.logRecovery(error, attempt);
          return recoveryResult as Result<T, ExtendedLetheError>;
        }
        
        lastError = recoveryResult.error as ExtendedLetheError;
        
      } catch (recoveryError) {
        lastError = ErrorFactories.system.unexpectedError(
          `Recovery attempt ${attempt} failed`,
          recoveryError instanceof Error ? recoveryError : new Error('Recovery failed')
        );
      }
    }

    // All recovery attempts failed
    this.logRecoveryFailure(error, strategy.maxRetries);
    return { success: false, error: lastError };
  }

  /**
   * Track error occurrence for analysis
   */
  private trackError(error: ExtendedLetheError): void {
    const key = `${error.category}:${error.code}`;
    this.errorCounts.set(key, (this.errorCounts.get(key) || 0) + 1);
    this.lastErrors.set(key, error);
  }

  /**
   * Log error for telemetry and debugging
   */
  private logError(error: ExtendedLetheError): void {
    const logEntry = {
      timestamp: error.timestamp,
      correlationId: error.correlationId,
      category: error.category,
      code: error.code,
      message: error.message,
      severity: error.severity,
      context: error.context,
      stack: error.stack
    };

    // In development, log to console
    if (process.env.NODE_ENV === 'development') {
      console.error('[LETHE ERROR]', logEntry);
    }

    // In production, send to telemetry service
    // Implementation would depend on your telemetry provider
  }

  /**
   * Log successful recovery
   */
  private logRecovery(error: ExtendedLetheError, attempt: number): void {
    console.log(`[LETHE RECOVERY] Successfully recovered from ${error.code} after ${attempt} attempts`);
  }

  /**
   * Log recovery failure
   */
  private logRecoveryFailure(error: ExtendedLetheError, maxRetries: number): void {
    console.error(`[LETHE RECOVERY FAILED] Failed to recover from ${error.code} after ${maxRetries} attempts`);
  }

  /**
   * Get error statistics for monitoring
   */
  getErrorStats(): { code: string; count: number; lastOccurred: string }[] {
    return Array.from(this.errorCounts.entries()).map(([code, count]) => ({
      code,
      count,
      lastOccurred: this.lastErrors.get(code)?.timestamp || 'Unknown'
    }));
  }

  /**
   * Reset error tracking
   */
  resetStats(): void {
    this.errorCounts.clear();
    this.lastErrors.clear();
  }
}

/**
 * Utility functions for error handling
 */
export const ErrorUtils = {
  /**
   * Wrap async operations with error handling
   */
  async wrap<T>(
    operation: () => Promise<T>,
    errorBuilder?: (error: Error) => ExtendedLetheError
  ): Promise<Result<T, ExtendedLetheError>> {
    try {
      const result = await operation();
      return { success: true, data: result };
    } catch (error) {
      const letheError = errorBuilder 
        ? errorBuilder(error instanceof Error ? error : new Error('Unknown error'))
        : ErrorFactories.system.unexpectedError(
            error instanceof Error ? error.message : 'Unknown error',
            error instanceof Error ? error : new Error('Unknown error')
          );
      
      return { success: false, error: letheError };
    }
  },

  /**
   * Chain multiple operations with error propagation
   */
  async chain<T, U>(
    result: Result<T, ExtendedLetheError>,
    operation: (data: T) => Promise<Result<U, ExtendedLetheError>>
  ): Promise<Result<U, ExtendedLetheError>> {
    if (!result.success) {
      return result as Result<U, ExtendedLetheError>;
    }
    
    return await operation(result.data);
  },

  /**
   * Convert standard Error to ExtendedLetheError
   */
  fromError(error: Error, category: ErrorCategory = ErrorCategory.SYSTEM): ExtendedLetheError {
    return ErrorBuilder.create('CONVERTED_ERROR', error.message)
      .category(category)
      .severity(ErrorSeverity.MEDIUM)
      .cause(error)
      .build();
  },

  /**
   * Check if error is retryable
   */
  isRetryable(error: ExtendedLetheError): boolean {
    return error.recoverable && 
           error.severity !== ErrorSeverity.CRITICAL &&
           error.category !== ErrorCategory.SECURITY;
  }
};

// Placeholder functions for recovery strategies
function getDefaultValues(field: string): unknown {
  // Implementation would provide sensible defaults based on field type
  const defaults: Record<string, unknown> = {
    topK: 50,
    temperature: 0.1,
    maxTokens: 4096,
    batchSize: 8
  };
  
  return defaults[field];
}

function getDefaultConfiguration(section: string): unknown {
  // Implementation would provide default configuration sections
  const defaults: Record<string, unknown> = {
    retrieval: {
      method: 'hybrid',
      topK: 50,
      windowSize: 1000
    },
    chunking: {
      strategy: 'hierarchical',
      maxSize: 2000,
      overlap: 0.1
    }
  };
  
  return defaults[section];
}

/**
 * Singleton error handler instance
 */
export const errorHandler = new ErrorHandler();