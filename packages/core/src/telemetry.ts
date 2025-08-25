import { Result, LetheError, PerformanceMetrics } from './types.js';
import { ExtendedLetheError, ErrorCategory } from './errors.js';

/**
 * Comprehensive performance benchmarking and telemetry system for Lethe
 * Implements JSONL logging, performance tracking, and real-time monitoring
 * with strict adherence to performance budgets and SLA requirements.
 */

/**
 * Telemetry event types for structured logging
 */
export enum TelemetryEventType {
  ORCHESTRATION_START = 'orchestration_start',
  ORCHESTRATION_COMPLETE = 'orchestration_complete',
  RETRIEVAL_START = 'retrieval_start',
  RETRIEVAL_COMPLETE = 'retrieval_complete',
  CHUNKING_START = 'chunking_start',
  CHUNKING_COMPLETE = 'chunking_complete',
  RANKING_START = 'ranking_start',
  RANKING_COMPLETE = 'ranking_complete',
  LLM_RERANK_START = 'llm_rerank_start',
  LLM_RERANK_COMPLETE = 'llm_rerank_complete',
  CONTRADICTION_DETECTION = 'contradiction_detection',
  PERFORMANCE_ALERT = 'performance_alert',
  BUDGET_ALERT = 'budget_alert',
  ERROR_OCCURRED = 'error_occurred',
  SYSTEM_HEALTH = 'system_health',
  USER_INTERACTION = 'user_interaction'
}

/**
 * Performance metrics for different operation types
 */
export interface OperationMetrics {
  operationType: string;
  startTime: number;
  endTime: number;
  duration: number;
  memoryBefore: number;
  memoryAfter: number;
  memoryPeak: number;
  cpuUsage: number;
  success: boolean;
  itemsProcessed?: number;
  bytesProcessed?: number;
  cacheHits?: number;
  cacheMisses?: number;
  apiCalls?: number;
  variant?: string;
}

/**
 * System health metrics
 */
export interface SystemHealthMetrics {
  timestamp: string;
  memoryUsage: {
    used: number;
    free: number;
    total: number;
    heapUsed: number;
    heapTotal: number;
  };
  cpuUsage: {
    user: number;
    system: number;
    idle: number;
  };
  processMetrics: {
    uptime: number;
    pid: number;
    ppid: number;
    platform: string;
    arch: string;
    nodeVersion: string;
  };
  performance: {
    eventLoopDelay: number;
    gcCount: number;
    gcDuration: number;
  };
}

/**
 * Telemetry event structure for JSONL logging
 */
export interface TelemetryEvent {
  timestamp: string;
  event: TelemetryEventType;
  sessionId: string;
  correlationId?: string;
  variant?: string;
  level: 'info' | 'warn' | 'error' | 'debug';
  metrics?: OperationMetrics;
  systemHealth?: SystemHealthMetrics;
  error?: ExtendedLetheError;
  context?: Record<string, unknown>;
  performance?: {
    totalTime?: number;
    budgetUsed?: number;
    candidateCount?: number;
    contradictionCount?: number;
    memoryPeak?: number;
    cpuPeak?: number;
  };
  sla?: {
    targetLatency: number;
    actualLatency: number;
    targetThroughput: number;
    actualThroughput: number;
    budgetLimit: number;
    budgetUsed: number;
    violated: boolean;
  };
}

/**
 * Performance benchmark configuration
 */
export interface BenchmarkConfig {
  enabled: boolean;
  collectSystemMetrics: boolean;
  collectMemoryMetrics: boolean;
  collectCPUMetrics: boolean;
  collectGCMetrics: boolean;
  sampleRate: number; // 0.0 to 1.0
  bufferSize: number;
  flushInterval: number; // milliseconds
  retentionDays: number;
  compressionEnabled: boolean;
}

/**
 * Performance SLA definitions
 */
export interface PerformanceSLA {
  variant: string;
  targetLatencyP50: number; // milliseconds
  targetLatencyP95: number; // milliseconds
  targetLatencyP99: number; // milliseconds
  maxMemoryUsageMB: number;
  maxCPUUsagePercent: number;
  minThroughputQPS: number; // queries per second
  budgetLimitRatio: number; // 0.0 to 1.0
  errorRateThreshold: number; // 0.0 to 1.0
}

/**
 * Default SLA configurations for each variant
 */
export const DEFAULT_SLAS: Record<string, PerformanceSLA> = {
  V1: {
    variant: 'V1',
    targetLatencyP50: 3000,
    targetLatencyP95: 6000,
    targetLatencyP99: 8000,
    maxMemoryUsageMB: 1500,
    maxCPUUsagePercent: 80,
    minThroughputQPS: 10,
    budgetLimitRatio: 0.9,
    errorRateThreshold: 0.05
  },
  V2: {
    variant: 'V2',
    targetLatencyP50: 3500,
    targetLatencyP95: 7000,
    targetLatencyP99: 9000,
    maxMemoryUsageMB: 1500,
    maxCPUUsagePercent: 85,
    minThroughputQPS: 8,
    budgetLimitRatio: 0.9,
    errorRateThreshold: 0.05
  },
  V3: {
    variant: 'V3',
    targetLatencyP50: 4000,
    targetLatencyP95: 8000,
    targetLatencyP99: 10000,
    maxMemoryUsageMB: 1500,
    maxCPUUsagePercent: 85,
    minThroughputQPS: 6,
    budgetLimitRatio: 0.9,
    errorRateThreshold: 0.05
  },
  V4: {
    variant: 'V4',
    targetLatencyP50: 4000,
    targetLatencyP95: 8000,
    targetLatencyP99: 12000,
    maxMemoryUsageMB: 2000,
    maxCPUUsagePercent: 90,
    minThroughputQPS: 5,
    budgetLimitRatio: 0.9,
    errorRateThreshold: 0.06
  },
  V5: {
    variant: 'V5',
    targetLatencyP50: 4500,
    targetLatencyP95: 9000,
    targetLatencyP99: 12000,
    maxMemoryUsageMB: 2000,
    maxCPUUsagePercent: 90,
    minThroughputQPS: 4,
    budgetLimitRatio: 0.9,
    errorRateThreshold: 0.06
  }
};

/**
 * Performance tracker for individual operations
 */
export class PerformanceTracker {
  private startTime: number;
  private startMemory: NodeJS.MemoryUsage;
  private peakMemory: number;
  private samples: number[];
  private operationType: string;
  private metadata: Record<string, unknown>;

  constructor(operationType: string, metadata: Record<string, unknown> = {}) {
    this.operationType = operationType;
    this.metadata = metadata;
    this.startTime = performance.now();
    this.startMemory = process.memoryUsage();
    this.peakMemory = this.startMemory.heapUsed;
    this.samples = [];
  }

  /**
   * Record a performance sample during operation
   */
  sample(value?: number): void {
    const currentMemory = process.memoryUsage().heapUsed;
    if (currentMemory > this.peakMemory) {
      this.peakMemory = currentMemory;
    }
    
    if (value !== undefined) {
      this.samples.push(value);
    }
  }

  /**
   * Complete tracking and return metrics
   */
  complete(success: boolean = true, itemsProcessed?: number): OperationMetrics {
    const endTime = performance.now();
    const endMemory = process.memoryUsage();
    
    return {
      operationType: this.operationType,
      startTime: this.startTime,
      endTime: endTime,
      duration: endTime - this.startTime,
      memoryBefore: this.startMemory.heapUsed,
      memoryAfter: endMemory.heapUsed,
      memoryPeak: this.peakMemory,
      cpuUsage: process.cpuUsage().user / 1000000, // Convert to seconds
      success,
      itemsProcessed,
      bytesProcessed: this.metadata.bytesProcessed as number,
      cacheHits: this.metadata.cacheHits as number,
      cacheMisses: this.metadata.cacheMisses as number,
      apiCalls: this.metadata.apiCalls as number,
      variant: this.metadata.variant as string
    };
  }
}

/**
 * Main telemetry system with JSONL logging and real-time monitoring
 */
export class TelemetrySystem {
  private config: BenchmarkConfig;
  private eventBuffer: TelemetryEvent[] = [];
  private flushTimer?: NodeJS.Timeout;
  private latencyHistogram = new Map<string, number[]>();
  private throughputCounter = new Map<string, { count: number; startTime: number }>();
  private errorCounts = new Map<string, number>();
  private slaViolations = new Map<string, number>();
  private systemHealthTimer?: NodeJS.Timeout;
  private performanceObserver?: PerformanceObserver;

  constructor(config: Partial<BenchmarkConfig> = {}) {
    this.config = {
      enabled: true,
      collectSystemMetrics: true,
      collectMemoryMetrics: true,
      collectCPUMetrics: true,
      collectGCMetrics: true,
      sampleRate: 1.0,
      bufferSize: 1000,
      flushInterval: 10000, // 10 seconds
      retentionDays: 30,
      compressionEnabled: true,
      ...config
    };

    if (this.config.enabled) {
      this.startPerformanceMonitoring();
    }
  }

  /**
   * Start a performance tracking session for an operation
   */
  startTracking(operationType: string, metadata: Record<string, unknown> = {}): PerformanceTracker {
    return new PerformanceTracker(operationType, metadata);
  }

  /**
   * Log a telemetry event
   */
  logEvent(
    eventType: TelemetryEventType,
    sessionId: string,
    data: {
      variant?: string;
      correlationId?: string;
      metrics?: OperationMetrics;
      error?: ExtendedLetheError;
      context?: Record<string, unknown>;
      level?: 'info' | 'warn' | 'error' | 'debug';
    } = {}
  ): void {
    if (!this.config.enabled || Math.random() > this.config.sampleRate) {
      return;
    }

    const event: TelemetryEvent = {
      timestamp: new Date().toISOString(),
      event: eventType,
      sessionId,
      level: data.level || 'info',
      correlationId: data.correlationId,
      variant: data.variant,
      metrics: data.metrics,
      error: data.error,
      context: data.context
    };

    // Add system health if configured
    if (this.config.collectSystemMetrics) {
      event.systemHealth = this.collectSystemHealth();
    }

    // Add SLA tracking
    if (data.metrics && data.variant) {
      event.sla = this.evaluateSLA(data.metrics, data.variant);
    }

    // Track performance metrics
    if (data.metrics) {
      this.trackPerformanceMetrics(data.metrics, data.variant);
    }

    // Track errors
    if (data.error) {
      this.trackError(data.error);
    }

    this.eventBuffer.push(event);

    // Trigger immediate flush for errors or SLA violations
    if (data.level === 'error' || (event.sla && event.sla.violated)) {
      this.flush(true);
    } else if (this.eventBuffer.length >= this.config.bufferSize) {
      this.flush();
    }
  }

  /**
   * Log orchestration start
   */
  logOrchestrationStart(sessionId: string, variant: string, context: Record<string, unknown>): void {
    this.logEvent(TelemetryEventType.ORCHESTRATION_START, sessionId, {
      variant,
      context,
      level: 'info'
    });

    // Initialize throughput tracking
    this.throughputCounter.set(sessionId, {
      count: 0,
      startTime: Date.now()
    });
  }

  /**
   * Log orchestration completion with comprehensive metrics
   */
  logOrchestrationComplete(
    sessionId: string,
    variant: string,
    metrics: OperationMetrics,
    performanceData: {
      totalTime: number;
      budgetUsed: number;
      candidateCount: number;
      contradictionCount?: number;
    }
  ): void {
    // Update throughput tracking
    const throughput = this.throughputCounter.get(sessionId);
    if (throughput) {
      throughput.count++;
    }

    this.logEvent(TelemetryEventType.ORCHESTRATION_COMPLETE, sessionId, {
      variant,
      metrics,
      context: {
        performance: performanceData
      },
      level: 'info'
    });

    // Track latency for histogram
    if (!this.latencyHistogram.has(variant)) {
      this.latencyHistogram.set(variant, []);
    }
    this.latencyHistogram.get(variant)!.push(performanceData.totalTime);
  }

  /**
   * Log performance alert
   */
  logPerformanceAlert(
    sessionId: string,
    variant: string,
    alertType: 'LATENCY' | 'MEMORY' | 'CPU' | 'BUDGET' | 'ERROR_RATE',
    details: Record<string, unknown>
  ): void {
    this.logEvent(TelemetryEventType.PERFORMANCE_ALERT, sessionId, {
      variant,
      context: {
        alertType,
        details
      },
      level: 'warn'
    });
  }

  /**
   * Log error occurrence
   */
  logError(sessionId: string, error: ExtendedLetheError, context?: Record<string, unknown>): void {
    this.logEvent(TelemetryEventType.ERROR_OCCURRED, sessionId, {
      error,
      context,
      level: 'error',
      correlationId: error.correlationId
    });
  }

  /**
   * Get performance statistics for monitoring dashboards
   */
  getPerformanceStats(variant?: string): PerformanceStats {
    const stats: PerformanceStats = {
      timestamp: new Date().toISOString(),
      totalEvents: this.eventBuffer.length,
      latencyStats: {},
      throughputStats: {},
      errorStats: {},
      slaViolationStats: {},
      systemHealth: this.collectSystemHealth()
    };

    // Calculate latency percentiles
    for (const [v, latencies] of this.latencyHistogram.entries()) {
      if (variant && v !== variant) continue;
      
      const sorted = [...latencies].sort((a, b) => a - b);
      stats.latencyStats[v] = {
        p50: this.percentile(sorted, 0.5),
        p95: this.percentile(sorted, 0.95),
        p99: this.percentile(sorted, 0.99),
        mean: sorted.reduce((a, b) => a + b, 0) / sorted.length,
        min: sorted[0] || 0,
        max: sorted[sorted.length - 1] || 0,
        count: sorted.length
      };
    }

    // Calculate throughput
    for (const [sessionId, throughput] of this.throughputCounter.entries()) {
      const elapsed = (Date.now() - throughput.startTime) / 1000; // seconds
      const qps = throughput.count / elapsed;
      stats.throughputStats[sessionId] = {
        qps: qps,
        totalRequests: throughput.count,
        elapsedSeconds: elapsed
      };
    }

    // Error statistics
    for (const [errorCode, count] of this.errorCounts.entries()) {
      stats.errorStats[errorCode] = count;
    }

    // SLA violation statistics
    for (const [v, violations] of this.slaViolations.entries()) {
      stats.slaViolationStats[v] = violations;
    }

    return stats;
  }

  /**
   * Export performance data for analysis
   */
  async exportPerformanceData(
    format: 'json' | 'csv' | 'jsonl' = 'jsonl',
    variant?: string,
    timeRange?: { start: string; end: string }
  ): Promise<Result<string, LetheError>> {
    try {
      let filteredEvents = this.eventBuffer;

      // Filter by variant
      if (variant) {
        filteredEvents = filteredEvents.filter(event => event.variant === variant);
      }

      // Filter by time range
      if (timeRange) {
        const startTime = new Date(timeRange.start).getTime();
        const endTime = new Date(timeRange.end).getTime();
        
        filteredEvents = filteredEvents.filter(event => {
          const eventTime = new Date(event.timestamp).getTime();
          return eventTime >= startTime && eventTime <= endTime;
        });
      }

      let exportData: string;

      switch (format) {
        case 'json':
          exportData = JSON.stringify(filteredEvents, null, 2);
          break;
        
        case 'csv':
          exportData = this.convertToCSV(filteredEvents);
          break;
        
        case 'jsonl':
        default:
          exportData = filteredEvents.map(event => JSON.stringify(event)).join('\n');
          break;
      }

      return { success: true, data: exportData };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'EXPORT_FAILED',
          message: `Failed to export performance data: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date().toISOString(),
          context: { format, variant, timeRange }
        }
      };
    }
  }

  /**
   * Start performance monitoring
   */
  private startPerformanceMonitoring(): void {
    // Start periodic flush timer
    this.flushTimer = setInterval(() => {
      this.flush();
    }, this.config.flushInterval);

    // Start system health monitoring
    if (this.config.collectSystemMetrics) {
      this.systemHealthTimer = setInterval(() => {
        this.logEvent(TelemetryEventType.SYSTEM_HEALTH, 'system', {
          level: 'debug',
          context: {
            systemHealth: this.collectSystemHealth()
          }
        });
      }, 30000); // Every 30 seconds
    }

    // Set up performance observer for detailed metrics
    if (this.config.collectGCMetrics && typeof PerformanceObserver !== 'undefined') {
      this.performanceObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.entryType === 'gc') {
            this.logEvent(TelemetryEventType.SYSTEM_HEALTH, 'system', {
              level: 'debug',
              context: {
                gcEvent: {
                  kind: (entry as any).kind,
                  duration: entry.duration,
                  startTime: entry.startTime
                }
              }
            });
          }
        }
      });
      
      this.performanceObserver.observe({ entryTypes: ['gc'] });
    }
  }

  /**
   * Collect system health metrics
   */
  private collectSystemHealth(): SystemHealthMetrics {
    const memoryUsage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    
    return {
      timestamp: new Date().toISOString(),
      memoryUsage: {
        used: memoryUsage.heapUsed,
        free: memoryUsage.heapTotal - memoryUsage.heapUsed,
        total: memoryUsage.heapTotal,
        heapUsed: memoryUsage.heapUsed,
        heapTotal: memoryUsage.heapTotal
      },
      cpuUsage: {
        user: cpuUsage.user / 1000000, // Convert to seconds
        system: cpuUsage.system / 1000000,
        idle: 0 // Not available in Node.js
      },
      processMetrics: {
        uptime: process.uptime(),
        pid: process.pid,
        ppid: process.ppid || 0,
        platform: process.platform,
        arch: process.arch,
        nodeVersion: process.version
      },
      performance: {
        eventLoopDelay: 0, // Would need additional measurement
        gcCount: 0, // Would track from performance observer
        gcDuration: 0
      }
    };
  }

  /**
   * Evaluate SLA compliance
   */
  private evaluateSLA(metrics: OperationMetrics, variant: string): {
    targetLatency: number;
    actualLatency: number;
    targetThroughput: number;
    actualThroughput: number;
    budgetLimit: number;
    budgetUsed: number;
    violated: boolean;
  } {
    const sla = DEFAULT_SLAS[variant];
    if (!sla) {
      return {
        targetLatency: 0,
        actualLatency: metrics.duration,
        targetThroughput: 0,
        actualThroughput: 0,
        budgetLimit: 0,
        budgetUsed: 0,
        violated: false
      };
    }

    const throughput = this.throughputCounter.get(variant);
    const actualThroughput = throughput 
      ? throughput.count / ((Date.now() - throughput.startTime) / 1000)
      : 0;

    const violated = 
      metrics.duration > sla.targetLatencyP95 ||
      actualThroughput < sla.minThroughputQPS ||
      (metrics.memoryAfter / (1024 * 1024)) > sla.maxMemoryUsageMB;

    if (violated) {
      this.slaViolations.set(variant, (this.slaViolations.get(variant) || 0) + 1);
    }

    return {
      targetLatency: sla.targetLatencyP95,
      actualLatency: metrics.duration,
      targetThroughput: sla.minThroughputQPS,
      actualThroughput,
      budgetLimit: sla.budgetLimitRatio,
      budgetUsed: 0, // Would be calculated from budget tracker
      violated
    };
  }

  /**
   * Track performance metrics for analysis
   */
  private trackPerformanceMetrics(metrics: OperationMetrics, variant?: string): void {
    const key = variant || 'unknown';
    
    if (!this.latencyHistogram.has(key)) {
      this.latencyHistogram.set(key, []);
    }
    
    this.latencyHistogram.get(key)!.push(metrics.duration);
  }

  /**
   * Track error occurrence
   */
  private trackError(error: ExtendedLetheError): void {
    const key = `${error.category}:${error.code}`;
    this.errorCounts.set(key, (this.errorCounts.get(key) || 0) + 1);
  }

  /**
   * Calculate percentile from sorted array
   */
  private percentile(sorted: number[], p: number): number {
    if (sorted.length === 0) return 0;
    
    const index = Math.ceil(sorted.length * p) - 1;
    return sorted[Math.max(0, index)];
  }

  /**
   * Convert events to CSV format
   */
  private convertToCSV(events: TelemetryEvent[]): string {
    if (events.length === 0) return '';

    const headers = [
      'timestamp',
      'event',
      'sessionId',
      'variant',
      'level',
      'duration',
      'memoryUsed',
      'cpuUsage',
      'success'
    ];

    const rows = events.map(event => [
      event.timestamp,
      event.event,
      event.sessionId,
      event.variant || '',
      event.level,
      event.metrics?.duration?.toString() || '',
      event.metrics?.memoryAfter?.toString() || '',
      event.metrics?.cpuUsage?.toString() || '',
      event.metrics?.success?.toString() || ''
    ]);

    return [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
  }

  /**
   * Flush event buffer to persistent storage
   */
  private flush(immediate: boolean = false): void {
    if (this.eventBuffer.length === 0) return;

    // In production, this would write to a log file or send to a telemetry service
    const events = [...this.eventBuffer];
    this.eventBuffer = [];

    // For now, just log to console in development
    if (process.env.NODE_ENV === 'development') {
      events.forEach(event => {
        console.log(JSON.stringify(event));
      });
    }

    // In production implementation:
    // - Write to rotating log files
    // - Send to Elasticsearch, DataDog, New Relic, etc.
    // - Compress and archive older data
    // - Implement retries for failed sends
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }

    if (this.systemHealthTimer) {
      clearInterval(this.systemHealthTimer);
    }

    if (this.performanceObserver) {
      this.performanceObserver.disconnect();
    }

    // Final flush
    this.flush(true);
  }
}

/**
 * Performance statistics interface
 */
export interface PerformanceStats {
  timestamp: string;
  totalEvents: number;
  latencyStats: Record<string, {
    p50: number;
    p95: number;
    p99: number;
    mean: number;
    min: number;
    max: number;
    count: number;
  }>;
  throughputStats: Record<string, {
    qps: number;
    totalRequests: number;
    elapsedSeconds: number;
  }>;
  errorStats: Record<string, number>;
  slaViolationStats: Record<string, number>;
  systemHealth: SystemHealthMetrics;
}

/**
 * Benchmark runner for performance testing
 */
export class BenchmarkRunner {
  private telemetry: TelemetrySystem;

  constructor(telemetry: TelemetrySystem) {
    this.telemetry = telemetry;
  }

  /**
   * Run performance benchmark for a specific variant
   */
  async runBenchmark(
    variant: string,
    testFunction: () => Promise<any>,
    options: {
      iterations: number;
      warmupIterations?: number;
      concurrency?: number;
      timeout?: number;
    }
  ): Promise<Result<BenchmarkResult, LetheError>> {
    const { iterations, warmupIterations = 5, concurrency = 1, timeout = 60000 } = options;

    try {
      // Warmup phase
      console.log(`Running ${warmupIterations} warmup iterations...`);
      for (let i = 0; i < warmupIterations; i++) {
        await testFunction();
      }

      // Main benchmark
      console.log(`Running ${iterations} benchmark iterations with concurrency ${concurrency}...`);
      
      const results: OperationMetrics[] = [];
      const startTime = Date.now();

      // Run tests with specified concurrency
      const batches = Math.ceil(iterations / concurrency);
      
      for (let batch = 0; batch < batches; batch++) {
        const batchPromises: Promise<OperationMetrics>[] = [];
        
        const batchSize = Math.min(concurrency, iterations - batch * concurrency);
        
        for (let i = 0; i < batchSize; i++) {
          const iterationNumber = batch * concurrency + i;
          
          batchPromises.push(this.runSingleIteration(
            `benchmark_${variant}_${iterationNumber}`,
            testFunction,
            { variant, iteration: iterationNumber }
          ));
        }

        const batchResults = await Promise.all(batchPromises);
        results.push(...batchResults);

        // Check for timeout
        if (Date.now() - startTime > timeout) {
          throw new Error(`Benchmark timeout exceeded: ${timeout}ms`);
        }
      }

      // Calculate statistics
      const durations = results.map(r => r.duration);
      const memoryUsages = results.map(r => r.memoryAfter - r.memoryBefore);
      const successCount = results.filter(r => r.success).length;

      const benchmarkResult: BenchmarkResult = {
        variant,
        iterations: results.length,
        successRate: successCount / results.length,
        latency: {
          p50: this.percentile(durations.slice().sort((a, b) => a - b), 0.5),
          p95: this.percentile(durations.slice().sort((a, b) => a - b), 0.95),
          p99: this.percentile(durations.slice().sort((a, b) => a - b), 0.99),
          mean: durations.reduce((a, b) => a + b, 0) / durations.length,
          min: Math.min(...durations),
          max: Math.max(...durations)
        },
        memory: {
          mean: memoryUsages.reduce((a, b) => a + b, 0) / memoryUsages.length,
          peak: Math.max(...results.map(r => r.memoryPeak)),
          min: Math.min(...memoryUsages),
          max: Math.max(...memoryUsages)
        },
        throughput: {
          qps: results.length / ((Date.now() - startTime) / 1000)
        },
        slaCompliance: this.evaluateSLACompliance(benchmarkResult, variant)
      };

      return { success: true, data: benchmarkResult };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'BENCHMARK_FAILED',
          message: `Benchmark failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date().toISOString(),
          context: { variant, options }
        }
      };
    }
  }

  /**
   * Run a single benchmark iteration
   */
  private async runSingleIteration(
    operationType: string,
    testFunction: () => Promise<any>,
    metadata: Record<string, unknown>
  ): Promise<OperationMetrics> {
    const tracker = this.telemetry.startTracking(operationType, metadata);
    
    let success = true;
    
    try {
      await testFunction();
    } catch (error) {
      success = false;
    }
    
    return tracker.complete(success);
  }

  /**
   * Calculate percentile
   */
  private percentile(sorted: number[], p: number): number {
    if (sorted.length === 0) return 0;
    const index = Math.ceil(sorted.length * p) - 1;
    return sorted[Math.max(0, index)];
  }

  /**
   * Evaluate SLA compliance for benchmark results
   */
  private evaluateSLACompliance(result: BenchmarkResult, variant: string): SLACompliance {
    const sla = DEFAULT_SLAS[variant];
    
    if (!sla) {
      return {
        latencyCompliant: true,
        memoryCompliant: true,
        throughputCompliant: true,
        overallCompliant: true,
        violations: []
      };
    }

    const violations: string[] = [];
    
    const latencyCompliant = result.latency.p95 <= sla.targetLatencyP95;
    if (!latencyCompliant) {
      violations.push(`P95 latency ${result.latency.p95.toFixed(0)}ms exceeds target ${sla.targetLatencyP95}ms`);
    }

    const memoryCompliant = (result.memory.peak / (1024 * 1024)) <= sla.maxMemoryUsageMB;
    if (!memoryCompliant) {
      violations.push(`Peak memory ${(result.memory.peak / (1024 * 1024)).toFixed(0)}MB exceeds target ${sla.maxMemoryUsageMB}MB`);
    }

    const throughputCompliant = result.throughput.qps >= sla.minThroughputQPS;
    if (!throughputCompliant) {
      violations.push(`Throughput ${result.throughput.qps.toFixed(1)} QPS below target ${sla.minThroughputQPS} QPS`);
    }

    return {
      latencyCompliant,
      memoryCompliant,
      throughputCompliant,
      overallCompliant: violations.length === 0,
      violations
    };
  }
}

/**
 * Benchmark result interface
 */
export interface BenchmarkResult {
  variant: string;
  iterations: number;
  successRate: number;
  latency: {
    p50: number;
    p95: number;
    p99: number;
    mean: number;
    min: number;
    max: number;
  };
  memory: {
    mean: number;
    peak: number;
    min: number;
    max: number;
  };
  throughput: {
    qps: number;
  };
  slaCompliance: SLACompliance;
}

/**
 * SLA compliance result
 */
export interface SLACompliance {
  latencyCompliant: boolean;
  memoryCompliant: boolean;
  throughputCompliant: boolean;
  overallCompliant: boolean;
  violations: string[];
}

/**
 * Singleton telemetry system instance
 */
export const telemetrySystem = new TelemetrySystem();