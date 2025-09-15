import type { 
  LLMCall, 
  LLMCallSummary, 
  CallFilters, 
  CallMetrics,
  CallComparison,
  DiffResult,
  MetricComparison 
} from '@lethe/llm-analyzer-shared';
import { DatabaseManager } from '../db/database';

export class CallService {
  constructor(private db: DatabaseManager) {}

  async getCalls(filters: CallFilters & {
    page?: number;
    limit?: number;
    sortBy?: string;
    sortOrder?: 'asc' | 'desc';
  }): Promise<{ calls: LLMCallSummary[]; total: number; metrics: CallMetrics }> {
    const { calls, total } = this.db.getCalls(filters);
    const metrics = this.calculateMetrics(calls);
    
    return { calls, total, metrics };
  }

  async getCall(id: string): Promise<LLMCall | null> {
    return this.db.getCall(id);
  }

  async compareCalls(callIds: string[]): Promise<{ calls: LLMCall[]; comparison: CallComparison }> {
    const calls = await Promise.all(
      callIds.map(id => this.db.getCall(id))
    );

    // Filter out null calls
    const validCalls = calls.filter((call): call is LLMCall => call !== null);
    
    if (validCalls.length < 2) {
      throw new Error('Need at least 2 valid calls to compare');
    }

    const comparison = this.generateComparison(validCalls);
    
    return { calls: validCalls, comparison };
  }

  private calculateMetrics(calls: LLMCallSummary[]): CallMetrics {
    const totalCalls = calls.length;
    const successfulCalls = calls.filter(c => c.status >= 200 && c.status < 300).length;
    const errorCalls = totalCalls - successfulCalls;
    
    const averageDuration = calls.length > 0 
      ? calls.reduce((sum, c) => sum + c.duration, 0) / calls.length 
      : 0;
    
    const totalTokens = calls.reduce((sum, c) => sum + (c.totalTokens || 0), 0);
    const totalCost = calls.reduce((sum, c) => sum + (c.cost || 0), 0);
    
    const providersUsed = [...new Set(calls.map(c => c.provider))];
    const modelsUsed = [...new Set(calls.map(c => c.model))];

    return {
      totalCalls,
      successfulCalls,
      errorCalls,
      averageDuration,
      totalTokens,
      totalCost,
      providersUsed,
      modelsUsed
    };
  }

  private generateComparison(calls: LLMCall[]): CallComparison {
    const metricsDiff = this.compareMetrics(calls);
    
    // For request/response diff, compare first two calls
    let requestDiff: DiffResult | undefined;
    let responseDiff: DiffResult | undefined;
    
    if (calls.length >= 2) {
      requestDiff = this.generateDiff(calls[0]?.requestBody, calls[1]?.requestBody);
      responseDiff = this.generateDiff(calls[0]?.responseBody, calls[1]?.responseBody);
    }

    return {
      requestDiff,
      responseDiff,
      metricsDiff
    };
  }

  private compareMetrics(calls: LLMCall[]): MetricComparison {
    const comparison: MetricComparison = {
      duration: this.compareMetricValues(calls.map(c => c.duration))
    };

    const inputTokens = calls.map(c => c.inputTokens).filter((t): t is number => t !== undefined);
    if (inputTokens.length > 0) {
      comparison.inputTokens = this.compareMetricValues(inputTokens);
    }

    const outputTokens = calls.map(c => c.outputTokens).filter((t): t is number => t !== undefined);
    if (outputTokens.length > 0) {
      comparison.outputTokens = this.compareMetricValues(outputTokens);
    }

    const totalTokens = calls.map(c => c.totalTokens).filter((t): t is number => t !== undefined);
    if (totalTokens.length > 0) {
      comparison.totalTokens = this.compareMetricValues(totalTokens);
    }

    const costs = calls.map(c => c.cost).filter((c): c is number => c !== undefined);
    if (costs.length > 0) {
      comparison.cost = this.compareMetricValues(costs);
    }

    return comparison;
  }

  private compareMetricValues(values: number[]) {
    if (values.length < 2) {
      return {
        values,
        difference: 0,
        percentageChange: 0
      };
    }

    const difference = values[1]! - values[0]!;
    const percentageChange = values[0] !== 0 ? (difference / values[0]!) * 100 : 0;

    return {
      values,
      difference,
      percentageChange
    };
  }

  private generateDiff(obj1: unknown, obj2: unknown): DiffResult {
    // Simple diff implementation - in a real app you might use a library like deep-diff
    const added: string[] = [];
    const removed: string[] = [];
    const modified: Array<{ path: string; oldValue: unknown; newValue: unknown }> = [];

    // For now, just do a simple JSON comparison
    if (JSON.stringify(obj1) !== JSON.stringify(obj2)) {
      modified.push({
        path: 'root',
        oldValue: obj1,
        newValue: obj2
      });
    }

    return { added, removed, modified };
  }
}