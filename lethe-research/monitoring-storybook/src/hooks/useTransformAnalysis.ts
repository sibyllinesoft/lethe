import { useMemo } from 'react';
import { TransformChangeV2, ChangeAnalysis } from '../types/transform';

/**
 * Custom hook for analyzing transform changes and generating insights
 * 
 * Features:
 * - Statistical analysis of change patterns
 * - Performance metrics aggregation
 * - Success rate calculation
 * - Change type frequency analysis
 * - Time-based trend detection
 */
export const useTransformAnalysis = (changes: TransformChangeV2[]): ChangeAnalysis => {
  return useMemo(() => {
    if (changes.length === 0) {
      return {
        totalChanges: 0,
        successRate: 0,
        avgPerformanceImpact: 0,
        topChangeTypes: [],
        timeRange: {
          start: new Date(),
          end: new Date()
        }
      };
    }

    // Calculate success rate
    const successfulChanges = changes.filter(change => {
      // Consider a change successful if:
      // 1. It has a confidence score > 0, or
      // 2. It has no confidence score (assume success), or
      // 3. It has performance data with reasonable duration
      const hasGoodConfidence = !change.metadata.confidence_score || change.metadata.confidence_score > 0;
      const hasReasonablePerformance = !change.metadata.performance || 
                                      change.metadata.performance.duration_us < 50000; // Less than 50ms
      return hasGoodConfidence && hasReasonablePerformance;
    }).length;

    const successRate = (successfulChanges / changes.length) * 100;

    // Calculate average performance impact
    const performanceData = changes
      .map(change => change.metadata.performance?.duration_us || 0)
      .filter(duration => duration > 0);
    
    const avgPerformanceImpact = performanceData.length > 0 
      ? performanceData.reduce((sum, duration) => sum + duration, 0) / performanceData.length 
      : 0;

    // Analyze change types
    const changeTypeMap = new Map<string, number>();
    changes.forEach(change => {
      const changeType = Object.keys(change.change_type)[0];
      const readableType = formatChangeType(changeType);
      changeTypeMap.set(readableType, (changeTypeMap.get(readableType) || 0) + 1);
    });

    const topChangeTypes = Array.from(changeTypeMap.entries())
      .map(([type, count]) => ({
        type,
        count,
        percentage: (count / changes.length) * 100
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10); // Top 10 change types

    // Calculate time range
    const timestamps = changes.map(change => new Date(change.timestamp));
    const timeRange = {
      start: new Date(Math.min(...timestamps.map(t => t.getTime()))),
      end: new Date(Math.max(...timestamps.map(t => t.getTime())))
    };

    return {
      totalChanges: changes.length,
      successRate,
      avgPerformanceImpact,
      topChangeTypes,
      timeRange
    };
  }, [changes]);
};

// Helper function to format change types for display
function formatChangeType(changeType: string): string {
  return changeType
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}