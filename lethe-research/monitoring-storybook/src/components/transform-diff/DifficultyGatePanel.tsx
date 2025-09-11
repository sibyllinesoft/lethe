import React, { useMemo } from 'react';
import { TransformChangeV2, DifficultyGateMetrics } from '../../types/transform';
import { AlertTriangle, Zap, TrendingUp, Settings, Info } from 'lucide-react';
import clsx from 'clsx';

interface DifficultyGatePanelProps {
  metrics: DifficultyGateMetrics;
  changes: TransformChangeV2[];
  className?: string;
}

/**
 * DifficultyGatePanel - Real-time difficulty gate analysis and recommendations
 * 
 * Features:
 * - Change entropy calculation (Shannon entropy of change types)
 * - Rollback frequency tracking
 * - Edit depth analysis
 * - Dynamic K2 cap recommendations
 * - Dimension selection guidance (256 vs 768)
 * - Visual complexity indicators
 */
export const DifficultyGatePanel: React.FC<DifficultyGatePanelProps> = ({
  metrics,
  changes,
  className
}) => {
  // Calculate real-time metrics from changes
  const calculatedMetrics = useMemo(() => {
    if (changes.length === 0) {
      return {
        changeEntropy: 0,
        rollbackFrequency: 0,
        editDepth: 0,
        complexityScore: 0
      };
    }

    // Calculate Shannon entropy of change types
    const changeTypeFreq = new Map<string, number>();
    changes.forEach(change => {
      const changeType = Object.keys(change.change_type)[0];
      changeTypeFreq.set(changeType, (changeTypeFreq.get(changeType) || 0) + 1);
    });

    const totalChanges = changes.length;
    let entropy = 0;
    changeTypeFreq.forEach(freq => {
      const probability = freq / totalChanges;
      entropy -= probability * Math.log2(probability);
    });

    // Calculate rollback frequency (based on low confidence scores)
    const lowConfidenceChanges = changes.filter(change => 
      change.metadata.confidence_score !== undefined && change.metadata.confidence_score < 0.5
    ).length;
    const rollbackFrequency = lowConfidenceChanges / totalChanges;

    // Calculate edit depth (based on content size changes)
    const editDepths = changes
      .map(change => {
        const perf = change.metadata.performance;
        if (!perf || perf.input_size_bytes === 0) return 0;
        return Math.abs(perf.output_size_bytes - perf.input_size_bytes) / perf.input_size_bytes;
      })
      .filter(depth => depth > 0);

    const avgEditDepth = editDepths.length > 0 
      ? editDepths.reduce((sum, depth) => sum + depth, 0) / editDepths.length 
      : 0;

    // Composite complexity score
    const complexityScore = (entropy * 0.4) + (rollbackFrequency * 0.3) + (avgEditDepth * 0.3);

    return {
      changeEntropy: entropy,
      rollbackFrequency,
      editDepth: avgEditDepth,
      complexityScore
    };
  }, [changes]);

  // Merge calculated metrics with provided metrics
  const finalMetrics = {
    changeEntropy: metrics.changeEntropy || calculatedMetrics.changeEntropy,
    rollbackFrequency: metrics.rollbackFrequency || calculatedMetrics.rollbackFrequency,
    editDepth: metrics.editDepth || calculatedMetrics.editDepth,
    complexityScore: metrics.complexityScore || calculatedMetrics.complexityScore,
    recommendedK2Cap: metrics.recommendedK2Cap,
    recommendedDimension: metrics.recommendedDimension
  };

  // Generate recommendations based on complexity
  const recommendations = useMemo(() => {
    const { complexityScore, changeEntropy, rollbackFrequency, editDepth } = finalMetrics;
    const recommendations = [];

    // K2 cap recommendation
    if (complexityScore > 0.7) {
      recommendations.push({
        type: 'critical',
        title: 'Reduce K2 Cap',
        description: `High complexity (${complexityScore.toFixed(2)}). Recommend K2 cap of 1024 tokens.`,
        icon: AlertTriangle
      });
    } else if (complexityScore > 0.5) {
      recommendations.push({
        type: 'warning',
        title: 'Moderate K2 Cap',
        description: `Moderate complexity. Recommend K2 cap of 2048 tokens.`,
        icon: TrendingUp
      });
    } else {
      recommendations.push({
        type: 'info',
        title: 'Standard K2 Cap',
        description: `Low complexity. Standard K2 cap of 4096 tokens is appropriate.`,
        icon: Info
      });
    }

    // Dimension recommendation
    if (changeEntropy > 2.5 || editDepth > 0.3) {
      recommendations.push({
        type: 'info',
        title: 'Use 768 Dimensions',
        description: 'High entropy or edit depth detected. 768-dim embeddings recommended.',
        icon: Settings
      });
    } else {
      recommendations.push({
        type: 'info',
        title: 'Use 256 Dimensions',
        description: 'Simple changes detected. 256-dim embeddings are sufficient.',
        icon: Settings
      });
    }

    // Rollback frequency warning
    if (rollbackFrequency > 0.2) {
      recommendations.push({
        type: 'warning',
        title: 'High Rollback Rate',
        description: `${(rollbackFrequency * 100).toFixed(1)}% rollback rate. Review transform logic.`,
        icon: AlertTriangle
      });
    }

    return recommendations;
  }, [finalMetrics]);

  // Get color based on metric value
  const getMetricColor = (value: number, thresholds: [number, number]) => {
    if (value > thresholds[1]) return 'text-red-600 dark:text-red-400';
    if (value > thresholds[0]) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-green-600 dark:text-green-400';
  };

  const getMetricBg = (value: number, thresholds: [number, number]) => {
    if (value > thresholds[1]) return 'bg-red-50 dark:bg-red-900/20';
    if (value > thresholds[0]) return 'bg-yellow-50 dark:bg-yellow-900/20';
    return 'bg-green-50 dark:bg-green-900/20';
  };

  return (
    <div className={clsx('bg-white dark:bg-gray-900 rounded-lg p-4', className)}>
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
        <Zap className="mr-2" size={20} />
        Difficulty Gate Analysis
      </h3>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className={clsx(
          'rounded p-3',
          getMetricBg(finalMetrics.changeEntropy, [1.5, 2.5])
        )}>
          <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Change Entropy
          </div>
          <div className={clsx(
            'text-lg font-bold',
            getMetricColor(finalMetrics.changeEntropy, [1.5, 2.5])
          )}>
            {finalMetrics.changeEntropy.toFixed(2)}
          </div>
        </div>

        <div className={clsx(
          'rounded p-3',
          getMetricBg(finalMetrics.rollbackFrequency, [0.1, 0.2])
        )}>
          <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Rollback Rate
          </div>
          <div className={clsx(
            'text-lg font-bold',
            getMetricColor(finalMetrics.rollbackFrequency, [0.1, 0.2])
          )}>
            {(finalMetrics.rollbackFrequency * 100).toFixed(1)}%
          </div>
        </div>

        <div className={clsx(
          'rounded p-3',
          getMetricBg(finalMetrics.editDepth, [0.2, 0.4])
        )}>
          <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Edit Depth
          </div>
          <div className={clsx(
            'text-lg font-bold',
            getMetricColor(finalMetrics.editDepth, [0.2, 0.4])
          )}>
            {finalMetrics.editDepth.toFixed(3)}
          </div>
        </div>

        <div className={clsx(
          'rounded p-3',
          getMetricBg(finalMetrics.complexityScore, [0.5, 0.7])
        )}>
          <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Complexity
          </div>
          <div className={clsx(
            'text-lg font-bold',
            getMetricColor(finalMetrics.complexityScore, [0.5, 0.7])
          )}>
            {finalMetrics.complexityScore.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="space-y-3">
        <h4 className="font-medium text-gray-900 dark:text-white">Recommendations</h4>
        {recommendations.map((rec, index) => (
          <div
            key={index}
            className={clsx(
              'flex items-start p-3 rounded-lg border-l-4',
              rec.type === 'critical' && 'bg-red-50 dark:bg-red-900/20 border-red-500',
              rec.type === 'warning' && 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-500',
              rec.type === 'info' && 'bg-blue-50 dark:bg-blue-900/20 border-blue-500'
            )}
          >
            <rec.icon 
              className={clsx(
                'mt-0.5 mr-3 flex-shrink-0',
                rec.type === 'critical' && 'text-red-600 dark:text-red-400',
                rec.type === 'warning' && 'text-yellow-600 dark:text-yellow-400',
                rec.type === 'info' && 'text-blue-600 dark:text-blue-400'
              )} 
              size={16} 
            />
            <div>
              <div className="font-medium text-gray-900 dark:text-white text-sm">
                {rec.title}
              </div>
              <div className="text-gray-600 dark:text-gray-300 text-sm">
                {rec.description}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Current settings */}
      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600 dark:text-gray-300">Recommended K2 Cap:</span>
            <span className="ml-2 font-medium text-gray-900 dark:text-white">
              {finalMetrics.recommendedK2Cap || 'Auto'} tokens
            </span>
          </div>
          <div>
            <span className="text-gray-600 dark:text-gray-300">Recommended Dimensions:</span>
            <span className="ml-2 font-medium text-gray-900 dark:text-white">
              {finalMetrics.recommendedDimension || 'Auto'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};