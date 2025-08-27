import React, { useState } from 'react';
import { 
  ChevronUp, 
  ChevronDown, 
  Search,
  Filter,
  Clock,
  CheckCircle,
  AlertCircle
} from 'lucide-react';
import { PromptPerformance, SortConfig, TableFilters } from '../types/monitoring';
import { format } from 'date-fns';
import { 
  Card, 
  Text, 
  Heading, 
  Button, 
  Badge, 
  DataTable, 
  designTokens, 
  performanceColors 
} from '../design-system';

interface PromptPerformanceTableProps {
  data: PromptPerformance[];
  className?: string;
}

/**
 * Interactive table showing prompt performance data
 * Based on PromptDashboard.get_prompt_performance() output
 * Includes sorting, filtering, and search functionality
 */
export const PromptPerformanceTable: React.FC<PromptPerformanceTableProps> = ({ 
  data, 
  className 
}) => {
  const [sortConfig, setSortConfig] = useState<SortConfig>({ 
    key: 'execution_count', 
    direction: 'desc' 
  });
  const [filters, setFilters] = useState<TableFilters>({});
  const [showFilters, setShowFilters] = useState(false);

  // Sorting function
  const sortedData = React.useMemo(() => {
    const sorted = [...data].sort((a, b) => {
      const aVal = (a as any)[sortConfig.key];
      const bVal = (b as any)[sortConfig.key];
      
      if (typeof aVal === 'string') {
        return sortConfig.direction === 'asc' 
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }
      
      return sortConfig.direction === 'asc' 
        ? aVal - bVal 
        : bVal - aVal;
    });
    
    // Apply search filter
    if (filters.search) {
      return sorted.filter(item => 
        item.prompt_id.toLowerCase().includes(filters.search!.toLowerCase())
      );
    }
    
    return sorted;
  }, [data, sortConfig, filters.search]);

  const handleSort = (key: string) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const SortIcon = ({ column }: { column: string }) => {
    if (sortConfig.key !== column) {
      return (
        <ChevronUp 
          className="h-4 w-4" 
          style={{ color: designTokens.colors.graphite[300] }}
        />
      );
    }
    return sortConfig.direction === 'asc' 
      ? <ChevronUp 
          className="h-4 w-4" 
          style={{ color: designTokens.colors.data.primary[0] }}
        />
      : <ChevronDown 
          className="h-4 w-4" 
          style={{ color: designTokens.colors.data.primary[0] }}
        />;
  };

  const formatTime = (ms: number): string => {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const formatDate = (dateString: string): string => {
    return format(new Date(dateString), 'MMM dd, HH:mm');
  };

  const getSuccessRateBadgeVariant = (rate: number): 'success' | 'warning' | 'error' => {
    if (rate >= 95) return 'success';
    if (rate >= 90) return 'warning';
    return 'error';
  };

  const getPerformanceColor = (time: number): string => {
    if (time < 500) return designTokens.colors.success[600];
    if (time < 1000) return designTokens.colors.warning[600];
    return designTokens.colors.error[600];
  };

  const getQualityColor = (quality: number): string => {
    if (quality >= 0.9) return performanceColors.excellent;
    if (quality >= 0.8) return performanceColors.good;
    if (quality >= 0.7) return performanceColors.fair;
    if (quality >= 0.6) return performanceColors.poor;
    return performanceColors.critical;
  };

  return (
    <Card className={className}>
      {/* Header */}
      <div className="p-6 border-b border-graphite-200">
        <div className="flex items-center justify-between">
          <Heading level={3} className="text-graphite-900">
            Prompt Performance Analytics
          </Heading>
          <div className="flex items-center gap-4">
            <div className="relative">
              <Search 
                className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2" 
                style={{ color: designTokens.colors.graphite[400] }}
              />
              <input
                type="text"
                placeholder="Search prompts..."
                value={filters.search || ''}
                onChange={(e) => setFilters(prev => ({ ...prev, search: e.target.value }))}
                className="pl-10 pr-4 py-2 rounded-md transition-all duration-200 focus-ring"
                style={{
                  border: `1px solid ${designTokens.colors.graphite[300]}`,
                  backgroundColor: designTokens.colors.surface.primary,
                  color: designTokens.colors.graphite[900]
                }}
              />
            </div>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => setShowFilters(!showFilters)}
            >
              <Filter className="h-4 w-4 mr-2" />
              Filters
            </Button>
          </div>
        </div>

        {/* Summary Stats */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="flex items-center gap-3">
            <CheckCircle 
              className="h-5 w-5 flex-shrink-0" 
              style={{ color: designTokens.colors.success[500] }}
            />
            <div>
              <Text weight="semibold" color="primary" className="leading-none">
                {sortedData.length} Prompts
              </Text>
              <Text size="xs" color="tertiary" className="mt-1">Total tracked</Text>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Clock 
              className="h-5 w-5 flex-shrink-0" 
              style={{ color: designTokens.colors.data.primary[0] }}
            />
            <div>
              <Text weight="semibold" color="primary" className="leading-none">
                {formatTime(sortedData.reduce((sum, p) => sum + p.avg_execution_time, 0) / sortedData.length)}
              </Text>
              <Text size="xs" color="tertiary" className="mt-1">Avg execution time</Text>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <AlertCircle 
              className="h-5 w-5 flex-shrink-0" 
              style={{ color: designTokens.colors.warning[500] }}
            />
            <div>
              <Text weight="semibold" color="primary" className="leading-none">
                {((sortedData.reduce((sum, p) => sum + p.success_rate, 0) / sortedData.length)).toFixed(1)}%
              </Text>
              <Text size="xs" color="tertiary" className="mt-1">Avg success rate</Text>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div 
              className="w-5 h-5 rounded flex-shrink-0" 
              style={{ 
                background: `linear-gradient(45deg, ${designTokens.colors.data.primary[0]}, ${designTokens.colors.data.primary[4]})` 
              }}
            />
            <div>
              <Text weight="semibold" color="primary" className="leading-none">
                {sortedData.reduce((sum, p) => sum + p.execution_count, 0).toLocaleString()}
              </Text>
              <Text size="xs" color="tertiary" className="mt-1">Total executions</Text>
            </div>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead style={{ backgroundColor: designTokens.colors.graphite[50] }}>
            <tr>
              <th 
                className="table-header"
                onClick={() => handleSort('prompt_id')}
              >
                <div className="flex items-center gap-1">
                  <Text size="xs" weight="medium" color="secondary" className="uppercase tracking-wider">
                    Prompt ID
                  </Text>
                  <SortIcon column="prompt_id" />
                </div>
              </th>
              <th 
                className="table-header"
                onClick={() => handleSort('execution_count')}
              >
                <div className="flex items-center gap-1">
                  <Text size="xs" weight="medium" color="secondary" className="uppercase tracking-wider">
                    Executions
                  </Text>
                  <SortIcon column="execution_count" />
                </div>
              </th>
              <th 
                className="table-header"
                onClick={() => handleSort('avg_execution_time')}
              >
                <div className="flex items-center gap-1">
                  <Text size="xs" weight="medium" color="secondary" className="uppercase tracking-wider">
                    Avg Time
                  </Text>
                  <SortIcon column="avg_execution_time" />
                </div>
              </th>
              <th 
                className="table-header"
                onClick={() => handleSort('success_rate')}
              >
                <div className="flex items-center gap-1">
                  <Text size="xs" weight="medium" color="secondary" className="uppercase tracking-wider">
                    Success Rate
                  </Text>
                  <SortIcon column="success_rate" />
                </div>
              </th>
              <th 
                className="table-header"
                onClick={() => handleSort('avg_quality_score')}
              >
                <div className="flex items-center gap-1">
                  <Text size="xs" weight="medium" color="secondary" className="uppercase tracking-wider">
                    Quality
                  </Text>
                  <SortIcon column="avg_quality_score" />
                </div>
              </th>
              <th 
                className="table-header"
                onClick={() => handleSort('error_count')}
              >
                <div className="flex items-center gap-1">
                  <Text size="xs" weight="medium" color="secondary" className="uppercase tracking-wider">
                    Errors
                  </Text>
                  <SortIcon column="error_count" />
                </div>
              </th>
              <th 
                className="table-header"
                onClick={() => handleSort('last_used')}
              >
                <div className="flex items-center gap-1">
                  <Text size="xs" weight="medium" color="secondary" className="uppercase tracking-wider">
                    Last Used
                  </Text>
                  <SortIcon column="last_used" />
                </div>
              </th>
            </tr>
          </thead>
          <tbody 
            style={{ 
              backgroundColor: designTokens.colors.surface.primary,
              borderTop: `1px solid ${designTokens.colors.graphite[200]}`
            }}
          >
            {sortedData.map((prompt) => (
              <tr 
                key={prompt.prompt_id} 
                className="transition-colors duration-200 border-b border-graphite-100 hover:bg-graphite-25"
              >
                <td className="table-cell">
                  <div>
                    <Text weight="medium" color="primary" className="leading-none">
                      {prompt.prompt_id}
                    </Text>
                    <Text size="sm" color="tertiary" className="mt-1">
                      {Math.round(prompt.avg_response_length)} chars avg
                    </Text>
                  </div>
                </td>
                <td className="table-cell">
                  <Text weight="medium" color="primary">
                    {prompt.execution_count.toLocaleString()}
                  </Text>
                </td>
                <td className="table-cell">
                  <Text 
                    weight="medium" 
                    style={{ color: getPerformanceColor(prompt.avg_execution_time) }}
                  >
                    {formatTime(prompt.avg_execution_time)}
                  </Text>
                </td>
                <td className="table-cell">
                  <Badge 
                    variant={getSuccessRateBadgeVariant(prompt.success_rate)}
                    size="sm"
                  >
                    {prompt.success_rate.toFixed(1)}%
                  </Badge>
                </td>
                <td className="table-cell">
                  {prompt.avg_quality_score ? (
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-2 rounded-full overflow-hidden" style={{ backgroundColor: designTokens.colors.graphite[200] }}>
                        <div 
                          className="h-full rounded-full transition-all duration-300" 
                          style={{ 
                            width: `${prompt.avg_quality_score * 100}%`,
                            backgroundColor: getQualityColor(prompt.avg_quality_score)
                          }}
                        />
                      </div>
                      <Text size="xs" color="secondary">
                        {prompt.avg_quality_score.toFixed(3)}
                      </Text>
                    </div>
                  ) : (
                    <Text size="sm" color="tertiary">N/A</Text>
                  )}
                </td>
                <td className="table-cell">
                  <Text 
                    weight="medium"
                    style={{ 
                      color: prompt.error_count > 0 
                        ? designTokens.colors.error[600] 
                        : designTokens.colors.graphite[500] 
                    }}
                  >
                    {prompt.error_count}
                  </Text>
                </td>
                <td className="table-cell">
                  <Text size="sm" color="secondary">
                    {formatDate(prompt.last_used)}
                  </Text>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Empty state */}
      {sortedData.length === 0 && (
        <div className="text-center py-12">
          <AlertCircle 
            className="mx-auto h-12 w-12" 
            style={{ color: designTokens.colors.graphite[400] }}
          />
          <Heading level={4} className="mt-4 text-graphite-900">
            No prompts found
          </Heading>
          <Text color="secondary" className="mt-2">
            Try adjusting your search criteria.
          </Text>
        </div>
      )}
    </Card>
  );
};

export default PromptPerformanceTable;