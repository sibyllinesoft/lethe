export interface DiffLine {
  type: 'added' | 'removed' | 'unchanged';
  content: string;
  lineNumber?: number;
}

export interface DiffResult {
  lines: DiffLine[];
  hasChanges: boolean;
}

/**
 * Simple diff implementation for comparing JSON objects
 */
export function createDiff(oldValue: unknown, newValue: unknown): DiffResult {
  const oldStr = JSON.stringify(oldValue, null, 2) || '';
  const newStr = JSON.stringify(newValue, null, 2) || '';
  
  const oldLines = oldStr.split('\n');
  const newLines = newStr.split('\n');
  
  const lines: DiffLine[] = [];
  let hasChanges = false;
  
  // Simple line-by-line comparison
  // In a real implementation, you might use a more sophisticated diff algorithm
  const maxLines = Math.max(oldLines.length, newLines.length);
  
  for (let i = 0; i < maxLines; i++) {
    const oldLine = oldLines[i];
    const newLine = newLines[i];
    
    if (oldLine === newLine) {
      lines.push({
        type: 'unchanged',
        content: oldLine || '',
        lineNumber: i + 1
      });
    } else if (oldLine === undefined) {
      lines.push({
        type: 'added',
        content: newLine || '',
        lineNumber: i + 1
      });
      hasChanges = true;
    } else if (newLine === undefined) {
      lines.push({
        type: 'removed',
        content: oldLine || '',
        lineNumber: i + 1
      });
      hasChanges = true;
    } else {
      // Line changed
      lines.push({
        type: 'removed',
        content: oldLine,
        lineNumber: i + 1
      });
      lines.push({
        type: 'added',
        content: newLine,
        lineNumber: i + 1
      });
      hasChanges = true;
    }
  }
  
  return { lines, hasChanges };
}

/**
 * Format a value for display in diff view
 */
export function formatForDiff(value: unknown): string {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2);
}

/**
 * Get a summary of changes between two values
 */
export function getDiffSummary(oldValue: unknown, newValue: unknown): string {
  const diff = createDiff(oldValue, newValue);
  
  if (!diff.hasChanges) {
    return 'No changes';
  }
  
  const added = diff.lines.filter(line => line.type === 'added').length;
  const removed = diff.lines.filter(line => line.type === 'removed').length;
  
  const parts = [];
  if (added > 0) parts.push(`+${added} lines`);
  if (removed > 0) parts.push(`-${removed} lines`);
  
  return parts.join(', ');
}

/**
 * Truncate diff for preview purposes
 */
export function truncateDiff(diff: DiffResult, maxLines: number = 10): DiffResult {
  if (diff.lines.length <= maxLines) {
    return diff;
  }
  
  return {
    ...diff,
    lines: [
      ...diff.lines.slice(0, maxLines),
      {
        type: 'unchanged',
        content: `... (${diff.lines.length - maxLines} more lines)`,
      }
    ]
  };
}