import type { Meta, StoryObj } from '@storybook/react';
import { CLIOutputCard } from './CLIOutputCard';
import { mockDataset } from '../data/mockData';

const meta: Meta<typeof CLIOutputCard> = {
  title: 'Components/CLIOutputCard',
  component: CLIOutputCard,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: 'Card component for displaying individual CLI command outputs with syntax highlighting and copy functionality.'
      }
    }
  },
  tags: ['autodocs'],
  argTypes: {
    compact: {
      control: 'boolean'
    }
  }
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    output: mockDataset.cliOutputs.status,
    compact: false
  }
};

export const Compact: Story = {
  args: {
    output: mockDataset.cliOutputs.status,
    compact: true
  }
};

export const ListPrompts: Story = {
  args: {
    output: mockDataset.cliOutputs.list,
    compact: false
  }
};

export const AnalyzeCommand: Story = {
  args: {
    output: mockDataset.cliOutputs.analyze,
    compact: false
  }
};

export const ErrorCommand: Story = {
  args: {
    output: {
      command: 'prompt-monitor analyze nonexistent_prompt',
      timestamp: '2024-01-15T14:30:00Z',
      output: [
        '🔍 Analyzing Prompt: nonexistent_prompt',
        '============================================================',
        '❌ Error: Prompt "nonexistent_prompt" not found in database',
        '',
        'Available prompts:',
        '  - code_generation_abc123',
        '  - text_analysis_def456', 
        '  - creative_writing_ghi789',
        '  - question_answering_jkl012',
        '',
        'Use "prompt-monitor list-prompts" to see all available prompts.',
        'Use "prompt-monitor --help" for more commands.'
      ],
      status: 'error' as const,
      duration_ms: 156
    },
    compact: false
  }
};

export const WarningCommand: Story = {
  args: {
    output: {
      command: 'prompt-monitor cleanup --days 7',
      timestamp: '2024-01-15T14:30:00Z',
      output: [
        '🧹 Cleaning up data older than 7 days (2024-01-08)',
        '⚠️ Warning: This will permanently delete 2,341 execution records',
        '⚠️ Warning: Some prompt analytics may be affected',
        '',
        'Are you sure you want to continue? (y/N)',
        '',
        '💡 Consider using --days 30 to retain more historical data',
        '💡 Use --dry-run to preview what will be deleted'
      ],
      status: 'warning' as const,
      duration_ms: 89
    },
    compact: false
  }
};

export const LongOutput: Story = {
  args: {
    output: {
      command: 'prompt-monitor export --format json --verbose',
      timestamp: '2024-01-15T14:30:00Z',
      output: [
        '📤 Exporting prompt execution data...',
        '================================================================================',
        '📊 Export Configuration:',
        '  Format: JSON',
        '  Include metadata: Yes',
        '  Include environment variables: Yes',
        '  Include error details: Yes',
        '  Date range: All time',
        '',
        '🔍 Scanning database...',
        '  Found 15,432 execution records',
        '  Found 127 unique prompts',
        '  Found 1,234 comparison records',
        '',
        '📝 Processing records...',
        '  [████████████████████████████████████████] 100% (15,432/15,432)',
        '',
        '💾 Writing to file...',
        '  Output file: exports/prompt_data_20240115_143000.json',
        '  File size: 2.34 MB',
        '  Compression: Enabled (67% reduction)',
        '',
        '✅ Export completed successfully!',
        '📁 File location: /home/user/lethe/exports/prompt_data_20240115_143000.json',
        '📊 Summary:',
        '  Total executions: 15,432',
        '  Date range: 2023-12-01 to 2024-01-15',
        '  Export duration: 3.2 seconds',
        '  Average processing rate: 4,822 records/second'
      ],
      status: 'success' as const,
      duration_ms: 3200
    },
    compact: false
  }
};

export const FastCommand: Story = {
  args: {
    output: {
      command: 'prompt-monitor --version',
      timestamp: '2024-01-15T14:30:00Z',
      output: [
        'Lethe Prompt Monitor v1.2.3',
        'Built on 2024-01-10',
        'Python 3.11.0'
      ],
      status: 'success' as const,
      duration_ms: 23
    },
    compact: false
  }
};