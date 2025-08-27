import type { Meta, StoryObj } from '@storybook/react';
import { CLITerminal } from './CLITerminal';
import { mockDataset } from '../data/mockData';

const meta: Meta<typeof CLITerminal> = {
  title: 'Components/CLITerminal',
  component: CLITerminal,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: 'Terminal-style component that displays CLI command outputs with syntax highlighting and interactive features.'
      }
    }
  },
  tags: ['autodocs'],
  argTypes: {
    interactive: {
      control: 'boolean'
    }
  }
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    outputs: [
      mockDataset.cliOutputs.status,
      mockDataset.cliOutputs.list,
      mockDataset.cliOutputs.analyze
    ],
    interactive: false
  }
};

export const Interactive: Story = {
  args: {
    outputs: [mockDataset.cliOutputs.status],
    interactive: true
  }
};

export const SingleCommand: Story = {
  args: {
    outputs: [mockDataset.cliOutputs.status],
    interactive: false
  }
};

export const ErrorCommand: Story = {
  args: {
    outputs: [
      {
        command: 'prompt-monitor analyze nonexistent_prompt',
        timestamp: '2024-01-15T14:30:00Z',
        output: [
          '🔍 Analyzing Prompt: nonexistent_prompt',
          '============================================================',
          '❌ Error: Prompt not found in database',
          '',
          'Available prompts:',
          '  - code_generation_abc123',
          '  - text_analysis_def456',
          '  - creative_writing_ghi789',
          '',
          'Use "prompt-monitor list-prompts" to see all available prompts.'
        ],
        status: 'error' as const,
        duration_ms: 123
      }
    ],
    interactive: false
  }
};

export const LongOutput: Story = {
  args: {
    outputs: [
      {
        command: 'prompt-monitor list-prompts --verbose',
        timestamp: '2024-01-15T14:30:00Z',
        output: [
          '📋 Tracked Prompts (Detailed)',
          '================================================================================',
          'Prompt ID: code_generation_abc123',
          '  Executions: 45',
          '  Success Rate: 98.5%',
          '  Average Time: 324ms',
          '  Quality Score: 0.892',
          '  Last Used: 2024-01-15T14:30:00',
          '  Template: code_generation',
          '  Variables: {language: python, complexity: medium}',
          '',
          'Prompt ID: text_analysis_def456',
          '  Executions: 32',
          '  Success Rate: 95.2%',
          '  Average Time: 567ms',
          '  Quality Score: 0.845',
          '  Last Used: 2024-01-15T12:15:00',
          '  Template: text_analysis',
          '  Variables: {domain: scientific, length: long}',
          '',
          'Prompt ID: creative_writing_ghi789',
          '  Executions: 28',
          '  Success Rate: 97.1%',
          '  Average Time: 1,245ms',
          '  Quality Score: 0.923',
          '  Last Used: 2024-01-14T16:45:00',
          '  Template: creative_writing',
          '  Variables: {style: narrative, genre: science_fiction}',
          '',
          '📊 Summary Statistics:',
          '  Total Prompts: 3',
          '  Total Executions: 105',
          '  Overall Success Rate: 96.7%',
          '  Average Quality: 0.887'
        ],
        status: 'success' as const,
        duration_ms: 234
      }
    ],
    interactive: false
  }
};

export const MultipleCommands: Story = {
  args: {
    outputs: [
      mockDataset.cliOutputs.status,
      mockDataset.cliOutputs.list,
      {
        command: 'prompt-monitor export --format json',
        timestamp: '2024-01-15T14:35:00Z',
        output: [
          '📤 Exporting data in JSON format...',
          '✅ Data exported to: exports/prompt_data_20240115_143500.json',
          '📁 File size: 2.3 MB',
          '📊 Records exported: 15,432 executions',
          '🕐 Export completed in 1.2s'
        ],
        status: 'success' as const,
        duration_ms: 1200
      }
    ],
    interactive: true
  }
};