import type { Meta, StoryObj } from '@storybook/react';
import { Activity, Clock, CheckCircle, Zap, Users } from 'lucide-react';
import { DashboardCard } from './DashboardCard';

const meta: Meta<typeof DashboardCard> = {
  title: 'Components/DashboardCard',
  component: DashboardCard,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: 'A metric card component for displaying key statistics with optional trend indicators.'
      }
    }
  },
  tags: ['autodocs'],
  argTypes: {
    trend: {
      control: 'select',
      options: ['up', 'down', 'stable']
    }
  }
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    title: 'Total Executions',
    value: '15,432',
    subtitle: 'All-time prompt executions',
    icon: <Activity className="h-8 w-8" />
  }
};

export const WithUpTrend: Story = {
  args: {
    title: 'Success Rate',
    value: '96.8%',
    subtitle: 'Successful executions',
    icon: <CheckCircle className="h-8 w-8" />,
    trend: 'up',
    trendValue: '+2.3% from last week'
  }
};

export const WithDownTrend: Story = {
  args: {
    title: 'Average Response Time',
    value: '1.2s',
    subtitle: 'Mean execution time',
    icon: <Zap className="h-8 w-8" />,
    trend: 'down',
    trendValue: '+15ms slower than last week'
  }
};

export const WithStableTrend: Story = {
  args: {
    title: 'Active Users',
    value: '127',
    subtitle: 'Users in last 7 days',
    icon: <Users className="h-8 w-8" />,
    trend: 'stable',
    trendValue: 'No significant change'
  }
};

export const NumericValue: Story = {
  args: {
    title: 'Recent Activity',
    value: 89,
    subtitle: 'Last 24 hours',
    icon: <Clock className="h-8 w-8" />,
    trend: 'up',
    trendValue: 'Active'
  }
};

export const CustomStyling: Story = {
  args: {
    title: 'Custom Metric',
    value: '42',
    subtitle: 'With custom styling',
    icon: <Activity className="h-8 w-8" />,
    className: 'border-2 border-blue-200 shadow-lg'
  }
};