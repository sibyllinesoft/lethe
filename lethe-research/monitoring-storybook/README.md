# 🔍 Lethe Prompt Monitoring System - Storybook

A comprehensive visual documentation system for the **Lethe Prompt Monitoring Service**, showcasing all monitoring components and interfaces in an interactive Storybook format.

## 🌟 Overview

This Storybook provides a complete visual reference for the Lethe prompt monitoring system, demonstrating:

- **Real-time Dashboard Components** - Summary statistics, timeline charts, performance metrics
- **Data Visualization Components** - Interactive charts built with Recharts
- **Execution Analysis Views** - Detailed prompt execution analysis with before/after comparisons  
- **CLI Interface Components** - Terminal-style representations of command-line tools
- **Responsive Data Tables** - Sortable and filterable performance data displays

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn package manager

### Installation & Setup

```bash
# Navigate to the monitoring storybook directory
cd monitoring-storybook

# Install dependencies
npm install

# Start the Storybook development server
npm run storybook
```

The Storybook will be available at `http://localhost:6006`

### Available Scripts

```bash
# Development
npm run storybook           # Start Storybook dev server on port 6006
npm run build-storybook     # Build static Storybook for deployment
npm run serve-storybook     # Serve built Storybook locally

# Code Quality  
npm run lint               # Run ESLint with auto-fix
npm run type-check         # Run TypeScript compiler checks
```

## 📊 Component Overview

### Dashboard Components

- **`DashboardCard`** - Metric display cards with trend indicators
- **`SummaryStatsGrid`** - Grid layout of key performance metrics
- **`MonitoringDashboard`** - Complete integrated dashboard interface

### Data Visualization

- **`TimelineChart`** - Historical execution trend visualization
- **`PerformanceBubbleChart`** - Prompt performance analysis with bubble plots
- **`ModelComparisonChart`** - Multi-model performance comparison

### Data Tables

- **`PromptPerformanceTable`** - Interactive table with sorting and filtering
- **`ExecutionDetailView`** - Comprehensive execution analysis view

### CLI Components

- **`CLITerminal`** - Terminal-style command interface
- **`CLIOutputCard`** - Individual command result display

## 🎯 Key Features

### Realistic Data

All components use realistic mock data based on the actual Python dataclasses:

- `PromptExecution` - Comprehensive execution tracking
- `PromptComparison` - Before/after analysis results
- `SummaryStats` - System-wide performance metrics
- `TimelineDataPoint` - Historical trend data
- `ModelComparison` - Cross-model performance data

### Interactive Documentation

Each component includes:
- **Multiple story variants** showing different states and configurations
- **Interactive controls** to modify component properties in real-time
- **Comprehensive documentation** explaining usage and integration
- **Accessibility considerations** with proper WCAG compliance

### Modern Technology Stack

- **React 18** - Modern React with hooks and concurrent features
- **TypeScript** - Full type safety and IntelliSense support  
- **Tailwind CSS** - Utility-first responsive styling
- **Recharts** - Powerful chart library for data visualization
- **Lucide React** - Beautiful and consistent icon library
- **Storybook 7** - Latest Storybook with improved performance

## 🏗️ Architecture

### Component Structure

```
src/
├── components/           # React components
│   ├── DashboardCard.tsx
│   ├── SummaryStatsGrid.tsx
│   ├── TimelineChart.tsx
│   ├── PerformanceBubbleChart.tsx
│   ├── ModelComparisonChart.tsx
│   ├── PromptPerformanceTable.tsx
│   ├── ExecutionDetailView.tsx
│   ├── CLITerminal.tsx
│   ├── CLIOutputCard.tsx
│   └── MonitoringDashboard.tsx
├── types/               # TypeScript interfaces
│   └── monitoring.ts
├── data/               # Mock data generators
│   └── mockData.ts
├── styles/             # Global CSS and styling
│   └── global.css
└── *.stories.tsx       # Storybook stories
```

### Type System

The TypeScript interfaces match the Python backend dataclasses:

```typescript
interface PromptExecution {
  execution_id: string;
  prompt_id: string;
  model_name: string;
  execution_time_ms: number;
  response_quality_score?: number;
  error_occurred: boolean;
  // ... and 25+ more fields
}
```

### Data Flow

1. **Mock Data Generation** - Realistic test data based on actual system patterns
2. **Component Props** - Type-safe interfaces for all component inputs
3. **Interactive Stories** - Multiple scenarios demonstrating component behavior
4. **Documentation** - Comprehensive usage examples and integration guides

## 📱 Responsive Design

All components are built with mobile-first responsive design:

- **Desktop** (1200px+) - Full dashboard layout with all features
- **Tablet** (768px-1199px) - Optimized layouts with collapsible sections  
- **Mobile** (320px-767px) - Stacked layouts with touch-friendly interactions

## 🎨 Design System

### Color Palette

- **Primary Blue** - `#3b82f6` (buttons, links, accents)
- **Success Green** - `#10b981` (success states, positive trends)
- **Warning Yellow** - `#f59e0b` (warnings, attention items)
- **Error Red** - `#ef4444` (errors, negative trends)
- **Gray Scale** - `#f8fafc` to `#1f2937` (backgrounds, text, borders)

### Typography

- **Primary Font** - Inter (clean, modern, highly legible)
- **Monospace** - System monospace (code, terminal outputs)
- **Font Sizes** - Consistent scale from 12px to 48px
- **Line Heights** - Optimized for readability

### Component Patterns

- **Cards** - Consistent border-radius, shadow, and padding
- **Buttons** - Clear hover states and focus indicators
- **Forms** - Proper labeling and validation styling
- **Tables** - Zebra striping, sortable headers, responsive design

## 🧪 Testing & Quality

### Accessibility

- **WCAG 2.1 AA** compliance throughout
- **Keyboard navigation** support for all interactive elements
- **Screen reader** compatibility with proper ARIA labels
- **Color contrast** ratios meet accessibility standards
- **Focus indicators** clearly visible for all focusable elements

### Performance

- **Lazy Loading** - Charts and heavy components load on demand
- **Code Splitting** - Optimal bundle sizes for fast loading
- **Responsive Images** - Proper sizing for different screen densities
- **Debounced Interactions** - Smooth filtering and sorting

### Browser Support

- **Chrome** 90+ (primary development target)
- **Firefox** 88+ (full feature support)
- **Safari** 14+ (tested on macOS and iOS)
- **Edge** 90+ (Chromium-based)

## 📚 Story Categories

### Components Stories

Individual component documentation with multiple variants:

- **Basic Usage** - Default configurations and common use cases
- **Data Scenarios** - Different data volumes and edge cases  
- **Interactive States** - Loading, error, and empty states
- **Styling Variants** - Different visual treatments and themes

### Pages Stories

Complete interface demonstrations:

- **Full Dashboard** - Integrated monitoring interface
- **High Volume** - Performance with large datasets
- **Error Scenarios** - System failure and recovery states
- **Mobile Views** - Responsive behavior on small screens

## 🔧 Development

### Adding New Components

1. Create the component in `src/components/`
2. Add TypeScript interfaces in `src/types/`
3. Create comprehensive stories in `ComponentName.stories.tsx`
4. Update mock data generators if needed
5. Add responsive styling with Tailwind CSS

### Customizing Mock Data

Edit `src/data/mockData.ts` to:

- Adjust data patterns and distributions
- Add new data scenarios and edge cases
- Modify realistic value ranges
- Include additional error conditions

### Extending the Design System

Update `src/styles/global.css` to:

- Add new color variants
- Define additional component classes
- Create responsive utilities
- Implement dark mode support (if needed)

## 🚢 Deployment

### Building for Production

```bash
# Create static build
npm run build-storybook

# Serve locally to test
npm run serve-storybook
```

### Deployment Options

- **Netlify** - Drag and drop the `storybook-static` folder
- **Vercel** - Connect the repository for automatic deployments
- **GitHub Pages** - Use the built static files
- **AWS S3** - Upload static files to S3 bucket with web hosting

## 🤝 Integration with Lethe System

This Storybook is designed to complement the actual Lethe prompt monitoring system:

### Backend Integration

The TypeScript interfaces exactly match the Python dataclasses:

```python
# Python (backend)
@dataclass
class PromptExecution:
    execution_id: str
    prompt_id: str
    model_name: str
    # ...
```

```typescript
// TypeScript (frontend)
interface PromptExecution {
  execution_id: string;
  prompt_id: string;  
  model_name: string;
  // ...
}
```

### API Compatibility

Components expect data in the same format as the Python dashboard:

- `get_summary_stats()` → `SummaryStatsGrid`
- `get_execution_timeline()` → `TimelineChart`
- `get_prompt_performance()` → `PromptPerformanceTable`
- `create_prompt_performance_chart()` → `PerformanceBubbleChart`

### CLI Tool Representation

The CLI components accurately represent the actual command-line tool:

- Terminal styling matches the real CLI experience
- Command outputs use actual formatting and emojis
- Error messages and success indicators are identical
- Interactive features simulate real command execution

## 📖 Learning Resources

### Storybook Documentation

- [Storybook Official Docs](https://storybook.js.org/docs)
- [React Component Stories](https://storybook.js.org/docs/react/writing-stories/introduction)
- [Controls and Args](https://storybook.js.org/docs/react/essentials/controls)

### Technology References

- [React 18 Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [Recharts Examples](https://recharts.org/en-US/examples)

## 🐛 Troubleshooting

### Common Issues

**Storybook won't start:**
```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install
npm run storybook
```

**TypeScript errors:**
```bash
# Run type checking
npm run type-check

# Check for missing dependencies
npm ls
```

**Charts not rendering:**
- Ensure proper dimensions are set
- Check console for data format errors
- Verify Recharts dependencies are installed

### Performance Issues

**Slow initial load:**
- Enable Storybook's lazy compilation
- Reduce the number of stories loaded initially
- Optimize mock data generation

**Chart rendering performance:**
- Limit data points for large datasets
- Use data sampling for performance testing
- Enable Recharts optimization features

## 📄 License

This Storybook is part of the Lethe Research project. See the main project license for terms and conditions.

---

**Built with ❤️ by the Lethe Research Team**

*This visual documentation system provides comprehensive insights into the Lethe prompt monitoring interface, helping teams understand, integrate, and extend the monitoring capabilities.*