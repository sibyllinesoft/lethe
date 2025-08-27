import React from 'react';
import { clsx } from 'clsx';
import { ChevronUp, ChevronDown, ArrowUpDown } from 'lucide-react';

// Table Component Types
interface TableProps {
  children: React.ReactNode;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  striped?: boolean;
  bordered?: boolean;
  hover?: boolean;
}

interface TableHeaderProps {
  children: React.ReactNode;
  className?: string;
}

interface TableBodyProps {
  children: React.ReactNode;
  className?: string;
}

interface TableRowProps {
  children: React.ReactNode;
  className?: string;
  selected?: boolean;
  clickable?: boolean;
  onClick?: () => void;
}

interface TableCellProps {
  children: React.ReactNode;
  className?: string;
  align?: 'left' | 'center' | 'right';
  width?: string;
  as?: 'td' | 'th';
}

interface TableHeaderCellProps extends Omit<TableCellProps, 'as'> {
  sortable?: boolean;
  sortDirection?: 'asc' | 'desc' | null;
  onSort?: () => void;
}

// Main Table Component
export const Table: React.FC<TableProps> = ({
  children,
  className,
  size = 'md',
  striped = false,
  bordered = true,
  hover = false
}) => {
  const baseStyles = [
    'w-full',
    'bg-surface-primary',
    'rounded-lg',
    'overflow-hidden',
  ];

  const sizeStyles = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg',
  };

  return (
    <div className="overflow-x-auto">
      <table className={clsx(
        baseStyles,
        sizeStyles[size],
        bordered && 'border border-graphite-200',
        className
      )}>
        {children}
      </table>
    </div>
  );
};

// Table Header Component
export const TableHeader: React.FC<TableHeaderProps> = ({
  children,
  className
}) => {
  return (
    <thead className={clsx(
      'bg-graphite-50 border-b border-graphite-200',
      className
    )}>
      {children}
    </thead>
  );
};

// Table Body Component
export const TableBody: React.FC<TableBodyProps> = ({
  children,
  className
}) => {
  return (
    <tbody className={clsx('divide-y divide-graphite-200', className)}>
      {children}
    </tbody>
  );
};

// Table Row Component
export const TableRow: React.FC<TableRowProps> = ({
  children,
  className,
  selected = false,
  clickable = false,
  onClick
}) => {
  const baseStyles = [
    'transition-colors duration-150',
    clickable && 'cursor-pointer',
    clickable && 'hover:bg-graphite-50',
    selected && 'bg-blue-50',
  ];

  return (
    <tr
      className={clsx(baseStyles, className)}
      onClick={onClick}
      role={clickable ? 'button' : undefined}
      tabIndex={clickable ? 0 : undefined}
    >
      {children}
    </tr>
  );
};

// Table Cell Component
export const TableCell: React.FC<TableCellProps> = ({
  children,
  className,
  align = 'left',
  width,
  as: Component = 'td'
}) => {
  const baseStyles = [
    'px-4 py-3',
    'text-graphite-700',
  ];

  const alignStyles = {
    left: 'text-left',
    center: 'text-center',
    right: 'text-right',
  };

  return (
    <Component
      className={clsx(
        baseStyles,
        alignStyles[align],
        className
      )}
      style={{ width }}
    >
      {children}
    </Component>
  );
};

// Table Header Cell Component
export const TableHeaderCell: React.FC<TableHeaderCellProps> = ({
  children,
  className,
  align = 'left',
  width,
  sortable = false,
  sortDirection = null,
  onSort
}) => {
  const baseStyles = [
    'px-4 py-3',
    'text-xs font-medium text-graphite-600 uppercase tracking-wider',
    'bg-graphite-50',
    sortable && 'cursor-pointer select-none hover:bg-graphite-100',
    sortable && 'transition-colors duration-150',
  ];

  const alignStyles = {
    left: 'text-left',
    center: 'text-center', 
    right: 'text-right',
  };

  const getSortIcon = () => {
    if (!sortable) return null;
    
    if (sortDirection === 'asc') {
      return <ChevronUp className="w-4 h-4 text-graphite-500" />;
    }
    
    if (sortDirection === 'desc') {
      return <ChevronDown className="w-4 h-4 text-graphite-500" />;
    }
    
    return <ArrowUpDown className="w-4 h-4 text-graphite-400" />;
  };

  const content = (
    <div className="flex items-center gap-2">
      <span>{children}</span>
      {getSortIcon()}
    </div>
  );

  return (
    <th
      className={clsx(
        baseStyles,
        alignStyles[align],
        className
      )}
      style={{ width }}
      onClick={sortable ? onSort : undefined}
    >
      {sortable ? content : children}
    </th>
  );
};

// Specialized Table Components

// Data Table - for complex data with sorting and filtering
interface DataTableProps<T> {
  data: T[];
  columns: Array<{
    key: keyof T | string;
    label: string;
    sortable?: boolean;
    width?: string;
    align?: 'left' | 'center' | 'right';
    render?: (value: any, row: T, index: number) => React.ReactNode;
  }>;
  loading?: boolean;
  emptyState?: React.ReactNode;
  onRowClick?: (row: T, index: number) => void;
  className?: string;
}

export function DataTable<T extends Record<string, any>>({
  data,
  columns,
  loading = false,
  emptyState,
  onRowClick,
  className
}: DataTableProps<T>) {
  const [sortConfig, setSortConfig] = React.useState<{
    key: string;
    direction: 'asc' | 'desc';
  } | null>(null);

  const sortedData = React.useMemo(() => {
    if (!sortConfig) return data;

    return [...data].sort((a, b) => {
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];

      if (aValue === bValue) return 0;

      const isAscending = sortConfig.direction === 'asc';
      
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return isAscending ? aValue - bValue : bValue - aValue;
      }

      const aStr = String(aValue).toLowerCase();
      const bStr = String(bValue).toLowerCase();
      
      if (isAscending) {
        return aStr < bStr ? -1 : 1;
      } else {
        return aStr > bStr ? -1 : 1;
      }
    });
  }, [data, sortConfig]);

  const handleSort = (key: string) => {
    setSortConfig((current) => {
      if (!current || current.key !== key) {
        return { key, direction: 'asc' };
      }
      
      if (current.direction === 'asc') {
        return { key, direction: 'desc' };
      }
      
      return null;
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-graphite-500">Loading...</div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center p-8">
        {emptyState || (
          <div className="text-center">
            <p className="text-graphite-500">No data available</p>
          </div>
        )}
      </div>
    );
  }

  return (
    <Table className={className}>
      <TableHeader>
        <TableRow>
          {columns.map((column, index) => (
            <TableHeaderCell
              key={`header-${index}`}
              align={column.align}
              width={column.width}
              sortable={column.sortable}
              sortDirection={
                sortConfig?.key === column.key ? sortConfig.direction : null
              }
              onSort={column.sortable ? () => handleSort(String(column.key)) : undefined}
            >
              {column.label}
            </TableHeaderCell>
          ))}
        </TableRow>
      </TableHeader>
      
      <TableBody>
        {sortedData.map((row, rowIndex) => (
          <TableRow
            key={rowIndex}
            clickable={!!onRowClick}
            onClick={() => onRowClick?.(row, rowIndex)}
          >
            {columns.map((column, columnIndex) => {
              const value = row[column.key];
              const rendered = column.render ? column.render(value, row, rowIndex) : value;
              
              return (
                <TableCell
                  key={`cell-${rowIndex}-${columnIndex}`}
                  align={column.align}
                  width={column.width}
                >
                  {rendered}
                </TableCell>
              );
            })}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}