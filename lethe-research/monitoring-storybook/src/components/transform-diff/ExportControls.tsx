import React, { useState } from 'react';
import { Download, FileImage, FileText, Database } from 'lucide-react';
import clsx from 'clsx';

interface ExportControlsProps {
  onExport: (format: 'png' | 'svg' | 'json') => void;
  className?: string;
}

/**
 * ExportControls - Export functionality for transform diff visualizations
 * 
 * Features:
 * - PNG export for high-quality images
 * - SVG export for vector graphics
 * - JSON export for raw data
 * - Dropdown menu interface
 * - Loading states during export
 */
export const ExportControls: React.FC<ExportControlsProps> = ({
  onExport,
  className
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isExporting, setIsExporting] = useState<'png' | 'svg' | 'json' | null>(null);

  const handleExport = async (format: 'png' | 'svg' | 'json') => {
    setIsExporting(format);
    setIsOpen(false);
    
    try {
      await onExport(format);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(null);
    }
  };

  const exportOptions = [
    {
      format: 'png' as const,
      label: 'Export as PNG',
      description: 'High-quality raster image',
      icon: FileImage,
      color: 'text-blue-600 dark:text-blue-400'
    },
    {
      format: 'svg' as const,
      label: 'Export as SVG',
      description: 'Scalable vector graphics',
      icon: FileText,
      color: 'text-green-600 dark:text-green-400'
    },
    {
      format: 'json' as const,
      label: 'Export Data',
      description: 'Raw JSON data',
      icon: Database,
      color: 'text-purple-600 dark:text-purple-400'
    }
  ];

  return (
    <div className={clsx('relative', className)}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={isExporting !== null}
        className={clsx(
          'flex items-center px-3 py-2 text-sm font-medium rounded-md border transition-colors',
          isExporting
            ? 'bg-gray-100 dark:bg-gray-800 text-gray-400 cursor-not-allowed'
            : 'bg-white dark:bg-gray-900 text-gray-700 dark:text-gray-300 border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800'
        )}
      >
        <Download 
          size={16} 
          className={clsx(
            'mr-2',
            isExporting ? 'animate-pulse' : ''
          )} 
        />
        {isExporting ? `Exporting ${isExporting.toUpperCase()}...` : 'Export'}
      </button>

      {/* Dropdown menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-56 bg-white dark:bg-gray-800 rounded-md shadow-lg border border-gray-200 dark:border-gray-700 z-50">
          <div className="py-1">
            {exportOptions.map((option) => (
              <button
                key={option.format}
                onClick={() => handleExport(option.format)}
                className="flex items-start w-full px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
              >
                <option.icon 
                  size={16} 
                  className={clsx('mt-0.5 mr-3 flex-shrink-0', option.color)} 
                />
                <div>
                  <div className="font-medium text-gray-900 dark:text-white text-sm">
                    {option.label}
                  </div>
                  <div className="text-gray-500 dark:text-gray-400 text-xs">
                    {option.description}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Overlay to close dropdown */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  );
};