import React, { useState, useEffect } from 'react'
import { Save, RotateCcw, Settings, AlertCircle } from 'lucide-react'
import { api, type Config } from '../lib/api'

export function ConfigEditor() {
  const [config, setConfig] = useState<Config | null>(null)
  const [originalConfig, setOriginalConfig] = useState<Config | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  useEffect(() => {
    loadConfig()
  }, [])

  const loadConfig = async () => {
    try {
      const data = await api.getConfig()
      setConfig(data)
      setOriginalConfig(JSON.parse(JSON.stringify(data))) // Deep clone
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load config')
    }
  }

  const handleSave = async () => {
    if (!config) return

    try {
      setSaving(true)
      setError(null)
      await api.updateConfig(config)
      setOriginalConfig(JSON.parse(JSON.stringify(config)))
      setSuccess(true)
      setTimeout(() => setSuccess(false), 3000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save config')
    } finally {
      setSaving(false)
    }
  }

  const handleReset = () => {
    if (originalConfig) {
      setConfig(JSON.parse(JSON.stringify(originalConfig)))
      setError(null)
      setSuccess(false)
    }
  }

  const hasChanges = config && originalConfig && 
    JSON.stringify(config) !== JSON.stringify(originalConfig)

  if (!config) {
    return (
      <div className="p-8">
        <div className="animate-pulse">
          <div className="h-8 bg-slate-700 rounded w-1/4 mb-6"></div>
          <div className="space-y-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="bg-slate-800 p-6 rounded-lg">
                <div className="h-6 bg-slate-700 rounded w-1/3 mb-4"></div>
                <div className="h-4 bg-slate-700 rounded w-full mb-2"></div>
                <div className="h-4 bg-slate-700 rounded w-2/3"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
          <Settings className="w-8 h-8" />
          Configuration
        </h1>
        <p className="text-slate-400">
          Adjust ctx-run parameters for optimal search performance
        </p>
      </div>

      {/* Actions */}
      <div className="bg-slate-800 rounded-lg p-6 mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {success && (
              <div className="flex items-center gap-2 text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm">Configuration saved successfully</span>
              </div>
            )}
            {error && (
              <div className="flex items-center gap-2 text-red-400">
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm">{error}</span>
              </div>
            )}
          </div>
          
          <div className="flex items-center gap-3">
            <button
              onClick={handleReset}
              disabled={!hasChanges}
              className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 disabled:cursor-not-allowed text-slate-300 hover:text-white rounded-lg text-sm transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              Reset
            </button>
            
            <button
              onClick={handleSave}
              disabled={saving || !hasChanges}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg text-sm transition-colors"
            >
              {saving ? (
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
              ) : (
                <Save className="w-4 h-4" />
              )}
              Save Changes
            </button>
          </div>
        </div>
      </div>

      <div className="space-y-6">
        {/* Models */}
        <ConfigSection
          title="Models"
          description="AI model configurations"
          config={config}
          setConfig={setConfig}
          section="models"
        />

        {/* Retrieval */}
        <ConfigSection
          title="Retrieval"
          description="Search and ranking parameters"
          config={config}
          setConfig={setConfig}
          section="retrieval"
        />

        {/* Chunking */}
        <ConfigSection
          title="Chunking"
          description="Text chunking configuration"
          config={config}
          setConfig={setConfig}
          section="chunking"
        />

        {/* Reranking */}
        <ConfigSection
          title="Reranking"
          description="Cross-encoder reranking settings"
          config={config}
          setConfig={setConfig}
          section="rerank"
        />

        {/* Diversify */}
        <ConfigSection
          title="Diversify"
          description="Context pack diversification"
          config={config}
          setConfig={setConfig}
          section="diversify"
        />

        {/* Planning */}
        <div className="bg-slate-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-white mb-2">Planning</h2>
          <p className="text-slate-400 mb-4">Query planning strategies</p>
          
          <div className="space-y-4">
            {Object.entries(config.plan).map(([planType, planConfig]) => (
              <div key={planType} className="bg-slate-700 rounded-lg p-4">
                <h3 className="text-md font-medium text-white mb-3 capitalize">
                  {planType} Plan
                </h3>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">
                      HyDE K
                    </label>
                    <input
                      type="number"
                      value={planConfig.hyde_k}
                      onChange={(e) => {
                        const newConfig = { ...config }
                        newConfig.plan[planType as keyof typeof config.plan].hyde_k = parseInt(e.target.value)
                        setConfig(newConfig)
                      }}
                      className="w-full bg-slate-600 border border-slate-500 rounded px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">
                      Beta
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      value={planConfig.beta}
                      onChange={(e) => {
                        const newConfig = { ...config }
                        newConfig.plan[planType as keyof typeof config.plan].beta = parseFloat(e.target.value)
                        setConfig(newConfig)
                      }}
                      className="w-full bg-slate-600 border border-slate-500 rounded px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-1">
                      Granularity
                    </label>
                    <input
                      type="text"
                      value={planConfig.granularity}
                      onChange={(e) => {
                        const newConfig = { ...config }
                        newConfig.plan[planType as keyof typeof config.plan].granularity = e.target.value
                        setConfig(newConfig)
                      }}
                      className="w-full bg-slate-600 border border-slate-500 rounded px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function ConfigSection({ 
  title, 
  description, 
  config, 
  setConfig, 
  section 
}: {
  title: string
  description: string
  config: Config
  setConfig: (config: Config) => void
  section: keyof Config
}) {
  const sectionData = config[section]

  const updateValue = (key: string, value: any) => {
    const newConfig = { ...config }
    ;(newConfig[section] as any)[key] = value
    setConfig(newConfig)
  }

  const updateNestedValue = (parentKey: string, key: string, value: any) => {
    const newConfig = { ...config }
    ;(newConfig[section] as any)[parentKey][key] = value
    setConfig(newConfig)
  }

  return (
    <div className="bg-slate-800 rounded-lg p-6">
      <h2 className="text-lg font-semibold text-white mb-2">{title}</h2>
      <p className="text-slate-400 mb-4">{description}</p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {Object.entries(sectionData).map(([key, value]) => {
          if (typeof value === 'object' && value !== null) {
            return (
              <div key={key} className="md:col-span-2 lg:col-span-3">
                <label className="block text-sm font-medium text-slate-300 mb-2 capitalize">
                  {key.replace(/_/g, ' ')}
                </label>
                <div className="bg-slate-700 rounded-lg p-3">
                  <div className="grid grid-cols-2 gap-3">
                    {Object.entries(value).map(([nestedKey, nestedValue]) => (
                      <div key={nestedKey}>
                        <label className="block text-xs font-medium text-slate-400 mb-1 capitalize">
                          {nestedKey.replace(/_/g, ' ')}
                        </label>
                        <input
                          type={typeof nestedValue === 'number' ? 'number' : 'text'}
                          step={typeof nestedValue === 'number' && nestedValue % 1 !== 0 ? '0.1' : '1'}
                          value={nestedValue as string | number}
                          onChange={(e) => {
                            const newValue = typeof nestedValue === 'number' 
                              ? (e.target.type === 'number' ? parseFloat(e.target.value) : parseInt(e.target.value))
                              : e.target.value
                            updateNestedValue(key, nestedKey, newValue)
                          }}
                          className="w-full bg-slate-600 border border-slate-500 rounded px-2 py-1 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )
          }

          if (typeof value === 'boolean') {
            return (
              <div key={key}>
                <label className="block text-sm font-medium text-slate-300 mb-2 capitalize">
                  {key.replace(/_/g, ' ')}
                </label>
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={value}
                    onChange={(e) => updateValue(key, e.target.checked)}
                    className="sr-only"
                  />
                  <div className={`relative w-10 h-6 rounded-full transition-colors ${
                    value ? 'bg-blue-600' : 'bg-slate-600'
                  }`}>
                    <div className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      value ? 'translate-x-4' : 'translate-x-0'
                    }`} />
                  </div>
                  <span className="ml-3 text-sm text-slate-300">
                    {value ? 'Enabled' : 'Disabled'}
                  </span>
                </label>
              </div>
            )
          }

          return (
            <div key={key}>
              <label className="block text-sm font-medium text-slate-300 mb-2 capitalize">
                {key.replace(/_/g, ' ')}
              </label>
              <input
                type={typeof value === 'number' ? 'number' : 'text'}
                step={typeof value === 'number' && value % 1 !== 0 ? '0.1' : '1'}
                value={value as string | number}
                onChange={(e) => {
                  const newValue = typeof value === 'number' 
                    ? (e.target.type === 'number' ? parseFloat(e.target.value) : parseInt(e.target.value))
                    : e.target.value
                  updateValue(key, newValue)
                }}
                className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )
        })}
      </div>
    </div>
  )
}