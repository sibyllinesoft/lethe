import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { 
  MessageSquare, 
  Search, 
  Settings, 
  BookOpen,
  Archive,
  Bug,
  Activity
} from 'lucide-react'
import { cn } from '../lib/utils'

const navigation = [
  { name: 'Sessions', href: '/', icon: MessageSquare },
  { name: 'Query', href: '/query', icon: Search },
  { name: 'Debug', href: '/debug', icon: Bug },
  { name: 'Checkpoints', href: '/checkpoints', icon: Archive },
  { name: 'Config', href: '/config', icon: Settings },
]

export function Sidebar() {
  const location = useLocation()

  return (
    <div className="w-64 bg-slate-800 border-r border-slate-700 flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-slate-700">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-white">ctx-run</h1>
            <p className="text-xs text-slate-400">Development Server</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href
            return (
              <li key={item.name}>
                <Link
                  to={item.href}
                  className={cn(
                    'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-300 hover:text-white hover:bg-slate-700'
                  )}
                >
                  <item.icon className="w-5 h-5" />
                  {item.name}
                </Link>
              </li>
            )
          })}
        </ul>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-slate-700">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <div className="w-2 h-2 bg-green-400 rounded-full"></div>
          <span>Connected</span>
        </div>
      </div>
    </div>
  )
}