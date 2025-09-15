import { create } from 'zustand'

interface UIState {
  selectedCallIds: string[]
  compareMode: boolean
  currentView: 'pre' | 'post' | 'diff'
  sidebarCollapsed: boolean
  
  // Actions
  setSelectedCallIds: (ids: string[]) => void
  toggleCompareMode: () => void
  setCurrentView: (view: 'pre' | 'post' | 'diff') => void
  toggleSidebar: () => void
  clearSelection: () => void
}

export const useUIStore = create<UIState>((set) => ({
  selectedCallIds: [],
  compareMode: false,
  currentView: 'diff',
  sidebarCollapsed: false,
  
  setSelectedCallIds: (ids) => 
    set({ selectedCallIds: ids, compareMode: ids.length === 2 }),
  
  toggleCompareMode: () => 
    set((state) => ({ compareMode: !state.compareMode })),
  
  setCurrentView: (view) => 
    set({ currentView: view }),
  
  toggleSidebar: () => 
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  
  clearSelection: () => 
    set({ selectedCallIds: [], compareMode: false }),
}))