import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '../lib/api'
import { CallsFilters } from '../types'

// Query keys
export const queryKeys = {
  calls: (filters?: CallsFilters) => ['calls', filters] as const,
  call: (id: string) => ['call', id] as const,
  prePostDiff: (id: string) => ['prePostDiff', id] as const,
  comparison: (idA: string, idB: string) => ['comparison', idA, idB] as const,
  runPairs: (runId: string) => ['runPairs', runId] as const,
  stats: () => ['stats'] as const,
  runs: () => ['runs'] as const,
  health: () => ['health'] as const,
}

// Hooks for calls
export const useCalls = (filters?: CallsFilters) => {
  return useQuery({
    queryKey: queryKeys.calls(filters),
    queryFn: () => apiClient.getCalls(filters),
    keepPreviousData: true,
  })
}

export const useCall = (id: string) => {
  return useQuery({
    queryKey: queryKeys.call(id),
    queryFn: () => apiClient.getCall(id),
    enabled: !!id,
  })
}

export const usePrePostDiff = (id: string) => {
  return useQuery({
    queryKey: queryKeys.prePostDiff(id),
    queryFn: () => apiClient.getPrePostDiff(id),
    enabled: !!id,
  })
}

// Hooks for comparison
export const useCallComparison = (callIdA?: string, callIdB?: string) => {
  return useQuery({
    queryKey: queryKeys.comparison(callIdA || '', callIdB || ''),
    queryFn: () => apiClient.compareCalls(callIdA!, callIdB!),
    enabled: !!(callIdA && callIdB),
  })
}

export const useRunPairs = (runId: string) => {
  return useQuery({
    queryKey: queryKeys.runPairs(runId),
    queryFn: () => apiClient.getRunPairs(runId),
    enabled: !!runId,
  })
}

// Hooks for metadata
export const useStats = () => {
  return useQuery({
    queryKey: queryKeys.stats(),
    queryFn: () => apiClient.getStats(),
    staleTime: 1000 * 60 * 5, // 5 minutes
  })
}

export const useRuns = () => {
  return useQuery({
    queryKey: queryKeys.runs(),
    queryFn: () => apiClient.getRuns(),
    staleTime: 1000 * 60 * 5, // 5 minutes
  })
}

export const useHealth = () => {
  return useQuery({
    queryKey: queryKeys.health(),
    queryFn: () => apiClient.getHealth(),
    refetchInterval: 1000 * 60, // 1 minute
    retry: 3,
  })
}

// Mutations
export const useRefreshCache = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async () => {
      // Invalidate all relevant queries to force refetch
      await queryClient.invalidateQueries({ queryKey: ['calls'] })
      await queryClient.invalidateQueries({ queryKey: ['stats'] })
      await queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
  })
}