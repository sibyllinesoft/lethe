import { CallPair, CallsFilters, CallsListResponse, DiffResult, PrePostDiff } from '../types'

const API_BASE = '/api'

class ApiClient {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${API_BASE}${endpoint}`
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }))
      throw new Error(errorData.error || `HTTP ${response.status}`)
    }

    return response.json()
  }

  async getCalls(filters?: CallsFilters): Promise<CallsListResponse> {
    const params = new URLSearchParams()
    
    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          params.append(key, String(value))
        }
      })
    }

    const queryString = params.toString()
    const endpoint = `/calls${queryString ? `?${queryString}` : ''}`
    
    return this.request<CallsListResponse>(endpoint)
  }

  async getCall(id: string): Promise<CallPair> {
    return this.request<CallPair>(`/calls/${id}`)
  }

  async getPrePostDiff(id: string): Promise<PrePostDiff> {
    return this.request(`/calls/${id}/pre-post-diff`)
  }

  async compareCallsQuery(callIdA: string, callIdB: string): Promise<DiffResult> {
    const params = new URLSearchParams({ call_id_a: callIdA, call_id_b: callIdB })
    return this.request(`/compare?${params}`)
  }

  async compareCalls(callIdA: string, callIdB: string): Promise<DiffResult> {
    return this.request('/compare', {
      method: 'POST',
      body: JSON.stringify({ call_id_a: callIdA, call_id_b: callIdB }),
    })
  }

  async getRunPairs(runId: string): Promise<{ run_id: string; call_ids: string[] }> {
    return this.request(`/compare/runs/${runId}`)
  }

  async getStats(): Promise<{ total_calls: number; providers: string[]; models: string[]; average_latency_ms: number }> {
    return this.request('/calls/stats')
  }

  async getRuns(): Promise<{ run_ids: string[] }> {
    return this.request<{ run_ids: string[] }>('/calls/runs')
  }

  async getHealth(): Promise<{ status: string; timestamp: string }> {
    return this.request('/health')
  }
}

export const apiClient = new ApiClient()
