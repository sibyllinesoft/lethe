import { CallPair, CallsFilters, CallsListResponse } from '../types'

const API_BASE = 'http://localhost:3002/api'

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

  async getPrePostDiff(id: string): Promise<any> {
    return this.request<any>(`/calls/${id}/pre-post-diff`)
  }

  async compareCallsQuery(callIdA: string, callIdB: string): Promise<any> {
    const params = new URLSearchParams({ call_id_a: callIdA, call_id_b: callIdB })
    return this.request<any>(`/compare?${params}`)
  }

  async compareCalls(callIdA: string, callIdB: string): Promise<any> {
    return this.request<any>('/compare', {
      method: 'POST',
      body: JSON.stringify({ call_id_a: callIdA, call_id_b: callIdB }),
    })
  }

  async getRunPairs(runId: string): Promise<any> {
    return this.request<any>(`/compare/runs/${runId}`)
  }

  async getStats(): Promise<any> {
    return this.request<any>('/calls/stats')
  }

  async getRuns(): Promise<{ run_ids: string[] }> {
    return this.request<{ run_ids: string[] }>('/calls/runs')
  }

  async getHealth(): Promise<{ status: string; timestamp: string }> {
    return this.request<{ status: string; timestamp: string }>('/health')
  }
}

export const apiClient = new ApiClient()