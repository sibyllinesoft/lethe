export interface Session {
  id: string
  messageCount: number
  chunkCount: number
  embeddingCount: number
  packCount: number
  createdAt: number
  updatedAt: number
}

export interface Message {
  id: string
  sessionId: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
}

export interface Chunk {
  id: string
  messageId: string
  sessionId: string
  text: string
  kind: 'prose' | 'code' | 'tool_result' | 'user_code'
  startOffset: number
  endOffset: number
  tokens: number
  hasEmbedding: boolean
}

export interface ContextPack {
  id: string
  sessionId: string
  query: string
  summary: string
  keyEntities: string[]
  claims: Array<{ text: string; chunks: string[] }>
  contradictions: Array<{ issue: string; chunks: string[] }>
  citations: Record<string, { messageId: string; span: [number, number] }>
  debug?: {
    hydeQueries: string[]
    candidateCount: number
    rerankTime: number
    totalTime: number
  }
}

export interface QueryResult {
  pack: ContextPack
  debug: {
    planType: string
    hydeQueries: string[]
    candidateCount: number
    rerankTime: number
    totalTime: number
  }
}

export interface Config {
  models: {
    embed: string
    rerank: string
    hyde?: string
    summarize?: string
  }
  retrieval: {
    alpha: number
    beta: number
    gamma_kind_boost: {
      tool_result: number
      user_code: number
    }
  }
  chunking: {
    target_tokens: number
    overlap: number
    split_code_blocks: boolean
    split_sentences: boolean
  }
  rerank: {
    topk_in: number
    topk_out: number
    batch_size: number
  }
  diversify: {
    pack_chunks: number
  }
  plan: {
    explore: { hyde_k: number; granularity: string; beta: number }
    verify: { hyde_k: number; granularity: string; beta: number }
    exploit: { hyde_k: number; granularity: string; beta: number }
  }
}

class ApiClient {
  private baseUrl = '/api'

  async get<T>(path: string): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`)
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    return response.json()
  }

  async post<T>(path: string, data?: any): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: data ? JSON.stringify(data) : undefined,
    })
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    return response.json()
  }

  async put<T>(path: string, data?: any): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: data ? JSON.stringify(data) : undefined,
    })
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    return response.json()
  }

  // Sessions
  async getSessions(): Promise<Session[]> {
    return this.get('/sessions')
  }

  async getSession(sessionId: string) {
    return this.get(`/sessions/${sessionId}`)
  }

  async getMessages(sessionId: string): Promise<Message[]> {
    return this.get(`/sessions/${sessionId}/messages`)
  }

  async getChunks(sessionId: string): Promise<Chunk[]> {
    return this.get(`/sessions/${sessionId}/chunks`)
  }

  // Query
  async query(query: string, sessionId: string): Promise<QueryResult> {
    return this.post('/query', { query, sessionId })
  }

  // Config
  async getConfig(): Promise<Config> {
    return this.get('/config')
  }

  async updateConfig(updates: Partial<Config>): Promise<{ success: boolean }> {
    return this.put('/config', updates)
  }

  // Health
  async health(): Promise<{ status: string; timestamp: string }> {
    return this.get('/health')
  }
}

export const api = new ApiClient()