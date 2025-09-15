import fs from 'fs/promises'
import { readFileSync } from 'fs'
import path from 'path'
import { ProxyLogEntry, ProxyRequestTransform, ProxyResponse, CallPair } from '../types.js'

export class LogParser {
  private logDir: string

  constructor(logDir = './logs') {
    this.logDir = logDir
  }

  async parseLogFile(filePath: string): Promise<ProxyLogEntry[]> {
    try {
      const content = await fs.readFile(filePath, 'utf-8')
      const entries: ProxyLogEntry[] = []
      
      // Handle both newline-separated and concatenated JSON
      let lines: string[] = []
      if (content.includes('\n')) {
        lines = content.split('\n')
      } else {
        // Try to split on }{ pattern for concatenated JSON
        lines = content.split(/(?<=\})\s*(?=\{)/)
      }
      
      lines.forEach((line, index) => {
        if (line.trim()) {
          try {
            // Clean up the line
            let cleanLine = line.trim()
            if (!cleanLine.startsWith('{')) {
              // Find the first { character
              const firstBrace = cleanLine.indexOf('{')
              if (firstBrace > -1) {
                cleanLine = cleanLine.substring(firstBrace)
              }
            }
            if (!cleanLine.endsWith('}')) {
              // Find the last } character  
              const lastBrace = cleanLine.lastIndexOf('}')
              if (lastBrace > -1) {
                cleanLine = cleanLine.substring(0, lastBrace + 1)
              }
            }
            
            const entry = JSON.parse(cleanLine) as ProxyLogEntry
            entry._line_number = index + 1
            entries.push(entry)
          } catch (error) {
            console.warn(`Failed to parse line ${index + 1}: ${error}`)
          }
        }
      })
      
      return entries
    } catch (error) {
      console.error(`Failed to read log file ${filePath}:`, error)
      return []
    }
  }

  processLogsToCallPairs(logs: ProxyLogEntry[]): CallPair[] {
    const requestsMap = new Map<string, ProxyRequestTransform>()
    const responsesMap = new Map<string, ProxyResponse>()
    
    // Group logs by request_id
    logs.forEach(log => {
      if (log.event === 'proxy_request_transform') {
        requestsMap.set(log.request_id, log as ProxyRequestTransform)
      } else if (log.event === 'proxy_response') {
        responsesMap.set(log.request_id, log as ProxyResponse)
      }
    })
    
    // Create CallPair objects
    const callPairs: CallPair[] = []
    
    requestsMap.forEach((request, requestId) => {
      const response = responsesMap.get(requestId)
      
      // Extract pre/post context from messages
      const preMessages = request.pre_transform.payload.messages || []
      const postMessages = request.post_transform.payload.messages || []
      
      const preContext = preMessages.filter(m => m.role === 'system')
      const postContext = postMessages.filter(m => m.role === 'system')
      
      // Get user prompt (last user message)
      const userMessages = preMessages.filter(m => m.role === 'user')
      const prompt = userMessages.length > 0 ? userMessages[userMessages.length - 1].content : ''
      
      // Get completion from response
      const completion = response?.response_preview || (response ? 'Response content not captured in logs' : undefined)
      
      const callPair: CallPair = {
        id: requestId,
        timestamp: request.timestamp,
        run_id: request.benchmark_metadata.run_id,
        query_id: request.benchmark_metadata.query_id,
        provider: request.provider,
        model: request.pre_transform.payload.model,
        benchmark_type: request.benchmark_metadata.benchmark_type,
        dataset: request.benchmark_metadata.dataset,
        request,
        response,
        pre_context: preContext,
        post_context: postContext,
        prompt,
        completion,
        input_tokens: request.pre_transform.token_estimate,
        output_tokens: response?.performance.response_tokens || 0,
        latency_ms: response?.performance.total_request_duration_ms || 0,
        status: response ? (response.status_code === 200 ? 'success' : 'error') : 'pending',
        temperature: request.pre_transform.payload.temperature,
        max_tokens: request.pre_transform.payload.max_tokens,
        transform_changes: request.transform.changes
      }
      
      callPairs.push(callPair)
    })
    
    // Sort by timestamp
    return callPairs.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
  }

  async findLogFiles(): Promise<string[]> {
    try {
      const files = await fs.readdir(this.logDir)
      return files
        .filter(file => file.endsWith('.jsonl'))
        .map(file => path.join(this.logDir, file))
    } catch (error) {
      console.error('Failed to read log directory:', error)
      return []
    }
  }

  // For demo purposes, use the most recent logs
  async loadSampleLogs(): Promise<CallPair[]> {
    try {
      // Look for live benchmark logs first
      const files = await fs.readdir(process.cwd())
      const logFiles = files
        .filter(file => file.startsWith('ollama-live-') && file.endsWith('.jsonl'))
        .sort()
        .reverse() // most recent first
      
      let logPath: string
      if (logFiles.length > 0) {
        logPath = path.join(process.cwd(), logFiles[0])
        console.log(`Loading live benchmark logs: ${logFiles[0]}`)
      } else {
        // Fallback to original demo logs
        logPath = path.join(process.cwd(), 'ollama-benchmark-ollama-demo-20250915T154208.jsonl')
        console.log('Loading demo logs: ollama-benchmark-ollama-demo-20250915T154208.jsonl')
      }
      
      const logs = await this.parseLogFile(logPath)
      return this.processLogsToCallPairs(logs)
    } catch (error) {
      console.error('Failed to load logs:', error)
      return []
    }
  }
}