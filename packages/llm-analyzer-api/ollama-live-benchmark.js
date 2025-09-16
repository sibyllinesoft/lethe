#!/usr/bin/env node

/**
 * Live Ollama Benchmark Harness
 * 
 * Actually calls Ollama API and generates proxy log entries
 * compatible with the LLM Analyzer.
 */

import fs from 'fs/promises'
import path from 'path'

// Simple benchmark prompts
const benchmarkPrompts = [
  {
    id: "qa-001",
    prompt: "What is the capital of Japan?",
    expected: "Tokyo"
  },
  {
    id: "qa-002", 
    prompt: "Explain photosynthesis in one sentence.",
    expected: "Plants convert sunlight into energy"
  },
  {
    id: "reasoning-001",
    prompt: "If a train travels 60 mph for 2 hours, how far does it go?",
    expected: "120 miles"
  }
]

// Call Ollama API
async function callOllama(prompt, model = 'gemma2:9b') {
  const startTime = Date.now()
  
  try {
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: model,
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9,
          max_tokens: 200
        }
      })
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }

    const data = await response.json()
    const endTime = Date.now()
    
    return {
      response: data.response || 'No response generated',
      latency: endTime - startTime,
      success: true,
      model: data.model || model,
      eval_count: data.eval_count || 0,
      prompt_eval_count: data.prompt_eval_count || 0,
      total_duration: data.total_duration || (endTime - startTime) * 1000000, // convert to nanoseconds
      load_duration: data.load_duration || 0,
      prompt_eval_duration: data.prompt_eval_duration || 0,
      eval_duration: data.eval_duration || 0
    }
  } catch (error) {
    const endTime = Date.now()
    console.error(`Ollama API error for prompt "${prompt}":`, error.message)
    return {
      response: `Error: ${error.message}`,
      latency: endTime - startTime,
      success: false,
      model: model,
      eval_count: 0,
      prompt_eval_count: 0,
      total_duration: 0,
      load_duration: 0,
      prompt_eval_duration: 0,
      eval_duration: 0
    }
  }
}

// Generate proxy log entries
function generateProxyLog(promptData, ollamaResult, runId, index) {
  const requestId = `ollama-${runId}-${String(index).padStart(3, '0')}`
  const baseTime = new Date()
  const responseTime = new Date(baseTime.getTime() + ollamaResult.latency)
  
  // Pre-transform payload (before adding system prompt)
  const preTransformPayload = {
    model: ollamaResult.model,
    messages: [
      { role: "user", content: promptData.prompt }
    ],
    temperature: 0.7,
    max_tokens: 200
  }
  
  // Post-transform payload (after adding system prompt)
  const postTransformPayload = {
    model: ollamaResult.model,
    messages: [
      { role: "system", content: "You are a helpful assistant that provides accurate, concise answers." },
      { role: "user", content: promptData.prompt }
    ],
    temperature: 0.7,
    max_tokens: 200
  }
  
  // Calculate token estimates (rough approximation: 1 token ≈ 4 characters)
  const preTokens = Math.ceil(promptData.prompt.length / 4)
  const postTokens = Math.ceil((promptData.prompt.length + 100) / 4) // +100 for system prompt
  const responseTokens = Math.ceil(ollamaResult.response.length / 4)
  
  const requestLog = {
    timestamp: baseTime.toISOString(),
    level: "INFO",
    event: "proxy_request_transform", 
    request_id: requestId,
    benchmark_metadata: {
      run_id: runId,
      query_id: promptData.id,
      provider: "ollama",
      benchmark_type: "live_benchmark",
      dataset: "ollama_test"
    },
    provider: "ollama",
    path: "/api/generate",
    method: "POST",
    transform: {
      enabled: true,
      duration_ms: 2,
      changes: ["system_prompt_added", "benchmark_metadata_added"],
      size_change_percent: ((postTokens - preTokens) / preTokens * 100).toFixed(1)
    },
    pre_transform: {
      size_bytes: promptData.prompt.length * 4,
      token_estimate: preTokens,
      payload: preTransformPayload
    },
    post_transform: {
      size_bytes: (promptData.prompt.length + 100) * 4,
      token_estimate: postTokens,
      payload: postTransformPayload
    },
    performance: {
      transform_duration_ms: 2,
      total_request_duration_ms: null,
      pre_transform_size_bytes: promptData.prompt.length * 4,
      post_transform_size_bytes: (promptData.prompt.length + 100) * 4,
      size_change_percent: ((postTokens - preTokens) / preTokens * 100).toFixed(1)
    }
  }

  const responseLog = {
    timestamp: responseTime.toISOString(), 
    level: "INFO",
    event: "proxy_response",
    request_id: requestId,
    provider: "ollama",
    status_code: ollamaResult.success ? 200 : 500,
    response_size_bytes: ollamaResult.response.length * 4,
    performance: {
      transform_duration_ms: 2,
      total_request_duration_ms: ollamaResult.latency,
      response_tokens: responseTokens,
      response_time_ms: ollamaResult.latency,
      ollama_metrics: {
        eval_count: ollamaResult.eval_count,
        prompt_eval_count: ollamaResult.prompt_eval_count,
        total_duration_ns: ollamaResult.total_duration,
        load_duration_ns: ollamaResult.load_duration,
        prompt_eval_duration_ns: ollamaResult.prompt_eval_duration,
        eval_duration_ns: ollamaResult.eval_duration
      }
    },
    response_preview: ollamaResult.response.substring(0, 100) + (ollamaResult.response.length > 100 ? '...' : ''),
    response_full: ollamaResult.response
  }

  return [requestLog, responseLog]
}

async function runLiveBenchmark() {
  console.log('🚀 Starting Live Ollama Benchmark...')
  
  const runId = `live-${new Date().toISOString().slice(0, 19).replace(/[:-]/g, '')}`
  const logEntries = []
  
  console.log(`📊 Run ID: ${runId}`)
  console.log(`📝 Processing ${benchmarkPrompts.length} prompts through Ollama...\n`)
  
  for (let i = 0; i < benchmarkPrompts.length; i++) {
    const prompt = benchmarkPrompts[i]
    console.log(`${i + 1}. ${prompt.prompt}`)
    
    // Call Ollama API
    const result = await callOllama(prompt.prompt)
    
    if (result.success) {
      console.log(`   ✅ ${result.response.substring(0, 80)}${result.response.length > 80 ? '...' : ''}`)
      console.log(`   ⏱️  ${result.latency}ms (${result.eval_count} tokens)`)
    } else {
      console.log(`   ❌ ${result.response}`)
      console.log(`   ⏱️  ${result.latency}ms (failed)`)
    }
    
    // Generate proxy log entries
    const [requestLog, responseLog] = generateProxyLog(prompt, result, runId, i + 1)
    logEntries.push(requestLog, responseLog)
    
    // Small delay between requests
    await new Promise(resolve => setTimeout(resolve, 500))
    console.log()
  }
  
  // Write to log file
  const logFile = `ollama-live-${runId}.jsonl`
  const logContent = logEntries.map(entry => JSON.stringify(entry)).join('\n')
  
  await fs.writeFile(logFile, logContent)
  
  console.log(`✅ Live benchmark complete!`)
  console.log(`📄 Log file: ${logFile}`)
  console.log(`📊 Generated ${logEntries.length} log entries (${benchmarkPrompts.length} request/response pairs)`)
  
  // Calculate summary stats
  const responses = logEntries.filter(entry => entry.event === 'proxy_response')
  const successfulResponses = responses.filter(r => r.status_code === 200)
  const avgLatency = successfulResponses.reduce((sum, r) => sum + r.performance.total_request_duration_ms, 0) / successfulResponses.length
  const totalTokens = successfulResponses.reduce((sum, r) => sum + r.performance.response_tokens, 0)
  
  console.log(`\n📈 Summary:`)
  console.log(`   Run ID: ${runId}`)
  console.log(`   Provider: ollama`) 
  console.log(`   Model: gemma2:9b`)
  console.log(`   Queries: ${benchmarkPrompts.length}`)
  console.log(`   Success Rate: ${successfulResponses.length}/${responses.length} (${(successfulResponses.length/responses.length*100).toFixed(1)}%)`)
  console.log(`   Avg Response Time: ${avgLatency.toFixed(0)}ms`)
  console.log(`   Total Tokens Generated: ${totalTokens}`)
  
  console.log(`\n🔄 To see results in LLM Analyzer:`)
  console.log(`   1. The API will automatically pick up this file`)
  console.log(`   2. Refresh the React app to see the new data`)
  console.log(`   3. Look for run ID: ${runId}`)
  
  return { runId, logFile, entries: logEntries.length, successRate: successfulResponses.length/responses.length }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runLiveBenchmark().catch(console.error)
}

export { runLiveBenchmark }