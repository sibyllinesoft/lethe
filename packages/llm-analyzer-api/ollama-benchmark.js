#!/usr/bin/env node

/**
 * Simple Ollama Benchmark Harness
 * 
 * Demonstrates routing benchmark prompts through Ollama and emitting
 * proxy log entries compatible with the LLM Analyzer.
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

// Generate proxy log entries
function generateProxyLog(prompt, response, runId, index) {
  const requestId = `ollama-${runId}-${String(index).padStart(3, '0')}`
  const baseTime = new Date()
  const responseTime = new Date(baseTime.getTime() + 2000 + Math.random() * 3000) // 2-5 second response
  
  const requestLog = {
    timestamp: baseTime.toISOString(),
    level: "INFO",
    event: "proxy_request_transform", 
    request_id: requestId,
    benchmark_metadata: {
      run_id: runId,
      query_id: prompt.id,
      provider: "ollama",
      benchmark_type: "qa_reasoning",
      dataset: "demo_benchmark"
    },
    provider: "ollama",
    path: "/v1/chat/completions",
    method: "POST",
    transform: {
      enabled: true,
      duration_ms: Math.floor(Math.random() * 5) + 1,
      changes: ["system_prelude_added", "benchmark_metadata_added"],
      size_change_percent: 15.0
    },
    pre_transform: {
      size_bytes: prompt.prompt.length * 4,
      token_estimate: Math.floor(prompt.prompt.length / 4),
      payload: {
        model: "llama3",
        messages: [
          { role: "user", content: prompt.prompt }
        ],
        temperature: 0.7,
        max_tokens: 500
      }
    },
    post_transform: {
      size_bytes: Math.floor(prompt.prompt.length * 4 * 1.15),
      token_estimate: Math.floor(prompt.prompt.length / 4 * 1.15),
      payload: {
        model: "llama3", 
        messages: [
          { role: "system", content: "You are a helpful assistant that provides accurate, concise answers." },
          { role: "user", content: prompt.prompt }
        ],
        temperature: 0.7,
        max_tokens: 500
      }
    },
    performance: {
      transform_duration_ms: Math.floor(Math.random() * 5) + 1,
      total_request_duration_ms: null,
      pre_transform_size_bytes: prompt.prompt.length * 4,
      post_transform_size_bytes: Math.floor(prompt.prompt.length * 4 * 1.15),
      size_change_percent: 15.0
    }
  }

  const responseLog = {
    timestamp: responseTime.toISOString(), 
    level: "INFO",
    event: "proxy_response",
    request_id: requestId,
    provider: "ollama",
    status_code: 200,
    response_size_bytes: response.length * 4,
    performance: {
      transform_duration_ms: requestLog.transform.duration_ms,
      total_request_duration_ms: responseTime.getTime() - baseTime.getTime(),
      response_tokens: Math.floor(response.length / 4),
      response_time_ms: responseTime.getTime() - baseTime.getTime() - 2
    }
  }

  return [requestLog, responseLog]
}

// Simulate Ollama responses (in real implementation, this would call Ollama API)
function simulateOllamaResponse(prompt) {
  const responses = {
    "qa-001": "Tokyo is the capital of Japan.",
    "qa-002": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
    "reasoning-001": "The train travels 120 miles (60 mph × 2 hours = 120 miles)."
  }
  
  return responses[prompt.id] || "I don't know the answer to that question."
}

async function runBenchmark() {
  console.log('🚀 Starting Ollama Benchmark Demo...')
  
  const runId = `ollama-demo-${new Date().toISOString().slice(0, 19).replace(/[:-]/g, '')}`
  const logEntries = []
  
  console.log(`📊 Run ID: ${runId}`)
  console.log(`📝 Processing ${benchmarkPrompts.length} prompts...\n`)
  
  for (let i = 0; i < benchmarkPrompts.length; i++) {
    const prompt = benchmarkPrompts[i]
    console.log(`${i + 1}. ${prompt.prompt}`)
    
    // Simulate routing through Ollama
    const response = simulateOllamaResponse(prompt)
    console.log(`   → ${response}`)
    
    // Generate proxy log entries
    const [requestLog, responseLog] = generateProxyLog(prompt, response, runId, i + 1)
    logEntries.push(requestLog, responseLog)
    
    // Small delay to simulate real timing
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  
  // Write to log file
  const logFile = `ollama-benchmark-${runId}.jsonl`
  const logContent = logEntries.map(entry => JSON.stringify(entry)).join('\n')
  
  await fs.writeFile(logFile, logContent)
  
  console.log(`\n✅ Benchmark complete!`)
  console.log(`📄 Log file: ${logFile}`)
  console.log(`📊 Generated ${logEntries.length} log entries (${benchmarkPrompts.length} request/response pairs)`)
  
  // Output summary for the LLM Analyzer
  console.log(`\n📈 Summary:`)
  console.log(`   Run ID: ${runId}`)
  console.log(`   Provider: ollama`) 
  console.log(`   Model: llama3`)
  console.log(`   Queries: ${benchmarkPrompts.length}`)
  console.log(`   Avg Response Time: ~3.5s`)
  console.log(`   Transformations: system_prelude_added, benchmark_metadata_added`)
  
  console.log(`\n🔧 To analyze in LLM Analyzer:`)
  console.log(`   1. Copy ${logFile} to the API directory`)
  console.log(`   2. Update log-parser.ts to point to this file`)
  console.log(`   3. Refresh the React app to see the new data`)
  
  return { runId, logFile, entries: logEntries.length }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runBenchmark().catch(console.error)
}