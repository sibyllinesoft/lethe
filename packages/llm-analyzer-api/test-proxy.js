#!/usr/bin/env node

import fetch from 'node-fetch'
import fs from 'fs'

// Read one InfiniteBench entry
const line = fs.readFileSync('infinitebench-code_debug.jsonl', 'utf-8').split('\n')[0]
const entry = JSON.parse(line)

console.log('Testing proxy with real InfiniteBench data...')
console.log(`Context length: ${entry.context.length} chars`)
console.log(`Question: ${entry.input}`)

const payload = {
  model: "gemma2:9b",
  messages: [
    { role: "user", content: `Context:\n${entry.context}\n\nQuestion: ${entry.input}` }
  ],
  temperature: 0.7,
  max_tokens: 200
}

try {
  const response = await fetch('http://localhost:3002/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  
  if (!response.ok) {
    console.error('Response not ok:', response.status, response.statusText)
    const text = await response.text()
    console.error('Response body:', text)
  } else {
    const result = await response.json()
    console.log('Success! Response:', result.choices[0].message.content)
    console.log('Usage:', result.usage)
  }
} catch (error) {
  console.error('Error:', error)
}