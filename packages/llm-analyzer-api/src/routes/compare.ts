import { Router } from 'express'
import { LogParser } from '../services/log-parser.js'
import { DiffService } from '../services/diff-service.js'

const router = Router()
const logParser = new LogParser()
const diffService = new DiffService()

// Cache for parsed logs (shared with calls route)
let cachedCalls: any[] = []
let lastCacheUpdate = 0
const CACHE_TTL = 5 * 60 * 1000 // 5 minutes

async function getCalls() {
  const now = Date.now()
  if (cachedCalls.length === 0 || now - lastCacheUpdate > CACHE_TTL) {
    console.log('Loading/refreshing calls cache...')
    cachedCalls = await logParser.loadSampleLogs()
    lastCacheUpdate = now
    console.log(`Loaded ${cachedCalls.length} calls`)
  }
  return cachedCalls
}

// POST /compare - Compare two calls
router.post('/', async (req, res) => {
  try {
    const { call_id_a, call_id_b } = req.body

    if (!call_id_a || !call_id_b) {
      return res.status(400).json({ 
        error: 'Both call_id_a and call_id_b are required' 
      })
    }

    const calls = await getCalls()
    const callA = calls.find(c => c.id === call_id_a)
    const callB = calls.find(c => c.id === call_id_b)

    if (!callA) {
      return res.status(404).json({ error: `Call ${call_id_a} not found` })
    }

    if (!callB) {
      return res.status(404).json({ error: `Call ${call_id_b} not found` })
    }

    const comparison = diffService.generateCallComparison(callA, callB)

    res.json(comparison)
  } catch (error) {
    console.error('Error comparing calls:', error)
    res.status(500).json({ error: 'Failed to compare calls' })
  }
})

// GET /compare?call_id_a=X&call_id_b=Y - Compare two calls via query params
router.get('/', async (req, res) => {
  try {
    const call_id_a = req.query.call_id_a as string
    const call_id_b = req.query.call_id_b as string

    if (!call_id_a || !call_id_b) {
      return res.status(400).json({ 
        error: 'Both call_id_a and call_id_b query parameters are required' 
      })
    }

    const calls = await getCalls()
    const callA = calls.find(c => c.id === call_id_a)
    const callB = calls.find(c => c.id === call_id_b)

    if (!callA) {
      return res.status(404).json({ error: `Call ${call_id_a} not found` })
    }

    if (!callB) {
      return res.status(404).json({ error: `Call ${call_id_b} not found` })
    }

    const comparison = diffService.generateCallComparison(callA, callB)

    res.json(comparison)
  } catch (error) {
    console.error('Error comparing calls:', error)
    res.status(500).json({ error: 'Failed to compare calls' })
  }
})

// GET /compare/runs/:run_id - Find pairs of calls within a run for A/B comparison
router.get('/runs/:run_id', async (req, res) => {
  try {
    const runId = req.params.run_id
    const calls = await getCalls()
    
    // Find calls in this run
    const runCalls = calls.filter(call => call.run_id === runId)
    
    if (runCalls.length === 0) {
      return res.status(404).json({ error: `No calls found for run ${runId}` })
    }

    // Group by query_id to find potential pairs
    const groupedByQuery = runCalls.reduce((acc, call) => {
      if (!acc[call.query_id]) {
        acc[call.query_id] = []
      }
      acc[call.query_id].push(call)
      return acc
    }, {} as Record<string, any[]>)

    // Find queries with multiple calls (potential A/B pairs)
    const pairs = []
    for (const [queryId, queryCalls] of Object.entries(groupedByQuery)) {
      if (queryCalls.length >= 2) {
        // Sort by timestamp to identify pre/post or different configurations
        queryCalls.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
        
        // Create pairs from consecutive calls
        for (let i = 0; i < queryCalls.length - 1; i++) {
          pairs.push({
            query_id: queryId,
            call_a: queryCalls[i],
            call_b: queryCalls[i + 1],
            time_diff_ms: new Date(queryCalls[i + 1].timestamp).getTime() - new Date(queryCalls[i].timestamp).getTime()
          })
        }
      }
    }

    res.json({
      run_id: runId,
      total_calls: runCalls.length,
      unique_queries: Object.keys(groupedByQuery).length,
      potential_pairs: pairs.length,
      pairs: pairs.slice(0, 20) // Limit to first 20 pairs
    })
  } catch (error) {
    console.error('Error finding run pairs:', error)
    res.status(500).json({ error: 'Failed to find run pairs' })
  }
})

export default router