import { Router } from 'express'
import { LogParser } from '../services/log-parser.js'
import { DiffService } from '../services/diff-service.js'
import { CallsFilters } from '../types.js'

const router = Router()
const logParser = new LogParser()
const diffService = new DiffService()

// Cache for parsed logs
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

// GET /calls - List calls with filtering and pagination
router.get('/', async (req, res) => {
  try {
    const filters: CallsFilters = {
      since: req.query.since as string,
      run_id: req.query.run_id as string,
      provider: req.query.provider as string,
      model: req.query.model as string,
      status: req.query.status as string,
      benchmark_type: req.query.benchmark_type as string,
      dataset: req.query.dataset as string,
      page: parseInt(req.query.page as string) || 1,
      limit: parseInt(req.query.limit as string) || 50
    }

    let calls = await getCalls()

    // Apply filters
    if (filters.since) {
      const sinceDate = new Date(filters.since)
      calls = calls.filter(call => new Date(call.timestamp) >= sinceDate)
    }

    if (filters.run_id) {
      calls = calls.filter(call => call.run_id === filters.run_id)
    }

    if (filters.provider) {
      calls = calls.filter(call => call.provider === filters.provider)
    }

    if (filters.model) {
      calls = calls.filter(call => call.model === filters.model)
    }

    if (filters.status) {
      calls = calls.filter(call => call.status === filters.status)
    }

    if (filters.benchmark_type) {
      calls = calls.filter(call => call.benchmark_type === filters.benchmark_type)
    }

    if (filters.dataset) {
      calls = calls.filter(call => call.dataset === filters.dataset)
    }

    // Apply pagination
    const total = calls.length
    const startIndex = (filters.page! - 1) * filters.limit!
    const endIndex = startIndex + filters.limit!
    const paginatedCalls = calls.slice(startIndex, endIndex)

    res.json({
      calls: paginatedCalls,
      total,
      page: filters.page,
      limit: filters.limit,
      totalPages: Math.ceil(total / filters.limit!)
    })
  } catch (error) {
    console.error('Error fetching calls:', error)
    res.status(500).json({ error: 'Failed to fetch calls' })
  }
})

// GET /stats - Get summary statistics
router.get('/stats', async (req, res) => {
  try {
    const calls = await getCalls()
    
    const stats = {
      total_calls: calls.length,
      providers: [...new Set(calls.map(c => c.provider))],
      models: [...new Set(calls.map(c => c.model))],
      run_ids: [...new Set(calls.map(c => c.run_id))],
      benchmark_types: [...new Set(calls.map(c => c.benchmark_type))],
      datasets: [...new Set(calls.map(c => c.dataset))],
      status_counts: calls.reduce((acc, call) => {
        acc[call.status] = (acc[call.status] || 0) + 1
        return acc
      }, {} as Record<string, number>),
      avg_latency_ms: calls.reduce((sum, call) => sum + call.latency_ms, 0) / calls.length,
      total_input_tokens: calls.reduce((sum, call) => sum + call.input_tokens, 0),
      total_output_tokens: calls.reduce((sum, call) => sum + call.output_tokens, 0)
    }

    res.json(stats)
  } catch (error) {
    console.error('Error fetching stats:', error)
    res.status(500).json({ error: 'Failed to fetch stats' })
  }
})

// GET /runs - Get unique run IDs
router.get('/runs', async (req, res) => {
  try {
    const calls = await getCalls()
    const runIds = [...new Set(calls.map(call => call.run_id))]
    
    res.json({ run_ids: runIds })
  } catch (error) {
    console.error('Error fetching runs:', error)
    res.status(500).json({ error: 'Failed to fetch runs' })
  }
})

// GET /calls/:id - Get specific call details
router.get('/:id', async (req, res) => {
  try {
    const calls = await getCalls()
    const call = calls.find(c => c.id === req.params.id)
    
    if (!call) {
      return res.status(404).json({ error: 'Call not found' })
    }

    res.json(call)
  } catch (error) {
    console.error('Error fetching call:', error)
    res.status(500).json({ error: 'Failed to fetch call' })
  }
})

// GET /calls/:id/pre-post-diff - Get pre/post transformation diff for a call
router.get('/:id/pre-post-diff', async (req, res) => {
  try {
    const calls = await getCalls()
    const call = calls.find(c => c.id === req.params.id)
    
    if (!call) {
      return res.status(404).json({ error: 'Call not found' })
    }

    const diff = diffService.generatePrePostComparison(call)
    
    if (!diff) {
      return res.status(400).json({ error: 'No transformation data available for this call' })
    }

    res.json(diff)
  } catch (error) {
    console.error('Error generating pre/post diff:', error)
    res.status(500).json({ error: 'Failed to generate diff' })
  }
})

export default router