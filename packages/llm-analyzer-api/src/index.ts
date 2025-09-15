import express from 'express'
import cors from 'cors'
import callsRouter from './routes/calls.js'
import compareRouter from './routes/compare.js'

const app = express()
const PORT = process.env.PORT || 3002

// Middleware
app.use(cors())
app.use(express.json({ limit: '10mb' }))
app.use(express.urlencoded({ extended: true, limit: '10mb' }))

// Routes
app.use('/api/calls', callsRouter)
app.use('/api/compare', compareRouter)

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() })
})

// Debug endpoint
app.get('/debug', (req, res) => {
  res.json({
    message: 'Debug endpoint working',
    timestamp: new Date().toISOString(),
    query: req.query,
    headers: {
      origin: req.get('origin'),
      userAgent: req.get('user-agent')
    }
  })
})

// Root endpoint
app.get('/', (req, res) => {
  res.json({ 
    message: 'LLM Analyzer API',
    version: '0.1.0',
    endpoints: {
      calls: '/api/calls',
      compare: '/api/compare',
      health: '/api/health',
      debug: '/debug'
    }
  })
})

// Error handling
app.use((error: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('Error:', error)
  res.status(500).json({ 
    error: 'Internal server error',
    message: error.message 
  })
})

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Endpoint not found' })
})

app.listen(PORT, () => {
  console.log(`LLM Analyzer API running on port ${PORT}`)
  console.log(`Health check: http://localhost:${PORT}/health`)
  console.log(`API docs: http://localhost:${PORT}/`)
})