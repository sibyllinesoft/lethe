import { Routes, Route } from 'react-router-dom'
import CallsListPage from './pages/CallsListPage'
import CallViewPage from './pages/CallViewPage'
import CompareViewPage from './pages/CompareViewPage'

function App() {
  return (
    <div className="container">
      <header className="header">
        <h1>LLM Analyzer</h1>
        <div>
          <span className="pill pill-provider">Proxy Log Analysis</span>
        </div>
      </header>
      
      <Routes>
        <Route path="/" element={<CallsListPage />} />
        <Route path="/call/:id" element={<CallViewPage />} />
        <Route path="/compare" element={<CompareViewPage />} />
      </Routes>
    </div>
  )
}

export default App