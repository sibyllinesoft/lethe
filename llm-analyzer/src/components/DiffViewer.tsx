import { useRef } from 'react'
import { DiffEditor } from '@monaco-editor/react'

interface DiffViewerProps {
  original: string
  modified: string
  language?: string
  title?: string
  height?: number
  disableCompression?: boolean
}

export default function DiffViewer({ 
  original, 
  modified, 
  language = 'json', 
  title,
  height = 600,
  disableCompression = false
}: DiffViewerProps) {
  const diffEditorRef = useRef<any>(null)

  const handleEditorDidMount = (editor: any) => {
    diffEditorRef.current = editor
    
    // Configure diff editor options
    editor.updateOptions({
      readOnly: true,
      renderSideBySide: true,
      enableSplitViewResizing: true,
      renderOverviewRuler: true,
      scrollBeyondLastLine: false,
      wordWrap: 'on',
      minimap: { enabled: false },
      fontSize: 12,
      lineNumbers: 'on',
      folding: true,
      glyphMargin: false,
      lineDecorationsWidth: 10,
      lineNumbersMinChars: 3,
      renderWhitespace: 'boundary',
      trimAutoWhitespace: false,
      renderLineHighlight: 'all',
    })
  }

  // Format JSON and compress large contexts for better readability
  const formatContent = (content: string, lang: string) => {
    if (lang === 'json') {
      try {
        const parsed = JSON.parse(content)
        const processed = disableCompression ? parsed : compressLargeContexts(parsed)
        return JSON.stringify(processed, null, 2)
      } catch {
        return content
      }
    }
    // For plaintext, ensure newlines are preserved
    if (lang === 'plaintext' || lang === 'text') {
      return content
    }
    return content
  }

  // Compress large text fields to show placeholders with size info
  const compressLargeContexts = (obj: any): any => {
    if (typeof obj === 'string') {
      if (obj.length > 5000) {
        const sizeKB = (obj.length / 1024).toFixed(1)
        const preview = obj.substring(0, 200)
        const suffix = obj.substring(obj.length - 200)
        return `${preview}\n\n[... COMPRESSED CONTEXT (${sizeKB}KB total) ...]\n\n${suffix}`
      }
      return obj
    }
    
    if (Array.isArray(obj)) {
      return obj.map(compressLargeContexts)
    }
    
    if (obj && typeof obj === 'object') {
      const compressed: any = {}
      for (const [key, value] of Object.entries(obj)) {
        if (key === 'content' && typeof value === 'string' && value.length > 5000) {
          const sizeKB = (value.length / 1024).toFixed(1)
          const preview = value.substring(0, 200)
          const suffix = value.substring(value.length - 200)
          compressed[key] = `${preview}\n\n[... COMPRESSED CONTEXT (${sizeKB}KB) - Question: ${extractQuestion(value)} ...]\n\n${suffix}`
        } else {
          compressed[key] = compressLargeContexts(value)
        }
      }
      return compressed
    }
    
    return obj
  }

  // Extract question from context for better understanding
  const extractQuestion = (content: string): string => {
    const questionMatch = content.match(/Question[:\s]*([^?\n]+\?)/i)
    if (questionMatch) return questionMatch[1].trim()
    
    const whichMatch = content.match(/Which\s+[^?\n]{10,60}\?/i)
    if (whichMatch) return whichMatch[0].trim()
    
    return "InfiniteBench code_debug task"
  }

  const formattedOriginal = formatContent(original, language)
  const formattedModified = formatContent(modified, language)

  return (
    <div className="diff-container">
      {title && (
        <div className="diff-header">
          <h3 className="text-sm font-medium">{title}</h3>
        </div>
      )}
      <div className="diff-content" style={{ height: height, overflow: 'visible', flex: 'none' }}>
        <DiffEditor
          height={height}
          theme="vs"
          language={language}
          original={formattedOriginal}
          modified={formattedModified}
          onMount={handleEditorDidMount}
          options={{
            readOnly: true,
            renderSideBySide: true,
            enableSplitViewResizing: true,
            renderOverviewRuler: true,
            scrollBeyondLastLine: false,
            wordWrap: 'on',
            minimap: { enabled: false },
            fontSize: 12,
            lineNumbers: 'on',
            folding: true,
            glyphMargin: false,
            lineDecorationsWidth: 10,
            lineNumbersMinChars: 3,
            renderWhitespace: 'boundary',
            trimAutoWhitespace: false,
            renderLineHighlight: 'all',
          }}
        />
      </div>
    </div>
  )
}

interface TextDiffProps {
  text1: string
  text2: string
  title?: string
}

export function TextDiff({ text1, text2, title }: TextDiffProps) {
  return (
    <DiffViewer
      original={text1}
      modified={text2}
      language="plaintext"
      title={title}
    />
  )
}

interface JsonDiffProps {
  obj1: any
  obj2: any
  title?: string
  height?: number
  disableCompression?: boolean
}

export function JsonDiff({ obj1, obj2, title, height = 600, disableCompression = false }: JsonDiffProps) {
  const json1 = typeof obj1 === 'string' ? obj1 : JSON.stringify(obj1, null, 2)
  const json2 = typeof obj2 === 'string' ? obj2 : JSON.stringify(obj2, null, 2)

  return (
    <DiffViewer
      original={json1}
      modified={json2}
      language="json"
      title={title}
      height={height}
      disableCompression={disableCompression}
    />
  )
}

interface InlineDiffProps {
  text: string
  className?: string
}

export function InlineDiff({ text, className }: InlineDiffProps) {
  return (
    <div 
      className={`inline-diff ${className || ''}`}
      dangerouslySetInnerHTML={{ __html: text }}
    />
  )
}

interface ContentViewerProps {
  content: string
  language?: string
  readOnly?: boolean
  height?: number
}

export function ContentViewer({ content, language = 'json', height = 300 }: ContentViewerProps) {
  return (
    <DiffViewer
      original=""
      modified={content}
      language={language}
      height={height}
    />
  )
}
