import { useEffect, useRef } from 'react'
import { Editor } from '@monaco-editor/react'
import * as monaco from 'monaco-editor'

interface DiffViewerProps {
  original: string
  modified: string
  language?: string
  title?: string
  height?: number
}

export default function DiffViewer({ 
  original, 
  modified, 
  language = 'json', 
  title,
  height = 400 
}: DiffViewerProps) {
  const diffEditorRef = useRef<monaco.editor.IStandaloneDiffEditor | null>(null)

  const handleEditorDidMount = (editor: monaco.editor.IStandaloneDiffEditor) => {
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
    })
  }

  // Format JSON for better readability
  const formatContent = (content: string, lang: string) => {
    if (lang === 'json') {
      try {
        return JSON.stringify(JSON.parse(content), null, 2)
      } catch {
        return content
      }
    }
    return content
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
      <div className="diff-content" style={{ height: height }}>
        <Editor
          height="100%"
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
}

export function JsonDiff({ obj1, obj2, title }: JsonDiffProps) {
  const json1 = typeof obj1 === 'string' ? obj1 : JSON.stringify(obj1, null, 2)
  const json2 = typeof obj2 === 'string' ? obj2 : JSON.stringify(obj2, null, 2)

  return (
    <DiffViewer
      original={json1}
      modified={json2}
      language="json"
      title={title}
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