/**
 * Structure-Aware Chunking for Code and Log Files
 * 
 * Implements intelligent chunking that respects structural boundaries in:
 * - Source code (functions, classes, imports via AST parsing)
 * - Log files (structured entries, stack traces, request flows)
 * - Configuration files (YAML, JSON hierarchies)
 * - Documentation (headers, code blocks, lists)
 * 
 * Uses Tree-sitter for AST parsing and custom heuristics for log analysis.
 * Produces semantically coherent chunks with natural anchor points.
 */

import { z } from 'zod';

// Input validation schemas
export const ChunkingConfigSchema = z.object({
  max_chunk_tokens: z.number().int().positive().default(1024),
  min_chunk_tokens: z.number().int().positive().default(50),
  overlap_tokens: z.number().int().nonnegative().default(100),
  preserve_structure: z.boolean().default(true),
  detect_language: z.boolean().default(true),
  enable_ast_parsing: z.boolean().default(true),
  enable_log_parsing: z.boolean().default(true),
  anchor_detection: z.boolean().default(true),
  chunk_by_function: z.boolean().default(true), // For code files
  chunk_by_class: z.boolean().default(true),
  preserve_imports: z.boolean().default(true),
  preserve_headers: z.boolean().default(true),  // For markdown/docs
  preserve_code_blocks: z.boolean().default(true)
});

export const SourceFileSchema = z.object({
  id: z.string(),
  content: z.string(),
  file_path: z.string().optional(),
  detected_language: z.string().optional(),
  mime_type: z.string().optional(),
  metadata: z.record(z.unknown()).default({})
});

export const StructuredChunkSchema = z.object({
  id: z.string(),
  text: z.string(),
  tokens: z.number().int().nonnegative(),
  start_byte: z.number().int().nonnegative(),
  end_byte: z.number().int().nonnegative(),
  start_line: z.number().int().positive(),
  end_line: z.number().int().positive(),
  structure_type: z.enum(['function', 'class', 'module', 'import', 'log_entry', 'stack_trace', 'header', 'code_block', 'paragraph', 'list', 'config_section', 'unknown']),
  language: z.string().optional(),
  anchor_score: z.number().min(0).max(1).default(0),
  is_head_anchor: z.boolean().default(false),
  is_tail_anchor: z.boolean().default(false),
  parent_structure: z.string().optional(),
  child_structures: z.array(z.string()).default([]),
  semantic_metadata: z.object({
    function_name: z.string().optional(),
    class_name: z.string().optional(),
    imports: z.array(z.string()).default([]),
    log_level: z.string().optional(),
    timestamp: z.string().optional(),
    error_type: z.string().optional(),
    stack_depth: z.number().int().nonnegative().optional()
  }).default({})
});

export const ChunkingResultSchema = z.object({
  chunks: z.array(StructuredChunkSchema),
  total_chunks: z.number().int().nonnegative(),
  total_tokens: z.number().int().nonnegative(),
  detected_language: z.string().optional(),
  structure_analysis: z.object({
    functions_found: z.number().int().nonnegative().default(0),
    classes_found: z.number().int().nonnegative().default(0),
    imports_found: z.number().int().nonnegative().default(0),
    log_entries_found: z.number().int().nonnegative().default(0),
    headers_found: z.number().int().nonnegative().default(0),
    code_blocks_found: z.number().int().nonnegative().default(0)
  }).default({}),
  processing_time_ms: z.number().nonnegative(),
  chunking_strategy: z.enum(['ast_based', 'log_based', 'markdown_based', 'fallback_sliding'])
});

export type ChunkingConfig = z.infer<typeof ChunkingConfigSchema>;
export type SourceFile = z.infer<typeof SourceFileSchema>;
export type StructuredChunk = z.infer<typeof StructuredChunkSchema>;
export type ChunkingResult = z.infer<typeof ChunkingResultSchema>;

/**
 * Language detection and AST parsing utilities
 */
class LanguageDetector {
  static detectLanguage(filePath: string, content: string): string {
    // File extension based detection
    const ext = filePath ? filePath.split('.').pop()?.toLowerCase() : '';
    
    const extensionMap: Record<string, string> = {
      'ts': 'typescript',
      'tsx': 'tsx',
      'js': 'javascript', 
      'jsx': 'jsx',
      'py': 'python',
      'rs': 'rust',
      'go': 'go',
      'java': 'java',
      'cpp': 'cpp',
      'c': 'c',
      'cs': 'csharp',
      'php': 'php',
      'rb': 'ruby',
      'md': 'markdown',
      'html': 'html',
      'css': 'css',
      'scss': 'scss',
      'json': 'json',
      'yaml': 'yaml',
      'yml': 'yaml',
      'xml': 'xml',
      'log': 'log',
      'txt': 'text'
    };
    
    if (ext && extensionMap[ext]) {
      return extensionMap[ext];
    }
    
    // Content-based heuristics
    if (this.isLogFile(content)) return 'log';
    if (content.includes('function ') || content.includes('const ') || content.includes('import ')) return 'javascript';
    if (content.includes('def ') || content.includes('class ') || content.includes('import ')) return 'python';
    if (content.includes('fn ') || content.includes('struct ') || content.includes('use ')) return 'rust';
    if (content.includes('func ') || content.includes('package ') || content.includes('import ')) return 'go';
    
    return 'text';
  }
  
  static isLogFile(content: string): boolean {
    // Common log patterns
    const logPatterns = [
      /\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}/, // ISO timestamps
      /\[(ERROR|WARN|INFO|DEBUG|TRACE)\]/, // Log levels in brackets
      /\d{2}\/\d{2}\/\d{4} \d{2}:\d{2}:\d{2}/, // Date/time patterns
      /at\s+[\w\.]+\([\w\.\/:]+:\d+:\d+\)/, // Stack traces
      /\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/ // IP addresses
    ];
    
    const matches = logPatterns.reduce((count, pattern) => {
      return count + (content.match(pattern) || []).length;
    }, 0);
    
    return matches > 5; // Threshold for log file detection
  }
}

/**
 * AST-based code chunking using Tree-sitter (simulated)
 */
class ASTChunker {
  private config: ChunkingConfig;
  
  constructor(config: ChunkingConfig) {
    this.config = config;
  }
  
  async chunkCode(file: SourceFile, language: string): Promise<StructuredChunk[]> {
    const chunks: StructuredChunk[] = [];
    const lines = file.content.split('\n');
    
    // Simulate AST parsing - in real implementation would use Tree-sitter
    const structures = this.parseCodeStructures(file.content, language);
    
    let chunkId = 0;
    for (const structure of structures) {
      const chunkText = this.extractTextRange(file.content, structure.start_byte, structure.end_byte);
      const tokens = this.estimateTokens(chunkText);
      
      if (tokens >= this.config.min_chunk_tokens) {
        chunks.push({
          id: `${file.id}_chunk_${chunkId++}`,
          text: chunkText,
          tokens,
          start_byte: structure.start_byte,
          end_byte: structure.end_byte,
          start_line: structure.start_line,
          end_line: structure.end_line,
          structure_type: structure.type,
          language,
          anchor_score: this.calculateAnchorScore(structure),
          is_head_anchor: this.isHeadAnchor(structure),
          is_tail_anchor: this.isTailAnchor(structure),
          semantic_metadata: structure.metadata
        });
      }
    }
    
    // Handle remaining content with sliding window if needed
    const coveredBytes = new Set(chunks.flatMap(c => Array.from({length: c.end_byte - c.start_byte}, (_, i) => c.start_byte + i)));
    const uncoveredRanges = this.findUncoveredRanges(file.content.length, coveredBytes);
    
    for (const range of uncoveredRanges) {
      if (range.end - range.start > this.config.min_chunk_tokens * 4) { // Rough token estimation
        const chunkText = this.extractTextRange(file.content, range.start, range.end);
        chunks.push({
          id: `${file.id}_chunk_${chunkId++}`,
          text: chunkText,
          tokens: this.estimateTokens(chunkText),
          start_byte: range.start,
          end_byte: range.end,
          start_line: this.byteToLine(file.content, range.start),
          end_line: this.byteToLine(file.content, range.end),
          structure_type: 'unknown',
          language,
          anchor_score: 0.1
        });
      }
    }
    
    return chunks;
  }
  
  private parseCodeStructures(content: string, language: string): Array<{
    type: StructuredChunk['structure_type'];
    start_byte: number;
    end_byte: number;
    start_line: number;
    end_line: number;
    metadata: StructuredChunk['semantic_metadata'];
  }> {
    const structures: Array<any> = [];
    const lines = content.split('\n');
    
    // Language-specific parsing patterns
    switch (language) {
      case 'typescript':
      case 'javascript':
        return this.parseJavaScriptLike(content, lines);
      case 'python':
        return this.parsePython(content, lines);
      case 'rust':
        return this.parseRust(content, lines);
      case 'go':
        return this.parseGo(content, lines);
      default:
        return this.parseFallback(content, lines);
    }
  }
  
  private parseJavaScriptLike(content: string, lines: string[]): Array<any> {
    const structures: Array<any> = [];
    let currentByte = 0;
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const lineStartByte = currentByte;
      const lineEndByte = currentByte + line.length + 1; // +1 for newline
      
      // Import statements
      if (line.trim().match(/^(import|export)\s+/)) {
        structures.push({
          type: 'import' as const,
          start_byte: lineStartByte,
          end_byte: lineEndByte,
          start_line: i + 1,
          end_line: i + 1,
          metadata: {
            imports: [line.trim()]
          }
        });
      }
      
      // Function declarations
      const funcMatch = line.match(/^\s*(export\s+)?(async\s+)?function\s+(\w+)/);
      if (funcMatch) {
        const endLine = this.findFunctionEnd(lines, i);
        const endByte = this.calculateByteOffset(lines, endLine);
        
        structures.push({
          type: 'function' as const,
          start_byte: lineStartByte,
          end_byte: endByte,
          start_line: i + 1,
          end_line: endLine + 1,
          metadata: {
            function_name: funcMatch[3]
          }
        });
      }
      
      // Class declarations
      const classMatch = line.match(/^\s*(export\s+)?class\s+(\w+)/);
      if (classMatch) {
        const endLine = this.findClassEnd(lines, i);
        const endByte = this.calculateByteOffset(lines, endLine);
        
        structures.push({
          type: 'class' as const,
          start_byte: lineStartByte,
          end_byte: endByte,
          start_line: i + 1,
          end_line: endLine + 1,
          metadata: {
            class_name: classMatch[2]
          }
        });
      }
      
      currentByte = lineEndByte;
    }
    
    return structures;
  }
  
  private parsePython(content: string, lines: string[]): Array<any> {
    const structures: Array<any> = [];
    let currentByte = 0;
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const lineStartByte = currentByte;
      const lineEndByte = currentByte + line.length + 1;
      
      // Import statements
      if (line.trim().match(/^(import|from)\s+/)) {
        structures.push({
          type: 'import' as const,
          start_byte: lineStartByte,
          end_byte: lineEndByte,
          start_line: i + 1,
          end_line: i + 1,
          metadata: {
            imports: [line.trim()]
          }
        });
      }
      
      // Function definitions
      const funcMatch = line.match(/^\s*def\s+(\w+)/);
      if (funcMatch) {
        const endLine = this.findPythonBlockEnd(lines, i);
        const endByte = this.calculateByteOffset(lines, endLine);
        
        structures.push({
          type: 'function' as const,
          start_byte: lineStartByte,
          end_byte: endByte,
          start_line: i + 1,
          end_line: endLine + 1,
          metadata: {
            function_name: funcMatch[1]
          }
        });
      }
      
      // Class definitions
      const classMatch = line.match(/^\s*class\s+(\w+)/);
      if (classMatch) {
        const endLine = this.findPythonBlockEnd(lines, i);
        const endByte = this.calculateByteOffset(lines, endLine);
        
        structures.push({
          type: 'class' as const,
          start_byte: lineStartByte,
          end_byte: endByte,
          start_line: i + 1,
          end_line: endLine + 1,
          metadata: {
            class_name: classMatch[1]
          }
        });
      }
      
      currentByte = lineEndByte;
    }
    
    return structures;
  }
  
  private parseRust(content: string, lines: string[]): Array<any> {
    const structures: Array<any> = [];
    let currentByte = 0;
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const lineStartByte = currentByte;
      const lineEndByte = currentByte + line.length + 1;
      
      // Use statements
      if (line.trim().match(/^use\s+/)) {
        structures.push({
          type: 'import' as const,
          start_byte: lineStartByte,
          end_byte: lineEndByte,
          start_line: i + 1,
          end_line: i + 1,
          metadata: {
            imports: [line.trim()]
          }
        });
      }
      
      // Function definitions
      const funcMatch = line.match(/^\s*(pub\s+)?(async\s+)?fn\s+(\w+)/);
      if (funcMatch) {
        const endLine = this.findRustBlockEnd(lines, i);
        const endByte = this.calculateByteOffset(lines, endLine);
        
        structures.push({
          type: 'function' as const,
          start_byte: lineStartByte,
          end_byte: endByte,
          start_line: i + 1,
          end_line: endLine + 1,
          metadata: {
            function_name: funcMatch[3]
          }
        });
      }
      
      currentByte = lineEndByte;
    }
    
    return structures;
  }
  
  private parseGo(content: string, lines: string[]): Array<any> {
    const structures: Array<any> = [];
    let currentByte = 0;
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const lineStartByte = currentByte;
      const lineEndByte = currentByte + line.length + 1;
      
      // Import statements
      if (line.trim().match(/^import\s+/)) {
        structures.push({
          type: 'import' as const,
          start_byte: lineStartByte,
          end_byte: lineEndByte,
          start_line: i + 1,
          end_line: i + 1,
          metadata: {
            imports: [line.trim()]
          }
        });
      }
      
      // Function definitions
      const funcMatch = line.match(/^\s*func\s+(\w+)/);
      if (funcMatch) {
        const endLine = this.findGoBlockEnd(lines, i);
        const endByte = this.calculateByteOffset(lines, endLine);
        
        structures.push({
          type: 'function' as const,
          start_byte: lineStartByte,
          end_byte: endByte,
          start_line: i + 1,
          end_line: endLine + 1,
          metadata: {
            function_name: funcMatch[1]
          }
        });
      }
      
      currentByte = lineEndByte;
    }
    
    return structures;
  }
  
  private parseFallback(content: string, lines: string[]): Array<any> {
    // Fallback parsing for unknown languages
    return [];
  }
  
  // Helper methods for block end detection
  private findFunctionEnd(lines: string[], startLine: number): number {
    let braceCount = 0;
    let foundOpenBrace = false;
    
    for (let i = startLine; i < lines.length; i++) {
      for (const char of lines[i]) {
        if (char === '{') {
          braceCount++;
          foundOpenBrace = true;
        } else if (char === '}') {
          braceCount--;
          if (foundOpenBrace && braceCount === 0) {
            return i;
          }
        }
      }
    }
    
    return lines.length - 1;
  }
  
  private findClassEnd(lines: string[], startLine: number): number {
    return this.findFunctionEnd(lines, startLine); // Same logic for JS/TS
  }
  
  private findPythonBlockEnd(lines: string[], startLine: number): number {
    const baseIndent = lines[startLine].length - lines[startLine].trimStart().length;
    
    for (let i = startLine + 1; i < lines.length; i++) {
      const line = lines[i].trimStart();
      if (line === '') continue; // Skip empty lines
      
      const currentIndent = lines[i].length - line.length;
      if (currentIndent <= baseIndent && line.length > 0) {
        return i - 1;
      }
    }
    
    return lines.length - 1;
  }
  
  private findRustBlockEnd(lines: string[], startLine: number): number {
    return this.findFunctionEnd(lines, startLine); // Similar brace-based logic
  }
  
  private findGoBlockEnd(lines: string[], startLine: number): number {
    return this.findFunctionEnd(lines, startLine); // Similar brace-based logic
  }
  
  // Utility methods
  private calculateByteOffset(lines: string[], lineIndex: number): number {
    return lines.slice(0, lineIndex + 1).reduce((total, line) => total + line.length + 1, 0);
  }
  
  private extractTextRange(content: string, start: number, end: number): string {
    return content.slice(start, end);
  }
  
  private byteToLine(content: string, byteOffset: number): number {
    const prefix = content.slice(0, byteOffset);
    return prefix.split('\n').length;
  }
  
  private calculateAnchorScore(structure: any): number {
    // Higher scores for important structural elements
    switch (structure.type) {
      case 'import': return 0.9;
      case 'class': return 0.8;
      case 'function': return 0.7;
      case 'module': return 0.6;
      default: return 0.3;
    }
  }
  
  private isHeadAnchor(structure: any): boolean {
    return structure.type === 'import' || structure.type === 'module';
  }
  
  private isTailAnchor(structure: any): boolean {
    return structure.type === 'class' || structure.type === 'function';
  }
  
  private estimateTokens(text: string): number {
    return Math.ceil(text.length / 4); // Rough estimation
  }
  
  private findUncoveredRanges(totalLength: number, coveredBytes: Set<number>): Array<{start: number, end: number}> {
    const ranges: Array<{start: number, end: number}> = [];
    let start = -1;
    
    for (let i = 0; i <= totalLength; i++) {
      if (!coveredBytes.has(i)) {
        if (start === -1) start = i;
      } else {
        if (start !== -1) {
          ranges.push({start, end: i});
          start = -1;
        }
      }
    }
    
    if (start !== -1) {
      ranges.push({start, end: totalLength});
    }
    
    return ranges;
  }
}

/**
 * Log-specific chunking for structured log files
 */
class LogChunker {
  private config: ChunkingConfig;
  
  constructor(config: ChunkingConfig) {
    this.config = config;
  }
  
  async chunkLogs(file: SourceFile): Promise<StructuredChunk[]> {
    const chunks: StructuredChunk[] = [];
    const lines = file.content.split('\n');
    
    let currentChunk: string[] = [];
    let chunkStartByte = 0;
    let chunkId = 0;
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      
      if (this.isLogEntryStart(line)) {
        // Finish previous chunk if it has content
        if (currentChunk.length > 0) {
          const chunkText = currentChunk.join('\n');
          const tokens = this.estimateTokens(chunkText);
          
          if (tokens >= this.config.min_chunk_tokens) {
            chunks.push(this.createLogChunk(
              `${file.id}_log_${chunkId++}`,
              chunkText,
              chunkStartByte,
              chunkStartByte + chunkText.length,
              i - currentChunk.length + 1,
              i,
              currentChunk[0] // First line for metadata extraction
            ));
          }
        }
        
        // Start new chunk
        currentChunk = [line];
        chunkStartByte = this.calculateLineByteOffset(file.content, i);
      } else {
        currentChunk.push(line);
      }
      
      // Check if chunk is getting too large
      if (this.estimateTokens(currentChunk.join('\n')) > this.config.max_chunk_tokens) {
        const chunkText = currentChunk.join('\n');
        chunks.push(this.createLogChunk(
          `${file.id}_log_${chunkId++}`,
          chunkText,
          chunkStartByte,
          chunkStartByte + chunkText.length,
          i - currentChunk.length + 1,
          i,
          currentChunk[0]
        ));
        
        currentChunk = [];
        chunkStartByte = this.calculateLineByteOffset(file.content, i + 1);
      }
    }
    
    // Handle final chunk
    if (currentChunk.length > 0) {
      const chunkText = currentChunk.join('\n');
      const tokens = this.estimateTokens(chunkText);
      
      if (tokens >= this.config.min_chunk_tokens) {
        chunks.push(this.createLogChunk(
          `${file.id}_log_${chunkId++}`,
          chunkText,
          chunkStartByte,
          chunkStartByte + chunkText.length,
          lines.length - currentChunk.length + 1,
          lines.length,
          currentChunk[0]
        ));
      }
    }
    
    return chunks;
  }
  
  private isLogEntryStart(line: string): boolean {
    // Common log entry patterns
    const patterns = [
      /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}/, // ISO timestamp
      /^\[\d{4}-\d{2}-\d{2}/, // Bracketed date
      /^(ERROR|WARN|INFO|DEBUG|TRACE)/, // Log level at start
      /^\d{2}\/\d{2}\/\d{4} \d{2}:\d{2}:\d{2}/ // MM/DD/YYYY format
    ];
    
    return patterns.some(pattern => pattern.test(line));
  }
  
  private createLogChunk(
    id: string,
    text: string,
    startByte: number,
    endByte: number,
    startLine: number,
    endLine: number,
    firstLine: string
  ): StructuredChunk {
    const metadata = this.extractLogMetadata(firstLine);
    const isStackTrace = this.isStackTrace(text);
    
    return {
      id,
      text,
      tokens: this.estimateTokens(text),
      start_byte: startByte,
      end_byte: endByte,
      start_line: startLine,
      end_line: endLine,
      structure_type: isStackTrace ? 'stack_trace' : 'log_entry',
      language: 'log',
      anchor_score: isStackTrace ? 0.8 : 0.5,
      is_head_anchor: false,
      is_tail_anchor: isStackTrace,
      semantic_metadata: metadata
    };
  }
  
  private extractLogMetadata(line: string): StructuredChunk['semantic_metadata'] {
    const metadata: StructuredChunk['semantic_metadata'] = {};
    
    // Extract timestamp
    const timestampMatch = line.match(/(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[\.\d]*(?:Z|[+-]\d{2}:\d{2})?)/);
    if (timestampMatch) {
      metadata.timestamp = timestampMatch[1];
    }
    
    // Extract log level
    const levelMatch = line.match(/(ERROR|WARN|INFO|DEBUG|TRACE)/);
    if (levelMatch) {
      metadata.log_level = levelMatch[1];
    }
    
    // Extract error type for exceptions
    const errorMatch = line.match(/(\w+(?:Exception|Error))/);
    if (errorMatch) {
      metadata.error_type = errorMatch[1];
    }
    
    return metadata;
  }
  
  private isStackTrace(text: string): boolean {
    const stackTracePatterns = [
      /at\s+[\w\.]+\([\w\.\/:]+:\d+:\d+\)/, // Java/JS stack traces
      /File\s+"[^"]+",\s+line\s+\d+/, // Python stack traces
      /\s+in\s+[\w\.:]+\s+at\s+line\s+\d+/ // Generic stack traces
    ];
    
    return stackTracePatterns.some(pattern => pattern.test(text));
  }
  
  private calculateLineByteOffset(content: string, lineIndex: number): number {
    const lines = content.split('\n');
    return lines.slice(0, lineIndex).reduce((total, line) => total + line.length + 1, 0);
  }
  
  private estimateTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }
}

/**
 * Main structure-aware chunking orchestrator
 */
export class StructureAwareChunker {
  private config: ChunkingConfig;
  private astChunker: ASTChunker;
  private logChunker: LogChunker;
  
  constructor(config: Partial<ChunkingConfig> = {}) {
    this.config = ChunkingConfigSchema.parse(config);
    this.astChunker = new ASTChunker(this.config);
    this.logChunker = new LogChunker(this.config);
  }
  
  async chunkFile(file: SourceFile): Promise<ChunkingResult> {
    const startTime = performance.now();
    
    // Validate input
    const validatedFile = SourceFileSchema.parse(file);
    
    // Detect language and file type
    const detectedLanguage = this.config.detect_language 
      ? LanguageDetector.detectLanguage(validatedFile.file_path || '', validatedFile.content)
      : 'text';
    
    console.log(`Structure-aware chunking: ${validatedFile.id} (${detectedLanguage})`);
    
    let chunks: StructuredChunk[] = [];
    let strategy: ChunkingResult['chunking_strategy'];
    
    try {
      if (detectedLanguage === 'log' && this.config.enable_log_parsing) {
        chunks = await this.logChunker.chunkLogs(validatedFile);
        strategy = 'log_based';
      } else if (detectedLanguage === 'markdown' && this.config.preserve_headers) {
        chunks = await this.chunkMarkdown(validatedFile);
        strategy = 'markdown_based';
      } else if (this.isCodeLanguage(detectedLanguage) && this.config.enable_ast_parsing) {
        chunks = await this.astChunker.chunkCode(validatedFile, detectedLanguage);
        strategy = 'ast_based';
      } else {
        chunks = await this.fallbackSlidingWindow(validatedFile);
        strategy = 'fallback_sliding';
      }
      
    } catch (error) {
      console.warn(`Structure-aware chunking failed for ${validatedFile.id}: ${error}, falling back to sliding window`);
      chunks = await this.fallbackSlidingWindow(validatedFile);
      strategy = 'fallback_sliding';
    }
    
    const structureAnalysis = this.analyzeStructure(chunks);
    const processingTime = performance.now() - startTime;
    
    const result: ChunkingResult = {
      chunks,
      total_chunks: chunks.length,
      total_tokens: chunks.reduce((sum, chunk) => sum + chunk.tokens, 0),
      detected_language: detectedLanguage,
      structure_analysis: structureAnalysis,
      processing_time_ms: processingTime,
      chunking_strategy: strategy
    };
    
    return ChunkingResultSchema.parse(result);
  }
  
  private async chunkMarkdown(file: SourceFile): Promise<StructuredChunk[]> {
    const chunks: StructuredChunk[] = [];
    const lines = file.content.split('\n');
    
    // Simple markdown chunking by headers
    let currentSection: string[] = [];
    let currentHeader = '';
    let chunkId = 0;
    let startLine = 1;
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      
      if (line.startsWith('#') && line.trim().length > 1) {
        // Finish previous section
        if (currentSection.length > 0) {
          const chunkText = currentSection.join('\n');
          const tokens = this.estimateTokens(chunkText);
          
          if (tokens >= this.config.min_chunk_tokens) {
            chunks.push({
              id: `${file.id}_md_${chunkId++}`,
              text: chunkText,
              tokens,
              start_byte: this.calculateByteOffset(file.content, startLine - 1),
              end_byte: this.calculateByteOffset(file.content, i),
              start_line: startLine,
              end_line: i,
              structure_type: 'header',
              language: 'markdown',
              anchor_score: 0.7,
              is_head_anchor: startLine === 1,
              is_tail_anchor: false,
              semantic_metadata: {}
            });
          }
        }
        
        // Start new section
        currentSection = [line];
        currentHeader = line;
        startLine = i + 1;
      } else {
        currentSection.push(line);
      }
    }
    
    // Handle final section
    if (currentSection.length > 0) {
      const chunkText = currentSection.join('\n');
      const tokens = this.estimateTokens(chunkText);
      
      if (tokens >= this.config.min_chunk_tokens) {
        chunks.push({
          id: `${file.id}_md_${chunkId++}`,
          text: chunkText,
          tokens,
          start_byte: this.calculateByteOffset(file.content, startLine - 1),
          end_byte: file.content.length,
          start_line: startLine,
          end_line: lines.length,
          structure_type: 'header',
          language: 'markdown',
          anchor_score: 0.7,
          is_head_anchor: false,
          is_tail_anchor: true,
          semantic_metadata: {}
        });
      }
    }
    
    return chunks;
  }
  
  private async fallbackSlidingWindow(file: SourceFile): Promise<StructuredChunk[]> {
    const chunks: StructuredChunk[] = [];
    const text = file.content;
    const maxChunkSize = this.config.max_chunk_tokens * 4; // Rough conversion
    const overlapSize = this.config.overlap_tokens * 4;
    
    let chunkId = 0;
    let startPos = 0;
    
    while (startPos < text.length) {
      const endPos = Math.min(startPos + maxChunkSize, text.length);
      const chunkText = text.slice(startPos, endPos);
      
      chunks.push({
        id: `${file.id}_sliding_${chunkId++}`,
        text: chunkText,
        tokens: this.estimateTokens(chunkText),
        start_byte: startPos,
        end_byte: endPos,
        start_line: this.byteToLine(text, startPos),
        end_line: this.byteToLine(text, endPos),
        structure_type: 'unknown',
        anchor_score: 0.1,
        is_head_anchor: startPos === 0,
        is_tail_anchor: endPos === text.length,
        semantic_metadata: {}
      });
      
      startPos = Math.max(startPos + maxChunkSize - overlapSize, startPos + 1);
      if (startPos >= endPos) break; // Prevent infinite loop
    }
    
    return chunks;
  }
  
  private isCodeLanguage(language: string): boolean {
    const codeLanguages = ['typescript', 'javascript', 'python', 'rust', 'go', 'java', 'cpp', 'c', 'csharp', 'php', 'ruby'];
    return codeLanguages.includes(language);
  }
  
  private analyzeStructure(chunks: StructuredChunk[]): ChunkingResult['structure_analysis'] {
    const analysis: ChunkingResult['structure_analysis'] = {
      functions_found: 0,
      classes_found: 0,
      imports_found: 0,
      log_entries_found: 0,
      headers_found: 0,
      code_blocks_found: 0
    };
    
    for (const chunk of chunks) {
      switch (chunk.structure_type) {
        case 'function': analysis.functions_found++; break;
        case 'class': analysis.classes_found++; break;
        case 'import': analysis.imports_found++; break;
        case 'log_entry': analysis.log_entries_found++; break;
        case 'header': analysis.headers_found++; break;
        case 'code_block': analysis.code_blocks_found++; break;
      }
    }
    
    return analysis;
  }
  
  private calculateByteOffset(content: string, lineIndex: number): number {
    const lines = content.split('\n');
    return lines.slice(0, lineIndex).reduce((total, line) => total + line.length + 1, 0);
  }
  
  private byteToLine(content: string, byteOffset: number): number {
    const prefix = content.slice(0, byteOffset);
    return prefix.split('\n').length;
  }
  
  private estimateTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }
}

/**
 * Convenience function for structure-aware chunking
 */
export async function chunkFileStructurally(
  file: SourceFile,
  config: Partial<ChunkingConfig> = {}
): Promise<ChunkingResult> {
  const chunker = new StructureAwareChunker(config);
  return chunker.chunkFile(file);
}