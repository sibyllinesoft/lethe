import type { Message, Chunk } from '@ctx-run/sqlite';
import type { 
  EnhancedChunk, 
  ChunkingMetadata, 
  Result, 
  LetheError 
} from './types.js';
import { v4 as uuidv4 } from 'uuid';

export interface ChunkingConfig {
  targetTokens: number;
  overlap: number;
  splitCodeBlocks: boolean;
  splitSentences: boolean;
  strategy: 'basic' | 'ast' | 'hierarchical' | 'propositional';
  astMaxDepth?: number;
  hierarchicalLevels?: number;
  propositionalRelations?: boolean;
  semanticBoundaryDetection?: boolean;
}

export class MessageChunker {
  private config: ChunkingConfig;
  private astParser: ASTChunker;
  private hierarchicalParser: HierarchicalChunker;
  private propositionalParser: PropositionalChunker;

  constructor(config: ChunkingConfig) {
    this.config = config;
    this.astParser = new ASTChunker(config);
    this.hierarchicalParser = new HierarchicalChunker(config);
    this.propositionalParser = new PropositionalChunker(config);
  }

  chunkMessage(message: Message): Result<EnhancedChunk[], LetheError> {
    try {
      switch (this.config.strategy) {
        case 'ast':
          return this.astChunking(message);
        case 'hierarchical':
          return this.hierarchicalChunking(message);
        case 'propositional':
          return this.propositionalChunking(message);
        case 'basic':
        default:
          return this.basicChunking(message);
      }
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'CHUNKING_ERROR',
          message: `Message chunking failed: ${error.message}`,
          timestamp: Date.now(),
          details: { 
            messageId: message.id,
            strategy: this.config.strategy,
            textLength: message.text.length 
          }
        }
      };
    }
  }

  /**
   * Basic chunking strategy (V5 baseline)
   * Enhanced version of existing implementation
   */
  private basicChunking(message: Message): Result<EnhancedChunk[], LetheError> {
    const chunks: EnhancedChunk[] = [];
    const text = message.text;
    
    // First, identify code blocks
    const codeBlocks = this.extractCodeBlocks(text);
    
    let currentOffset = 0;
    
    // Remove code blocks from text for prose processing
    for (const block of codeBlocks.sort((a, b) => a.start - b.start)) {
      // Process any prose text before this code block
      if (block.start > currentOffset) {
        const proseText = text.slice(currentOffset, block.start);
        const proseChunks = this.chunkProseText(proseText, currentOffset, message);
        chunks.push(...proseChunks);
      }
      
      // Add the code block as a chunk with enhanced metadata
      chunks.push({
        id: uuidv4(),
        messageId: message.id,
        offsetStart: block.start,
        offsetEnd: block.end,
        kind: this.determineCodeKind(block.content, message.role),
        text: block.content,
        tokens: this.estimateTokens(block.content),
        metadata: {
          semantic_boundary_score: 1.0, // Code blocks are natural boundaries
        },
        quality_score: this.calculateQualityScore(block.content, 'code')
      });
      
      currentOffset = block.end;
    }
    
    // Process any remaining prose text
    if (currentOffset < text.length) {
      const remainingText = text.slice(currentOffset);
      const remainingChunks = this.chunkProseText(remainingText, currentOffset, message);
      chunks.push(...remainingChunks);
    }
    
    return { success: true, data: chunks };
  }

  /**
   * AST-based chunking strategy (V5)
   * Parses code into semantic AST nodes for better precision
   */
  private astChunking(message: Message): Result<EnhancedChunk[], LetheError> {
    try {
      const chunks: EnhancedChunk[] = [];
      const text = message.text;
      
      // Identify code blocks first
      const codeBlocks = this.extractCodeBlocks(text);
      let currentOffset = 0;
      
      for (const block of codeBlocks.sort((a, b) => a.start - b.start)) {
        // Process prose before code block
        if (block.start > currentOffset) {
          const proseText = text.slice(currentOffset, block.start);
          const proseChunks = this.chunkProseText(proseText, currentOffset, message);
          chunks.push(...proseChunks);
        }
        
        // AST-based chunking for code blocks
        const language = this.detectLanguage(block.content);
        const astChunks = this.astParser.chunkCode(
          block.content, 
          language, 
          block.start, 
          message
        );
        
        chunks.push(...astChunks);
        currentOffset = block.end;
      }
      
      // Handle remaining prose
      if (currentOffset < text.length) {
        const remainingText = text.slice(currentOffset);
        const remainingChunks = this.chunkProseText(remainingText, currentOffset, message);
        chunks.push(...remainingChunks);
      }
      
      return { success: true, data: chunks };

    } catch (error) {
      // Fallback to basic chunking
      console.warn(`AST chunking failed, falling back to basic: ${error.message}`);
      return this.basicChunking(message);
    }
  }

  /**
   * Hierarchical chunking strategy (V5)
   * Creates nested semantic hierarchy with multiple levels
   */
  private hierarchicalChunking(message: Message): Result<EnhancedChunk[], LetheError> {
    try {
      const chunks: EnhancedChunk[] = [];
      const text = message.text;
      const levels = this.config.hierarchicalLevels || 3;
      
      // Create hierarchical structure
      const hierarchy = this.hierarchicalParser.buildHierarchy(text, levels);
      
      // Convert hierarchy to chunks
      const hierarchicalChunks = this.hierarchicalParser.flattenHierarchy(
        hierarchy,
        message,
        this.config.targetTokens
      );
      
      chunks.push(...hierarchicalChunks);
      
      return { success: true, data: chunks };

    } catch (error) {
      console.warn(`Hierarchical chunking failed, falling back to basic: ${error.message}`);
      return this.basicChunking(message);
    }
  }

  /**
   * Propositional chunking strategy (V5)
   * Chunks based on logical propositions and semantic relations
   */
  private propositionalChunking(message: Message): Result<EnhancedChunk[], LetheError> {
    try {
      const chunks: EnhancedChunk[] = [];
      const text = message.text;
      
      // Extract propositions and relations
      const propositions = this.propositionalParser.extractPropositions(text);
      const relations = this.propositionalParser.findRelations(propositions);
      
      // Create chunks based on propositional units
      const propositionalChunks = this.propositionalParser.createChunks(
        propositions,
        relations,
        message,
        this.config.targetTokens
      );
      
      chunks.push(...propositionalChunks);
      
      return { success: true, data: chunks };

    } catch (error) {
      console.warn(`Propositional chunking failed, falling back to basic: ${error.message}`);
      return this.basicChunking(message);
    }
  }

  // Enhanced helper methods
  private extractCodeBlocks(text: string): Array<{start: number, end: number, content: string}> {
    const codeBlocks: Array<{start: number, end: number, content: string}> = [];
    
    // Match fenced code blocks (```...```)
    const fencedRegex = /```[\s\S]*?```/g;
    let match;
    
    while ((match = fencedRegex.exec(text)) !== null) {
      codeBlocks.push({
        start: match.index,
        end: match.index + match[0].length,
        content: match[0]
      });
    }
    
    // Match inline code (`...`) if not already in a fenced block
    const inlineRegex = /`[^`\n]+`/g;
    while ((match = inlineRegex.exec(text)) !== null) {
      const start = match.index;
      const end = match.index + match[0].length;
      
      // Check if this inline code is already covered by a fenced block
      const covered = codeBlocks.some(block => 
        start >= block.start && end <= block.end
      );
      
      if (!covered && match[0].length > 10) { // Only chunk longer inline code
        codeBlocks.push({
          start,
          end,
          content: match[0]
        });
      }
    }
    
    return codeBlocks;
  }

  private determineCodeKind(content: string, role: string): 'code' | 'tool_result' | 'user_code' {
    if (role === 'tool') {
      return 'tool_result';
    }
    if (role === 'user') {
      return 'user_code';
    }
    return 'code';
  }

  private chunkProseText(text: string, baseOffset: number, message: Message): EnhancedChunk[] {
    const chunks: EnhancedChunk[] = [];
    
    if (!text.trim()) {
      return chunks;
    }
    
    // Enhanced semantic boundary detection
    const segments = this.config.semanticBoundaryDetection ? 
      this.detectSemanticBoundaries(text) : 
      (this.config.splitSentences ? this.splitBySentences(text) : [text]);
    
    let currentChunk = '';
    let chunkStart = baseOffset;
    let currentOffset = baseOffset;
    
    for (const segment of segments) {
      const segmentTokens = this.estimateTokens(segment);
      const currentTokens = this.estimateTokens(currentChunk);
      
      // If adding this segment would exceed target tokens, finalize current chunk
      if (currentTokens > 0 && currentTokens + segmentTokens > this.config.targetTokens) {
        const chunkText = currentChunk.trim();
        chunks.push({
          id: uuidv4(),
          messageId: message.id,
          offsetStart: chunkStart,
          offsetEnd: currentOffset,
          kind: 'prose',
          text: chunkText,
          tokens: currentTokens,
          metadata: {
            semantic_boundary_score: this.calculateSemanticBoundaryScore(chunkText)
          },
          quality_score: this.calculateQualityScore(chunkText, 'prose')
        });
        
        // Start new chunk with overlap
        const overlapText = this.getOverlapText(currentChunk, this.config.overlap);
        currentChunk = overlapText + segment;
        chunkStart = currentOffset - overlapText.length;
      } else {
        currentChunk += segment;
      }
      
      currentOffset += segment.length;
    }
    
    // Add final chunk if there's remaining text
    if (currentChunk.trim()) {
      const chunkText = currentChunk.trim();
      chunks.push({
        id: uuidv4(),
        messageId: message.id,
        offsetStart: chunkStart,
        offsetEnd: currentOffset,
        kind: 'prose',
        text: chunkText,
        tokens: this.estimateTokens(currentChunk),
        metadata: {
          semantic_boundary_score: this.calculateSemanticBoundaryScore(chunkText)
        },
        quality_score: this.calculateQualityScore(chunkText, 'prose')
      });
    }
    
    return chunks;
  }

  private detectLanguage(code: string): string {
    // Simple language detection based on patterns
    if (/import\s+.*from|export\s+.*{|const\s+.*=/i.test(code)) return 'typescript';
    if (/def\s+\w+\(|import\s+\w+|from\s+\w+\s+import/i.test(code)) return 'python';
    if (/function\s+\w+\(|var\s+\w+|let\s+\w+/i.test(code)) return 'javascript';
    if (/fn\s+\w+\(|use\s+\w+|mod\s+\w+/i.test(code)) return 'rust';
    if (/func\s+\w+\(|package\s+\w+|import\s+".*"/i.test(code)) return 'go';
    if (/<[^>]+>|class\s+\w+/i.test(code)) return 'html';
    if (/\{[^}]*\}/i.test(code)) return 'json';
    
    return 'text';
  }

  private detectSemanticBoundaries(text: string): string[] {
    // Enhanced semantic boundary detection
    const sentences = this.splitBySentences(text);
    const boundaries: string[] = [];
    let currentSegment = '';
    
    for (let i = 0; i < sentences.length; i++) {
      const sentence = sentences[i];
      const nextSentence = sentences[i + 1];
      
      currentSegment += sentence;
      
      // Check for semantic boundary indicators
      const hasBoundary = this.hasSemanticBoundary(sentence, nextSentence);
      
      if (hasBoundary || i === sentences.length - 1) {
        boundaries.push(currentSegment);
        currentSegment = '';
      }
    }
    
    return boundaries.filter(seg => seg.trim().length > 0);
  }

  private hasSemanticBoundary(current: string, next?: string): boolean {
    // Heuristics for semantic boundaries
    const boundaryIndicators = [
      /\b(however|therefore|furthermore|moreover|additionally)\b/i,
      /\b(in conclusion|to summarize|finally)\b/i,
      /\b(on the other hand|alternatively|instead)\b/i,
      /\b(for example|for instance|specifically)\b/i,
      /\b(first|second|third|next|then|finally)\b/i
    ];
    
    if (!next) return true;
    
    // Check for boundary indicators
    const hasIndicator = boundaryIndicators.some(pattern => 
      pattern.test(current) || pattern.test(next)
    );
    
    // Check for topic shifts (simplified)
    const currentWords = new Set(current.toLowerCase().split(/\s+/));
    const nextWords = new Set(next.toLowerCase().split(/\s+/));
    const overlap = [...currentWords].filter(w => nextWords.has(w)).length;
    const similarity = overlap / Math.max(currentWords.size, nextWords.size);
    
    return hasIndicator || similarity < 0.3;
  }

  private calculateSemanticBoundaryScore(text: string): number {
    // Calculate how well this chunk respects semantic boundaries
    const sentences = this.splitBySentences(text);
    if (sentences.length <= 1) return 1.0;
    
    let boundaryScore = 0;
    for (let i = 0; i < sentences.length - 1; i++) {
      if (this.hasSemanticBoundary(sentences[i], sentences[i + 1])) {
        boundaryScore += 1;
      }
    }
    
    return Math.min(1.0, boundaryScore / (sentences.length - 1));
  }

  private calculateQualityScore(text: string, type: 'prose' | 'code'): number {
    let score = 0.5; // Base score
    
    if (type === 'code') {
      // Code quality indicators
      if (/\/\/|#|\/\*/.test(text)) score += 0.2; // Has comments
      if (/function|class|def|fn/.test(text)) score += 0.2; // Has function/class definitions
      if (text.split('\n').length > 1) score += 0.1; // Multi-line
    } else {
      // Prose quality indicators
      const sentences = this.splitBySentences(text);
      if (sentences.length > 1) score += 0.1; // Multiple sentences
      if (/\b(because|since|therefore|thus|however)\b/i.test(text)) score += 0.2; // Logical connectors
      if (text.length > 100) score += 0.1; // Sufficient length
      if (text.split(' ').length / sentences.length >= 10) score += 0.1; // Good sentence length
    }
    
    return Math.min(1.0, score);
  }

  private splitBySentences(text: string): string[] {
    // Enhanced sentence splitting with better handling of edge cases
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    return sentences.map((s, idx) => {
      const trimmed = s.trim();
      if (idx < sentences.length - 1) {
        return trimmed + '.'; // Add back punctuation
      }
      return trimmed;
    });
  }

  private getOverlapText(text: string, overlapTokens: number): string {
    // Enhanced overlap calculation with better word boundary handling
    const words = text.trim().split(/\s+/);
    const overlapWords = Math.ceil(overlapTokens / 0.75);
    const startIndex = Math.max(0, words.length - overlapWords);
    
    // Ensure we don't break in the middle of sentences
    let overlapText = words.slice(startIndex).join(' ');
    
    // Try to end at a sentence boundary
    const lastPeriod = overlapText.lastIndexOf('.');
    if (lastPeriod > overlapText.length * 0.7) {
      overlapText = overlapText.substring(0, lastPeriod + 1);
    }
    
    return overlapText + ' ';
  }

  private estimateTokens(text: string): number {
    // Improved token estimation
    const words = text.trim().split(/\s+/).length;
    const specialChars = (text.match(/[^\w\s]/g) || []).length;
    const codePatterns = (text.match(/[{}()\[\];]/g) || []).length;
    
    return Math.ceil(words * 0.75 + specialChars * 0.25 + codePatterns * 0.1);
  }

  // Utility method to rechunk all messages in a session
  rechunkMessages(messages: Message[]): Result<EnhancedChunk[], LetheError> {
    try {
      const allChunks: EnhancedChunk[] = [];
      
      for (const message of messages) {
        const result = this.chunkMessage(message);
        if (!result.success) {
          return result;
        }
        allChunks.push(...result.data);
      }
      
      return { success: true, data: allChunks };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'RECHUNK_ERROR',
          message: `Failed to rechunk messages: ${error.message}`,
          timestamp: Date.now(),
          details: { messageCount: messages.length }
        }
      };
    }
  }

  // Configuration management
  updateConfig(newConfig: Partial<ChunkingConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    // Update strategy-specific parsers
    this.astParser.updateConfig(this.config);
    this.hierarchicalParser.updateConfig(this.config);
    this.propositionalParser.updateConfig(this.config);
  }

  getConfig(): ChunkingConfig {
    return { ...this.config };
  }
}

// ===== Specialized Chunking Strategy Classes =====

/**
 * AST-based chunking for code semantic structure
 */
class ASTChunker {
  private config: ChunkingConfig;

  constructor(config: ChunkingConfig) {
    this.config = config;
  }

  updateConfig(config: ChunkingConfig): void {
    this.config = config;
  }

  chunkCode(
    code: string, 
    language: string, 
    baseOffset: number, 
    message: Message
  ): EnhancedChunk[] {
    const chunks: EnhancedChunk[] = [];
    
    try {
      // Parse code into semantic units based on language
      const astNodes = this.parseCodeToAST(code, language);
      
      for (const node of astNodes) {
        const chunk: EnhancedChunk = {
          id: uuidv4(),
          messageId: message.id,
          offsetStart: baseOffset + node.start,
          offsetEnd: baseOffset + node.end,
          kind: this.determineCodeKind(node.content, message.role),
          text: node.content,
          tokens: this.estimateTokens(node.content),
          metadata: {
            ast_node_type: node.type,
            semantic_boundary_score: 1.0
          },
          quality_score: this.calculateCodeQuality(node)
        };
        
        chunks.push(chunk);
      }
      
      return chunks;

    } catch (error) {
      // Fallback to simple code block
      return [{
        id: uuidv4(),
        messageId: message.id,
        offsetStart: baseOffset,
        offsetEnd: baseOffset + code.length,
        kind: this.determineCodeKind(code, message.role),
        text: code,
        tokens: this.estimateTokens(code),
        metadata: {
          ast_node_type: 'fallback',
          semantic_boundary_score: 0.8
        },
        quality_score: 0.7
      }];
    }
  }

  private parseCodeToAST(code: string, language: string): Array<{
    type: string;
    content: string;
    start: number;
    end: number;
  }> {
    const nodes: Array<{ type: string; content: string; start: number; end: number; }> = [];
    
    // Simplified AST parsing based on language patterns
    switch (language) {
      case 'typescript':
      case 'javascript':
        return this.parseJavaScriptAST(code);
      case 'python':
        return this.parsePythonAST(code);
      case 'rust':
        return this.parseRustAST(code);
      case 'go':
        return this.parseGoAST(code);
      default:
        return this.parseGenericCodeAST(code);
    }
  }

  private parseJavaScriptAST(code: string): Array<{ type: string; content: string; start: number; end: number; }> {
    const nodes: Array<{ type: string; content: string; start: number; end: number; }> = [];
    
    // Function declarations
    const functionRegex = /(?:function\s+\w+|const\s+\w+\s*=\s*(?:\([^)]*\)\s*=>|\w+\s*=>)|\w+\([^)]*\)\s*{)/g;
    let match;
    
    while ((match = functionRegex.exec(code)) !== null) {
      const start = match.index;
      const end = this.findMatchingBrace(code, start);
      
      if (end > start) {
        nodes.push({
          type: 'function',
          content: code.slice(start, end),
          start,
          end
        });
      }
    }
    
    // Class declarations
    const classRegex = /class\s+\w+(?:\s+extends\s+\w+)?\s*{/g;
    while ((match = classRegex.exec(code)) !== null) {
      const start = match.index;
      const end = this.findMatchingBrace(code, start);
      
      if (end > start && !nodes.some(n => start >= n.start && end <= n.end)) {
        nodes.push({
          type: 'class',
          content: code.slice(start, end),
          start,
          end
        });
      }
    }
    
    // Interface declarations
    const interfaceRegex = /interface\s+\w+\s*{/g;
    while ((match = interfaceRegex.exec(code)) !== null) {
      const start = match.index;
      const end = this.findMatchingBrace(code, start);
      
      if (end > start && !nodes.some(n => start >= n.start && end <= n.end)) {
        nodes.push({
          type: 'interface',
          content: code.slice(start, end),
          start,
          end
        });
      }
    }
    
    return nodes.sort((a, b) => a.start - b.start);
  }

  private parsePythonAST(code: string): Array<{ type: string; content: string; start: number; end: number; }> {
    const nodes: Array<{ type: string; content: string; start: number; end: number; }> = [];
    const lines = code.split('\n');
    
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      const trimmed = line.trim();
      
      // Function definitions
      if (trimmed.startsWith('def ')) {
        const start = code.indexOf(line);
        const functionLines = this.getPythonBlock(lines, i);
        const end = start + functionLines.join('\n').length;
        
        nodes.push({
          type: 'function',
          content: functionLines.join('\n'),
          start,
          end
        });
        
        i += functionLines.length;
        continue;
      }
      
      // Class definitions
      if (trimmed.startsWith('class ')) {
        const start = code.indexOf(line);
        const classLines = this.getPythonBlock(lines, i);
        const end = start + classLines.join('\n').length;
        
        nodes.push({
          type: 'class',
          content: classLines.join('\n'),
          start,
          end
        });
        
        i += classLines.length;
        continue;
      }
      
      i++;
    }
    
    return nodes;
  }

  private parseRustAST(code: string): Array<{ type: string; content: string; start: number; end: number; }> {
    const nodes: Array<{ type: string; content: string; start: number; end: number; }> = [];
    
    // Function definitions
    const fnRegex = /fn\s+\w+[^{]*{/g;
    let match;
    
    while ((match = fnRegex.exec(code)) !== null) {
      const start = match.index;
      const end = this.findMatchingBrace(code, start);
      
      if (end > start) {
        nodes.push({
          type: 'function',
          content: code.slice(start, end),
          start,
          end
        });
      }
    }
    
    // Struct definitions
    const structRegex = /struct\s+\w+[^{]*{/g;
    while ((match = structRegex.exec(code)) !== null) {
      const start = match.index;
      const end = this.findMatchingBrace(code, start);
      
      if (end > start && !nodes.some(n => start >= n.start && end <= n.end)) {
        nodes.push({
          type: 'struct',
          content: code.slice(start, end),
          start,
          end
        });
      }
    }
    
    return nodes.sort((a, b) => a.start - b.start);
  }

  private parseGoAST(code: string): Array<{ type: string; content: string; start: number; end: number; }> {
    const nodes: Array<{ type: string; content: string; start: number; end: number; }> = [];
    
    // Function definitions
    const funcRegex = /func\s+(?:\([^)]*\)\s+)?\w+[^{]*{/g;
    let match;
    
    while ((match = funcRegex.exec(code)) !== null) {
      const start = match.index;
      const end = this.findMatchingBrace(code, start);
      
      if (end > start) {
        nodes.push({
          type: 'function',
          content: code.slice(start, end),
          start,
          end
        });
      }
    }
    
    // Type definitions
    const typeRegex = /type\s+\w+\s+(?:struct\s*{|interface\s*{)/g;
    while ((match = typeRegex.exec(code)) !== null) {
      const start = match.index;
      const end = this.findMatchingBrace(code, start);
      
      if (end > start && !nodes.some(n => start >= n.start && end <= n.end)) {
        nodes.push({
          type: 'type',
          content: code.slice(start, end),
          start,
          end
        });
      }
    }
    
    return nodes.sort((a, b) => a.start - b.start);
  }

  private parseGenericCodeAST(code: string): Array<{ type: string; content: string; start: number; end: number; }> {
    // Generic code parsing - split by logical blocks
    const blocks: Array<{ type: string; content: string; start: number; end: number; }> = [];
    const lines = code.split('\n');
    let currentBlock = '';
    let blockStart = 0;
    let currentOffset = 0;
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      currentBlock += line + '\n';
      
      // Simple heuristics for block boundaries
      const isBlockEnd = line.trim().endsWith('}') || 
                        line.trim().endsWith(';') && line.trim().length < 80 ||
                        line.trim() === '' && currentBlock.trim().length > 50;
      
      if (isBlockEnd || i === lines.length - 1) {
        if (currentBlock.trim().length > 0) {
          blocks.push({
            type: 'block',
            content: currentBlock.trim(),
            start: currentOffset + blockStart,
            end: currentOffset + blockStart + currentBlock.length
          });
        }
        
        currentBlock = '';
        blockStart = currentOffset + line.length + 1;
      }
      
      currentOffset += line.length + 1;
    }
    
    return blocks;
  }

  private findMatchingBrace(code: string, start: number): number {
    let braceCount = 0;
    let inString = false;
    let stringChar = '';
    
    for (let i = start; i < code.length; i++) {
      const char = code[i];
      const prevChar = i > 0 ? code[i - 1] : '';
      
      // Handle string literals
      if ((char === '"' || char === "'") && prevChar !== '\\') {
        if (!inString) {
          inString = true;
          stringChar = char;
        } else if (char === stringChar) {
          inString = false;
          stringChar = '';
        }
        continue;
      }
      
      if (inString) continue;
      
      if (char === '{') {
        braceCount++;
      } else if (char === '}') {
        braceCount--;
        if (braceCount === 0) {
          return i + 1;
        }
      }
    }
    
    return code.length; // No matching brace found
  }

  private getPythonBlock(lines: string[], startIdx: number): string[] {
    const block = [lines[startIdx]];
    const baseIndent = this.getIndentLevel(lines[startIdx]);
    
    for (let i = startIdx + 1; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();
      
      // Empty lines are part of the block
      if (trimmed === '') {
        block.push(line);
        continue;
      }
      
      // Check indentation to determine if still in block
      const indent = this.getIndentLevel(line);
      if (indent > baseIndent) {
        block.push(line);
      } else {
        break;
      }
    }
    
    return block;
  }

  private getIndentLevel(line: string): number {
    const match = line.match(/^(\s*)/);
    return match ? match[1].length : 0;
  }

  private determineCodeKind(content: string, role: string): 'code' | 'tool_result' | 'user_code' {
    if (role === 'tool') return 'tool_result';
    if (role === 'user') return 'user_code';
    return 'code';
  }

  private calculateCodeQuality(node: { type: string; content: string }): number {
    let score = 0.6; // Base score for AST nodes
    
    const content = node.content;
    
    // Type-specific quality indicators
    if (node.type === 'function') {
      if (/\/\/|#|\/\*/.test(content)) score += 0.2; // Has comments
      if (content.split('\n').length > 3) score += 0.1; // Multi-line
      if (/return\s+/.test(content)) score += 0.1; // Has return statement
    } else if (node.type === 'class') {
      if (/constructor|__init__/.test(content)) score += 0.2; // Has constructor
      if ((content.match(/function|def/g) || []).length > 1) score += 0.1; // Multiple methods
    }
    
    // General quality indicators
    if (content.length > 100) score += 0.1; // Substantial content
    
    return Math.min(1.0, score);
  }

  private estimateTokens(text: string): number {
    const words = text.trim().split(/\s+/).length;
    const specialChars = (text.match(/[^\w\s]/g) || []).length;
    const codePatterns = (text.match(/[{}()\[\];]/g) || []).length;
    
    return Math.ceil(words * 0.75 + specialChars * 0.25 + codePatterns * 0.1);
  }
}

/**
 * Hierarchical chunking with multiple semantic levels
 */
class HierarchicalChunker {
  private config: ChunkingConfig;

  constructor(config: ChunkingConfig) {
    this.config = config;
  }

  updateConfig(config: ChunkingConfig): void {
    this.config = config;
  }

  buildHierarchy(text: string, levels: number): HierarchicalNode {
    // Create root node
    const root: HierarchicalNode = {
      id: uuidv4(),
      level: 0,
      content: text,
      start: 0,
      end: text.length,
      children: [],
      type: 'document'
    };
    
    // Recursively build hierarchy
    this.buildLevel(root, 1, levels);
    
    return root;
  }

  flattenHierarchy(
    root: HierarchicalNode, 
    message: Message, 
    targetTokens: number
  ): EnhancedChunk[] {
    const chunks: EnhancedChunk[] = [];
    
    // Depth-first traversal to create chunks
    this.traverseAndChunk(root, message, targetTokens, chunks);
    
    return chunks;
  }

  private buildLevel(node: HierarchicalNode, currentLevel: number, maxLevels: number): void {
    if (currentLevel >= maxLevels) return;
    
    const content = node.content;
    const segments = this.segmentByLevel(content, currentLevel);
    
    let offset = node.start;
    
    for (const segment of segments) {
      const child: HierarchicalNode = {
        id: uuidv4(),
        level: currentLevel,
        content: segment,
        start: offset,
        end: offset + segment.length,
        children: [],
        type: this.determineNodeType(segment, currentLevel)
      };
      
      node.children.push(child);
      
      // Recursively build next level
      this.buildLevel(child, currentLevel + 1, maxLevels);
      
      offset += segment.length;
    }
  }

  private segmentByLevel(text: string, level: number): string[] {
    switch (level) {
      case 1: // Paragraph level
        return this.segmentByParagraphs(text);
      case 2: // Sentence level
        return this.segmentBySentences(text);
      case 3: // Phrase level
        return this.segmentByPhrases(text);
      default:
        return [text];
    }
  }

  private segmentByParagraphs(text: string): string[] {
    return text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
  }

  private segmentBySentences(text: string): string[] {
    return text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  }

  private segmentByPhrases(text: string): string[] {
    // Split by commas and conjunctions for phrases
    return text.split(/[,;]|\b(?:and|or|but|however|therefore)\b/)
               .filter(p => p.trim().length > 10);
  }

  private determineNodeType(content: string, level: number): string {
    if (level === 1) {
      if (/```/.test(content)) return 'code_paragraph';
      if (/^\s*[-*+]\s/.test(content)) return 'list_paragraph';
      return 'text_paragraph';
    } else if (level === 2) {
      if (/\?$/.test(content.trim())) return 'question';
      if (/!$/.test(content.trim())) return 'exclamation';
      return 'statement';
    } else {
      return 'phrase';
    }
  }

  private traverseAndChunk(
    node: HierarchicalNode, 
    message: Message, 
    targetTokens: number, 
    chunks: EnhancedChunk[]
  ): void {
    const tokens = this.estimateTokens(node.content);
    
    // If node is small enough, create a chunk
    if (tokens <= targetTokens) {
      chunks.push({
        id: uuidv4(),
        messageId: message.id,
        offsetStart: node.start,
        offsetEnd: node.end,
        kind: 'prose',
        text: node.content.trim(),
        tokens,
        metadata: {
          hierarchical_level: node.level,
          semantic_boundary_score: this.calculateHierarchicalScore(node)
        },
        quality_score: this.calculateHierarchicalQuality(node)
      });
    } else {
      // Recurse to children
      for (const child of node.children) {
        this.traverseAndChunk(child, message, targetTokens, chunks);
      }
    }
  }

  private calculateHierarchicalScore(node: HierarchicalNode): number {
    // Higher level nodes have higher boundary scores
    const levelScore = (4 - node.level) / 4;
    
    // Type-based adjustments
    let typeAdjustment = 0;
    if (node.type.includes('paragraph')) typeAdjustment = 0.2;
    if (node.type === 'question') typeAdjustment = 0.1;
    
    return Math.min(1.0, levelScore + typeAdjustment);
  }

  private calculateHierarchicalQuality(node: HierarchicalNode): number {
    let score = 0.5 + (node.level * 0.1); // Base score increases with depth
    
    const content = node.content;
    
    // Content quality indicators
    if (content.length > 50) score += 0.1;
    if (/\b(because|since|therefore|thus|however)\b/i.test(content)) score += 0.2;
    if (node.children.length > 0) score += 0.1; // Has structure
    
    return Math.min(1.0, score);
  }

  private estimateTokens(text: string): number {
    const words = text.trim().split(/\s+/).length;
    return Math.ceil(words * 0.75);
  }
}

interface HierarchicalNode {
  id: string;
  level: number;
  content: string;
  start: number;
  end: number;
  children: HierarchicalNode[];
  type: string;
}

/**
 * Propositional chunking based on logical relations
 */
class PropositionalChunker {
  private config: ChunkingConfig;

  constructor(config: ChunkingConfig) {
    this.config = config;
  }

  updateConfig(config: ChunkingConfig): void {
    this.config = config;
  }

  extractPropositions(text: string): Proposition[] {
    const propositions: Proposition[] = [];
    const sentences = this.splitIntoSentences(text);
    
    let offset = 0;
    
    for (const sentence of sentences) {
      const props = this.extractPropositionsFromSentence(sentence, offset);
      propositions.push(...props);
      offset += sentence.length + 1; // +1 for delimiter
    }
    
    return propositions;
  }

  findRelations(propositions: Proposition[]): PropositionalRelation[] {
    const relations: PropositionalRelation[] = [];
    
    for (let i = 0; i < propositions.length - 1; i++) {
      for (let j = i + 1; j < Math.min(propositions.length, i + 4); j++) { // Look ahead max 3
        const prop1 = propositions[i];
        const prop2 = propositions[j];
        
        const relation = this.detectRelation(prop1, prop2);
        if (relation) {
          relations.push({
            id: uuidv4(),
            source: prop1.id,
            target: prop2.id,
            type: relation,
            strength: this.calculateRelationStrength(prop1, prop2, relation)
          });
        }
      }
    }
    
    return relations;
  }

  createChunks(
    propositions: Proposition[], 
    relations: PropositionalRelation[], 
    message: Message, 
    targetTokens: number
  ): EnhancedChunk[] {
    const chunks: EnhancedChunk[] = [];
    const relationMap = this.buildRelationMap(relations);
    const visited = new Set<string>();
    
    for (const prop of propositions) {
      if (visited.has(prop.id)) continue;
      
      // Build connected component
      const component = this.buildConnectedComponent(prop, relationMap, propositions);
      const componentText = component.map(p => p.content).join(' ');
      const tokens = this.estimateTokens(componentText);
      
      // If component fits in target tokens, create chunk
      if (tokens <= targetTokens) {
        const startOffset = Math.min(...component.map(p => p.start));
        const endOffset = Math.max(...component.map(p => p.end));
        
        chunks.push({
          id: uuidv4(),
          messageId: message.id,
          offsetStart: startOffset,
          offsetEnd: endOffset,
          kind: 'prose',
          text: componentText.trim(),
          tokens,
          metadata: {
            propositional_relations: relations
              .filter(r => component.some(p => p.id === r.source || p.id === r.target))
              .map(r => r.type),
            semantic_boundary_score: this.calculatePropositionalScore(component, relations)
          },
          quality_score: this.calculatePropositionalQuality(component, relations)
        });
        
        // Mark all propositions as visited
        component.forEach(p => visited.add(p.id));
      } else {
        // Split large component
        const subChunks = this.splitLargeComponent(component, targetTokens, message, relations);
        chunks.push(...subChunks);
        component.forEach(p => visited.add(p.id));
      }
    }
    
    return chunks;
  }

  private splitIntoSentences(text: string): string[] {
    return text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  }

  private extractPropositionsFromSentence(sentence: string, baseOffset: number): Proposition[] {
    const propositions: Proposition[] = [];
    
    // Simple proposition extraction - split by conjunctions
    const clauses = sentence.split(/\b(?:and|or|but|because|since|although|while|if|unless)\b/i);
    
    let offset = baseOffset;
    
    for (const clause of clauses) {
      const trimmed = clause.trim();
      if (trimmed.length > 10) { // Minimum meaningful length
        propositions.push({
          id: uuidv4(),
          content: trimmed,
          start: offset,
          end: offset + clause.length,
          type: this.classifyProposition(trimmed)
        });
      }
      offset += clause.length;
    }
    
    return propositions;
  }

  private classifyProposition(content: string): string {
    const lower = content.toLowerCase();
    
    if (/\b(is|are|was|were)\b/.test(lower)) return 'state';
    if (/\b(will|shall|going to)\b/.test(lower)) return 'future';
    if (/\b(can|could|should|must|may)\b/.test(lower)) return 'modal';
    if (lower.includes('?')) return 'question';
    if (/\b(do|does|did|run|execute|perform)\b/.test(lower)) return 'action';
    
    return 'statement';
  }

  private detectRelation(prop1: Proposition, prop2: Proposition): string | null {
    const text1 = prop1.content.toLowerCase();
    const text2 = prop2.content.toLowerCase();
    
    // Causal relations
    if (/because|since|due to/.test(text2)) return 'cause';
    if (/therefore|thus|consequently/.test(text2)) return 'effect';
    
    // Temporal relations
    if (/then|next|after|before/.test(text2)) return 'temporal';
    
    // Contrast relations
    if (/however|but|although|despite/.test(text2)) return 'contrast';
    
    // Addition relations
    if (/also|additionally|furthermore|moreover/.test(text2)) return 'addition';
    
    // Semantic similarity (simple word overlap)
    const words1 = new Set(text1.split(/\s+/));
    const words2 = new Set(text2.split(/\s+/));
    const overlap = [...words1].filter(w => words2.has(w)).length;
    const similarity = overlap / Math.max(words1.size, words2.size);
    
    if (similarity > 0.3) return 'similar';
    
    return null;
  }

  private calculateRelationStrength(prop1: Proposition, prop2: Proposition, relationType: string): number {
    let strength = 0.5; // Base strength
    
    // Type-based adjustments
    switch (relationType) {
      case 'cause':
      case 'effect':
        strength = 0.9;
        break;
      case 'temporal':
        strength = 0.7;
        break;
      case 'contrast':
        strength = 0.6;
        break;
      case 'similar':
        strength = 0.4;
        break;
      default:
        strength = 0.3;
    }
    
    // Distance penalty
    const distance = Math.abs(prop1.start - prop2.start);
    const distancePenalty = Math.min(0.3, distance / 1000);
    
    return Math.max(0.1, strength - distancePenalty);
  }

  private buildRelationMap(relations: PropositionalRelation[]): Map<string, PropositionalRelation[]> {
    const map = new Map<string, PropositionalRelation[]>();
    
    for (const relation of relations) {
      if (!map.has(relation.source)) {
        map.set(relation.source, []);
      }
      if (!map.has(relation.target)) {
        map.set(relation.target, []);
      }
      
      map.get(relation.source)!.push(relation);
      map.get(relation.target)!.push(relation);
    }
    
    return map;
  }

  private buildConnectedComponent(
    startProp: Proposition, 
    relationMap: Map<string, PropositionalRelation[]>, 
    allPropositions: Proposition[]
  ): Proposition[] {
    const component: Proposition[] = [];
    const visited = new Set<string>();
    const queue = [startProp];
    const propMap = new Map(allPropositions.map(p => [p.id, p]));
    
    while (queue.length > 0) {
      const current = queue.shift()!;
      if (visited.has(current.id)) continue;
      
      visited.add(current.id);
      component.push(current);
      
      // Find connected propositions
      const relations = relationMap.get(current.id) || [];
      for (const relation of relations) {
        if (relation.strength > 0.3) { // Only strong relations
          const nextId = relation.source === current.id ? relation.target : relation.source;
          const nextProp = propMap.get(nextId);
          
          if (nextProp && !visited.has(nextId)) {
            queue.push(nextProp);
          }
        }
      }
    }
    
    return component.sort((a, b) => a.start - b.start);
  }

  private splitLargeComponent(
    component: Proposition[], 
    targetTokens: number, 
    message: Message, 
    relations: PropositionalRelation[]
  ): EnhancedChunk[] {
    const chunks: EnhancedChunk[] = [];
    let currentChunk: Proposition[] = [];
    let currentTokens = 0;
    
    for (const prop of component) {
      const propTokens = this.estimateTokens(prop.content);
      
      if (currentTokens + propTokens > targetTokens && currentChunk.length > 0) {
        // Finalize current chunk
        const chunkText = currentChunk.map(p => p.content).join(' ');
        const startOffset = Math.min(...currentChunk.map(p => p.start));
        const endOffset = Math.max(...currentChunk.map(p => p.end));
        
        chunks.push({
          id: uuidv4(),
          messageId: message.id,
          offsetStart: startOffset,
          offsetEnd: endOffset,
          kind: 'prose',
          text: chunkText.trim(),
          tokens: currentTokens,
          metadata: {
            propositional_relations: relations
              .filter(r => currentChunk.some(p => p.id === r.source || p.id === r.target))
              .map(r => r.type),
            semantic_boundary_score: this.calculatePropositionalScore(currentChunk, relations)
          },
          quality_score: this.calculatePropositionalQuality(currentChunk, relations)
        });
        
        currentChunk = [];
        currentTokens = 0;
      }
      
      currentChunk.push(prop);
      currentTokens += propTokens;
    }
    
    // Add remaining chunk
    if (currentChunk.length > 0) {
      const chunkText = currentChunk.map(p => p.content).join(' ');
      const startOffset = Math.min(...currentChunk.map(p => p.start));
      const endOffset = Math.max(...currentChunk.map(p => p.end));
      
      chunks.push({
        id: uuidv4(),
        messageId: message.id,
        offsetStart: startOffset,
        offsetEnd: endOffset,
        kind: 'prose',
        text: chunkText.trim(),
        tokens: currentTokens,
        metadata: {
          propositional_relations: relations
            .filter(r => currentChunk.some(p => p.id === r.source || p.id === r.target))
            .map(r => r.type),
          semantic_boundary_score: this.calculatePropositionalScore(currentChunk, relations)
        },
        quality_score: this.calculatePropositionalQuality(currentChunk, relations)
      });
    }
    
    return chunks;
  }

  private calculatePropositionalScore(
    propositions: Proposition[], 
    relations: PropositionalRelation[]
  ): number {
    if (propositions.length === 0) return 0;
    
    // Score based on relation density
    const relevantRelations = relations.filter(r => 
      propositions.some(p => p.id === r.source || p.id === r.target)
    );
    
    const relationDensity = relevantRelations.length / Math.max(1, propositions.length);
    const avgStrength = relevantRelations.length > 0 ? 
      relevantRelations.reduce((sum, r) => sum + r.strength, 0) / relevantRelations.length : 0.5;
    
    return Math.min(1.0, relationDensity * avgStrength + 0.3);
  }

  private calculatePropositionalQuality(
    propositions: Proposition[], 
    relations: PropositionalRelation[]
  ): number {
    if (propositions.length === 0) return 0;
    
    let score = 0.5; // Base score
    
    // Relation variety bonus
    const relationTypes = new Set(relations.map(r => r.type));
    score += relationTypes.size * 0.1;
    
    // Proposition type variety bonus
    const propTypes = new Set(propositions.map(p => p.type));
    score += propTypes.size * 0.1;
    
    // Coherence bonus (multiple propositions with relations)
    if (propositions.length > 1 && relations.length > 0) {
      score += 0.2;
    }
    
    return Math.min(1.0, score);
  }

  private estimateTokens(text: string): number {
    const words = text.trim().split(/\s+/).length;
    return Math.ceil(words * 0.75);
  }
}

interface Proposition {
  id: string;
  content: string;
  start: number;
  end: number;
  type: string;
}

interface PropositionalRelation {
  id: string;
  source: string;
  target: string;
  type: string;
  strength: number;
}
