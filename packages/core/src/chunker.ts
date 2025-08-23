import type { Message, Chunk } from '@ctx-run/sqlite';
import { v4 as uuidv4 } from 'uuid';

export interface ChunkingConfig {
  targetTokens: number;
  overlap: number;
  splitCodeBlocks: boolean;
  splitSentences: boolean;
}

export class MessageChunker {
  private config: ChunkingConfig;

  constructor(config: ChunkingConfig) {
    this.config = config;
  }

  chunkMessage(message: Message): Chunk[] {
    const chunks: Chunk[] = [];
    const text = message.text;
    
    // First, identify code blocks
    const codeBlocks = this.extractCodeBlocks(text);
    
    let currentOffset = 0;
    let nonCodeText = text;
    
    // Remove code blocks from text for prose processing
    for (const block of codeBlocks.sort((a, b) => a.start - b.start)) {
      // Process any prose text before this code block
      if (block.start > currentOffset) {
        const proseText = text.slice(currentOffset, block.start);
        const proseChunks = this.chunkProseText(proseText, currentOffset, message);
        chunks.push(...proseChunks);
      }
      
      // Add the code block as a chunk
      chunks.push({
        id: uuidv4(),
        messageId: message.id,
        offsetStart: block.start,
        offsetEnd: block.end,
        kind: this.determineCodeKind(block.content, message.role),
        text: block.content,
        tokens: this.estimateTokens(block.content)
      });
      
      currentOffset = block.end;
    }
    
    // Process any remaining prose text
    if (currentOffset < text.length) {
      const remainingText = text.slice(currentOffset);
      const remainingChunks = this.chunkProseText(remainingText, currentOffset, message);
      chunks.push(...remainingChunks);
    }
    
    return chunks;
  }

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

  private chunkProseText(text: string, baseOffset: number, message: Message): Chunk[] {
    const chunks: Chunk[] = [];
    
    if (!text.trim()) {
      return chunks;
    }
    
    // Split by sentences if enabled
    const segments = this.config.splitSentences ? 
      this.splitBySentences(text) : [text];
    
    let currentChunk = '';
    let chunkStart = baseOffset;
    let currentOffset = baseOffset;
    
    for (const segment of segments) {
      const segmentTokens = this.estimateTokens(segment);
      const currentTokens = this.estimateTokens(currentChunk);
      
      // If adding this segment would exceed target tokens, finalize current chunk
      if (currentTokens > 0 && currentTokens + segmentTokens > this.config.targetTokens) {
        chunks.push({
          id: uuidv4(),
          messageId: message.id,
          offsetStart: chunkStart,
          offsetEnd: currentOffset,
          kind: 'prose',
          text: currentChunk.trim(),
          tokens: currentTokens
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
      chunks.push({
        id: uuidv4(),
        messageId: message.id,
        offsetStart: chunkStart,
        offsetEnd: currentOffset,
        kind: 'prose',
        text: currentChunk.trim(),
        tokens: this.estimateTokens(currentChunk)
      });
    }
    
    return chunks;
  }

  private splitBySentences(text: string): string[] {
    // Simple sentence splitting - can be enhanced with better NLP
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    return sentences.map(s => s.trim() + '.');
  }

  private getOverlapText(text: string, overlapTokens: number): string {
    // Simple word-based overlap - estimate tokens as words * 0.75
    const words = text.trim().split(/\s+/);
    const overlapWords = Math.ceil(overlapTokens / 0.75);
    const startIndex = Math.max(0, words.length - overlapWords);
    return words.slice(startIndex).join(' ') + ' ';
  }

  private estimateTokens(text: string): number {
    // Simple token estimation: words * 0.75 + special chars * 0.25
    const words = text.trim().split(/\s+/).length;
    const specialChars = (text.match(/[^\w\s]/g) || []).length;
    return Math.ceil(words * 0.75 + specialChars * 0.25);
  }

  // Utility method to rechunk all messages in a session
  rechunkMessages(messages: Message[]): Chunk[] {
    const allChunks: Chunk[] = [];
    
    for (const message of messages) {
      const chunks = this.chunkMessage(message);
      allChunks.push(...chunks);
    }
    
    return allChunks;
  }
}