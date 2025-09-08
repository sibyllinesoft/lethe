import { Message, Chunk } from '@lethe/sqlite';
import { createHash } from 'crypto';

// Enhanced BPE-style token counter with better accuracy
function countTokens(text: string): number {
  // More accurate GPT tokenization approximation
  // Based on analysis of typical tokenization patterns
  
  if (!text.length) return 0;
  
  // Count different types of tokens
  let tokenCount = 0;
  
  // Handle whitespace - each sequence of whitespace is typically 1 token
  const whitespaceMatches = text.match(/\s+/g);
  const whitespaceTokens = whitespaceMatches ? whitespaceMatches.length : 0;
  
  // Remove whitespace for word analysis
  const nonWhitespaceText = text.replace(/\s+/g, '');
  
  if (!nonWhitespaceText) return whitespaceTokens;
  
  // Split into alphanumeric and non-alphanumeric segments
  const segments = nonWhitespaceText.split(/([a-zA-Z0-9]+)/);
  
  for (const segment of segments) {
    if (!segment) continue;
    
    if (/^[a-zA-Z0-9]+$/.test(segment)) {
      // Alphanumeric words
      if (segment.length <= 4) {
        tokenCount += 1; // Short words are typically 1 token
      } else if (segment.length <= 8) {
        tokenCount += Math.ceil(segment.length / 4); // Medium words ~4 chars per token
      } else {
        tokenCount += Math.ceil(segment.length / 3); // Long words ~3 chars per token
      }
    } else {
      // Special characters and punctuation
      // Each character or small group is typically 1 token
      const specialChars = segment.length;
      tokenCount += Math.max(1, Math.ceil(specialChars / 2));
    }
  }
  
  // Add whitespace tokens
  tokenCount += whitespaceTokens;
  
  // Apply realistic bounds - minimum 1 token for non-empty text
  return Math.max(1, tokenCount);
}

// Unicode-aware sentence splitter with fallback to word splitting
function splitSentences(text: string): string[] {
  // Simple regex-based sentence splitting
  const sentences = text.split(/(?<=[.!?])\s+/);
  const filtered = sentences.filter(s => s.trim().length > 0);
  
  // If no sentences found (no punctuation), fall back to word splitting
  if (filtered.length === 1 && !text.match(/[.!?]/)) {
    const words = text.split(/\s+/);
    return words.filter(w => w.trim().length > 0);
  }
  
  return filtered;
}

// Detect code fences
function extractCodeFences(text: string): { type: 'code' | 'text'; content: string; start: number; end: number }[] {
  const parts: { type: 'code' | 'text'; content: string; start: number; end: number }[] = [];
  const codePattern = /```[\s\S]*?```/g;
  let lastIndex = 0;
  let match;

  while ((match = codePattern.exec(text)) !== null) {
    // Add text before code block
    if (match.index > lastIndex) {
      const textContent = text.slice(lastIndex, match.index);
      if (textContent.trim()) {
        parts.push({
          type: 'text',
          content: textContent,
          start: lastIndex,
          end: match.index
        });
      }
    }

    // Add code block
    parts.push({
      type: 'code',
      content: match[0],
      start: match.index,
      end: match.index + match[0].length
    });

    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    const textContent = text.slice(lastIndex);
    if (textContent.trim()) {
      parts.push({
        type: 'text',
        content: textContent,
        start: lastIndex,
        end: text.length
      });
    }
  }

  return parts;
}

function createChunks(
  messageId: string,
  sessionId: string,
  parts: { type: 'code' | 'text'; content: string; start: number; end: number }[],
  targetTokens: number = 320,
  overlap: number = 64
): Chunk[] {
  const chunks: Chunk[] = [];

  for (const part of parts) {
    const tokens = countTokens(part.content);

    if (tokens <= targetTokens) {
      // Part fits in one chunk
      const chunkId = createHash('sha256')
        .update(`${messageId}-${part.start}-${part.end}`)
        .digest('hex')
        .slice(0, 16);

      chunks.push({
        id: chunkId,
        message_id: messageId,
        session_id: sessionId,
        offset_start: part.start,
        offset_end: part.end,
        kind: part.type,
        text: part.content,
        tokens
      });
    } else {
      // Need to split the part
      if (part.type === 'text') {
        // Split by sentences
        const sentences = splitSentences(part.content);
        let currentChunk = '';
        let currentStart = part.start;
        let currentTokens = 0;

        for (const sentence of sentences) {
          const sentenceTokens = countTokens(sentence);
          
          if (currentTokens + sentenceTokens > targetTokens && currentChunk) {
            // Create chunk
            const chunkEnd = currentStart + currentChunk.length;
            const chunkId = createHash('sha256')
              .update(`${messageId}-${currentStart}-${chunkEnd}`)
              .digest('hex')
              .slice(0, 16);

            chunks.push({
              id: chunkId,
              message_id: messageId,
              session_id: sessionId,
              offset_start: currentStart,
              offset_end: chunkEnd,
              kind: part.type,
              text: currentChunk.trim(),
              tokens: currentTokens
            });

            // Start new chunk with overlap
            const overlapText = currentChunk.slice(-overlap);
            currentChunk = overlapText + ' ' + sentence;
            currentStart = chunkEnd - overlapText.length;
            currentTokens = countTokens(currentChunk);
          } else {
            currentChunk += (currentChunk ? ' ' : '') + sentence;
            currentTokens = currentTokens + sentenceTokens;
          }
        }

        // Add final chunk
        if (currentChunk.trim()) {
          const chunkEnd = currentStart + currentChunk.length;
          const chunkId = createHash('sha256')
            .update(`${messageId}-${currentStart}-${chunkEnd}`)
            .digest('hex')
            .slice(0, 16);

          chunks.push({
            id: chunkId,
            message_id: messageId,
            session_id: sessionId,
            offset_start: currentStart,
            offset_end: chunkEnd,
            kind: part.type,
            text: currentChunk.trim(),
            tokens: currentTokens
          });
        }
      } else {
        // Code block - split by lines if needed
        const lines = part.content.split('\n');
        let currentChunk = '';
        let currentStart = part.start;
        let currentTokens = 0;
        let lineOffset = 0;

        for (let i = 0; i < lines.length; i++) {
          const line = lines[i] + (i < lines.length - 1 ? '\n' : '');
          const lineTokens = countTokens(line);
          
          if (currentTokens + lineTokens > targetTokens && currentChunk) {
            // Create chunk
            const chunkEnd = currentStart + currentChunk.length;
            const chunkId = createHash('sha256')
              .update(`${messageId}-${currentStart}-${chunkEnd}`)
              .digest('hex')
              .slice(0, 16);

            chunks.push({
              id: chunkId,
              message_id: messageId,
              session_id: sessionId,
              offset_start: currentStart,
              offset_end: chunkEnd,
              kind: part.type,
              text: currentChunk,
              tokens: currentTokens
            });

            // Start new chunk with overlap
            const overlapLines = Math.min(3, Math.floor(overlap / 20)); // Rough estimate
            const overlapText = lines.slice(Math.max(0, i - overlapLines), i).join('\n');
            currentChunk = overlapText + (overlapText ? '\n' : '') + line;
            currentStart = part.start + lineOffset - overlapText.length;
            currentTokens = countTokens(currentChunk);
          } else {
            currentChunk += line;
            currentTokens += lineTokens;
          }
          lineOffset += line.length;
        }

        // Add final chunk
        if (currentChunk.trim()) {
          const chunkEnd = currentStart + currentChunk.length;
          const chunkId = createHash('sha256')
            .update(`${messageId}-${currentStart}-${chunkEnd}`)
            .digest('hex')
            .slice(0, 16);

          chunks.push({
            id: chunkId,
            message_id: messageId,
            session_id: sessionId,
            offset_start: currentStart,
            offset_end: chunkEnd,
            kind: part.type,
            text: currentChunk,
            tokens: currentTokens
          });
        }
      }
    }
  }

  return chunks;
}

export function chunkMessage(msg: Message): Chunk[] {
  // Normalize text to NFC
  const normalizedText = msg.text.normalize('NFC');
  
  // Extract code fences and text parts
  const parts = extractCodeFences(normalizedText);
  
  // If no parts found, treat as single text part
  if (parts.length === 0) {
    parts.push({
      type: 'text',
      content: normalizedText,
      start: 0,
      end: normalizedText.length
    });
  }
  
  // Create chunks from parts
  return createChunks(msg.id, msg.session_id, parts);
}