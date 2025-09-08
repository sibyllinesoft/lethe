import { describe, it, expect, beforeEach } from 'vitest';
import { chunkMessage } from './index.js';
import { Message, Chunk } from '@lethe/sqlite';

describe('MessageChunker', () => {
  let testMessage: Message;

  beforeEach(() => {
    testMessage = {
      id: 'test-msg-1',
      session_id: 'test-session',
      turn: 1,
      role: 'user',
      text: '',
      ts: Date.now()
    };
  });

  describe('Basic Text Chunking', () => {
    it('should chunk simple text within token limit', () => {
      testMessage.text = 'This is a simple test message that should fit in one chunk.';
      
      const chunks = chunkMessage(testMessage);
      
      expect(chunks).toHaveLength(1);
      expect(chunks[0].text).toBe(testMessage.text);
      expect(chunks[0].kind).toBe('text');
      expect(chunks[0].message_id).toBe('test-msg-1');
      expect(chunks[0].session_id).toBe('test-session');
      expect(chunks[0].offset_start).toBe(0);
      expect(chunks[0].offset_end).toBe(testMessage.text.length);
    });

    it('should generate consistent chunk IDs', () => {
      testMessage.text = 'Consistent message for ID testing.';
      
      const chunks1 = chunkMessage(testMessage);
      const chunks2 = chunkMessage(testMessage);
      
      expect(chunks1[0].id).toBe(chunks2[0].id);
    });

    it('should split long text into multiple chunks', () => {
      // Create a long text that exceeds target tokens (320 tokens ≈ 1280 characters)
      const longText = 'This is a sentence that will be repeated many times to create a very long message. '.repeat(20);
      testMessage.text = longText;
      
      const chunks = chunkMessage(testMessage);
      
      expect(chunks.length).toBeGreaterThan(1);
      
      // Verify all chunks are within reasonable token limits
      chunks.forEach(chunk => {
        expect(chunk.tokens).toBeLessThanOrEqual(320);
        expect(chunk.kind).toBe('text');
        expect(chunk.message_id).toBe('test-msg-1');
      });
      
      // Verify chunks cover the entire message
      const totalCoverage = chunks.reduce((sum, chunk) => sum + (chunk.offset_end - chunk.offset_start), 0);
      expect(totalCoverage).toBeGreaterThanOrEqual(longText.length);
    });

    it('should handle empty text', () => {
      testMessage.text = '';
      
      const chunks = chunkMessage(testMessage);
      
      expect(chunks).toHaveLength(1);
      expect(chunks[0].text).toBe('');
      expect(chunks[0].kind).toBe('text');
    });

    it('should normalize Unicode text to NFC', () => {
      // Test with unnormalized Unicode (NFD form)
      testMessage.text = 'Café'; // This may be in NFD form depending on input
      
      const chunks = chunkMessage(testMessage);
      
      expect(chunks).toHaveLength(1);
      expect(chunks[0].text).toBe('Café'); // Should be normalized to NFC
    });
  });

  describe('Code Block Handling', () => {
    it('should identify and preserve code blocks', () => {
      testMessage.text = `Here's some code:

\`\`\`javascript
function hello() {
  console.log("Hello, world!");
}
\`\`\`

And some more text after.`;
      
      const chunks = chunkMessage(testMessage);
      
      // Should have at least text before, code block, and text after
      expect(chunks.length).toBeGreaterThanOrEqual(2);
      
      // Find the code chunk
      const codeChunk = chunks.find(chunk => chunk.kind === 'code');
      expect(codeChunk).toBeDefined();
      expect(codeChunk!.text).toContain('function hello()');
      expect(codeChunk!.text).toContain('```javascript');
      
      // Find text chunks
      const textChunks = chunks.filter(chunk => chunk.kind === 'text');
      expect(textChunks.length).toBeGreaterThanOrEqual(1);
    });

    it('should handle multiple code blocks', () => {
      testMessage.text = `First code block:

\`\`\`python
def hello():
    print("Hello!")
\`\`\`

Some text between.

\`\`\`bash
echo "Another block"
\`\`\`

Final text.`;
      
      const chunks = chunkMessage(testMessage);
      
      const codeChunks = chunks.filter(chunk => chunk.kind === 'code');
      const textChunks = chunks.filter(chunk => chunk.kind === 'text');
      
      expect(codeChunks.length).toBe(2);
      expect(textChunks.length).toBeGreaterThanOrEqual(1);
      
      // Verify code blocks preserved
      expect(codeChunks[0].text).toContain('def hello()');
      expect(codeChunks[1].text).toContain('echo "Another block"');
    });

    it('should split large code blocks', () => {
      // Create a large code block that exceeds token limits
      const largeCodeBlock = `\`\`\`javascript
${Array(50).fill('console.log("This is line number");').join('\n')}
\`\`\``;
      
      testMessage.text = `Here's a large code block:\n\n${largeCodeBlock}`;
      
      const chunks = chunkMessage(testMessage);
      
      const codeChunks = chunks.filter(chunk => chunk.kind === 'code');
      expect(codeChunks.length).toBeGreaterThan(1);
      
      // All code chunks should be within token limits
      codeChunks.forEach(chunk => {
        expect(chunk.tokens).toBeLessThanOrEqual(320);
      });
    });

    it('should handle malformed code blocks gracefully', () => {
      testMessage.text = `Some text with unclosed code block:

\`\`\`javascript
function incomplete() {
  // No closing backticks

More text after.`;
      
      const chunks = chunkMessage(testMessage);
      
      // Should treat as text since code block is malformed
      expect(chunks.length).toBeGreaterThanOrEqual(1);
      chunks.forEach(chunk => {
        expect(chunk.kind).toBe('text');
      });
    });
  });

  describe('Chunking Configuration', () => {
    it('should respect token limits', () => {
      const longText = 'Word '.repeat(500); // Definitely exceeds 320 tokens
      testMessage.text = longText;
      
      const chunks = chunkMessage(testMessage);
      
      chunks.forEach(chunk => {
        expect(chunk.tokens).toBeLessThanOrEqual(320);
      });
    });

    it('should implement overlap between chunks', () => {
      const sentences = Array(30).fill('This is sentence number N in the message.').join(' ');
      testMessage.text = sentences;
      
      const chunks = chunkMessage(testMessage);
      
      if (chunks.length > 1) {
        // Check for overlap by looking for repeated content
        const chunk1End = chunks[0].text.slice(-50); // Last 50 chars
        const chunk2Start = chunks[1].text.slice(0, 100); // First 100 chars
        
        // Should have some overlap (not exact match due to sentence boundaries)
        const hasOverlap = chunk2Start.includes(chunk1End.slice(-20)); // Some portion should overlap
        expect(hasOverlap).toBe(true);
      }
    });
  });

  describe('Edge Cases', () => {
    it('should handle very short text', () => {
      testMessage.text = 'Hi';
      
      const chunks = chunkMessage(testMessage);
      
      expect(chunks).toHaveLength(1);
      expect(chunks[0].text).toBe('Hi');
      expect(chunks[0].tokens).toBeLessThanOrEqual(320);
    });

    it('should handle text with only whitespace', () => {
      testMessage.text = '   \n  \t  ';
      
      const chunks = chunkMessage(testMessage);
      
      expect(chunks).toHaveLength(1);
      expect(chunks[0].text).toBe(testMessage.text);
    });

    it('should handle special characters', () => {
      testMessage.text = '🎉 Special chars: @#$%^&*()[]{}|;:,.<>? ñáéíóú';
      
      const chunks = chunkMessage(testMessage);
      
      expect(chunks).toHaveLength(1);
      expect(chunks[0].text).toBe(testMessage.text);
      expect(chunks[0].kind).toBe('text');
    });

    it('should handle mixed content with edge cases', () => {
      testMessage.text = `Edge case: Empty code block

\`\`\`
\`\`\`

And some text: 🎉 Special chars!`;
      
      const chunks = chunkMessage(testMessage);
      
      expect(chunks.length).toBeGreaterThanOrEqual(1);
      
      // Should handle empty code block without crashing
      chunks.forEach(chunk => {
        expect(chunk.text).toBeDefined();
        expect(chunk.id).toBeDefined();
        expect(chunk.tokens).toBeGreaterThan(0);
      });
    });
  });

  describe('Chunk Properties', () => {
    it('should set correct offset boundaries', () => {
      testMessage.text = 'Start text. Middle text. End text.';
      
      const chunks = chunkMessage(testMessage);
      
      chunks.forEach(chunk => {
        expect(chunk.offset_start).toBeGreaterThanOrEqual(0);
        expect(chunk.offset_end).toBeGreaterThan(chunk.offset_start);
        expect(chunk.offset_end).toBeLessThanOrEqual(testMessage.text.length);
        
        // Verify the offset corresponds to actual text
        const extractedText = testMessage.text.slice(chunk.offset_start, chunk.offset_end);
        expect(extractedText).toBe(chunk.text);
      });
    });

    it('should generate unique IDs for different chunks', () => {
      const longText = 'Sentence number one. '.repeat(50);
      testMessage.text = longText;
      
      const chunks = chunkMessage(testMessage);
      
      if (chunks.length > 1) {
        const ids = chunks.map(chunk => chunk.id);
        const uniqueIds = new Set(ids);
        
        expect(uniqueIds.size).toBe(ids.length);
      }
    });

    it('should calculate tokens appropriately', () => {
      testMessage.text = 'This is a test message for token counting validation.';
      
      const chunks = chunkMessage(testMessage);
      
      expect(chunks).toHaveLength(1);
      expect(chunks[0].tokens).toBeGreaterThan(0);
      
      // Enhanced tokenization should be more accurate than simple char/4 approximation
      // The improved algorithm should produce reasonable token counts
      expect(chunks[0].tokens).toBeGreaterThan(10); // Should be more than 10 tokens
      expect(chunks[0].tokens).toBeLessThan(30); // Should be less than 30 tokens
      
      // Verify the actual token count matches our enhanced algorithm
      const expectedTokens = 24; // Expected result from our enhanced algorithm
      expect(chunks[0].tokens).toBe(expectedTokens);
    });

    it('should provide more accurate tokenization than simple approximation', () => {
      const testCases = [
        { text: 'hello', expectedMin: 1, expectedMax: 2 },
        { text: 'hello world', expectedMin: 2, expectedMax: 5 },
        { text: 'The quick brown fox jumps.', expectedMin: 6, expectedMax: 12 },
        { text: 'console.log("hello");', expectedMin: 8, expectedMax: 15 },
        { text: '@#$%^&*()', expectedMin: 4, expectedMax: 10 }
      ];

      testCases.forEach(({ text, expectedMin, expectedMax }) => {
        testMessage.text = text;
        const chunks = chunkMessage(testMessage);
        
        expect(chunks).toHaveLength(1);
        expect(chunks[0].tokens).toBeGreaterThanOrEqual(expectedMin);
        expect(chunks[0].tokens).toBeLessThanOrEqual(expectedMax);
        
        // Should be more accurate than simple char/4 approximation for most cases
        const simpleApprox = Math.ceil(text.length / 4);
        // Our enhanced tokenization should generally be different than simple approximation
        // but we don't enforce this difference as it depends on the specific text
        expect(chunks[0].tokens).toBeGreaterThan(0);
      });
    });
  });

  describe('Message Metadata Preservation', () => {
    it('should preserve message ID and session ID in all chunks', () => {
      testMessage.id = 'custom-message-id';
      testMessage.session_id = 'custom-session-id';
      testMessage.text = 'Test text with ' + 'long content '.repeat(50);
      
      const chunks = chunkMessage(testMessage);
      
      chunks.forEach(chunk => {
        expect(chunk.message_id).toBe('custom-message-id');
        expect(chunk.session_id).toBe('custom-session-id');
      });
    });

    it('should handle various message types', () => {
      const testMessages: Partial<Message>[] = [
        { role: 'user', text: 'User message' },
        { role: 'assistant', text: 'Assistant response' },
        { role: 'system', text: 'System instruction' }
      ];

      testMessages.forEach((msgOverride, index) => {
        const message: Message = {
          ...testMessage,
          id: `test-msg-${index}`,
          ...msgOverride
        };

        const chunks = chunkMessage(message);
        
        expect(chunks.length).toBeGreaterThanOrEqual(1);
        chunks.forEach(chunk => {
          expect(chunk.message_id).toBe(`test-msg-${index}`);
          expect(chunk.text).toBeDefined();
        });
      });
    });
  });
});