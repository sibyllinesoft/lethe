"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.chunkMessage = chunkMessage;
const crypto_1 = require("crypto");
// Simple BPE-style token counter approximation
function countTokens(text) {
    // Rough approximation: 1 token ≈ 4 characters as fallback
    // Could be enhanced with gpt-tokenizer for better accuracy
    return Math.ceil(text.length / 4);
}
// Unicode-aware sentence splitter
function splitSentences(text) {
    // Simple regex-based sentence splitting
    const sentences = text.split(/(?<=[.!?])\s+/);
    return sentences.filter(s => s.trim().length > 0);
}
// Detect code fences
function extractCodeFences(text) {
    const parts = [];
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
function createChunks(messageId, sessionId, parts, targetTokens = 320, overlap = 64) {
    const chunks = [];
    for (const part of parts) {
        const tokens = countTokens(part.content);
        if (tokens <= targetTokens) {
            // Part fits in one chunk
            const chunkId = (0, crypto_1.createHash)('sha256')
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
        }
        else {
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
                        const chunkId = (0, crypto_1.createHash)('sha256')
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
                    }
                    else {
                        currentChunk += (currentChunk ? ' ' : '') + sentence;
                        currentTokens += sentenceTokens;
                    }
                }
                // Add final chunk
                if (currentChunk.trim()) {
                    const chunkEnd = currentStart + currentChunk.length;
                    const chunkId = (0, crypto_1.createHash)('sha256')
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
            }
            else {
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
                        const chunkId = (0, crypto_1.createHash)('sha256')
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
                    }
                    else {
                        currentChunk += line;
                        currentTokens += lineTokens;
                    }
                    lineOffset += line.length;
                }
                // Add final chunk
                if (currentChunk.trim()) {
                    const chunkEnd = currentStart + currentChunk.length;
                    const chunkId = (0, crypto_1.createHash)('sha256')
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
function chunkMessage(msg) {
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
//# sourceMappingURL=index.js.map