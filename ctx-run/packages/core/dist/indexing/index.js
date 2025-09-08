"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.upsertMessages = upsertMessages;
exports.upsertMessagesWithDb = upsertMessagesWithDb;
exports.ensureEmbeddings = ensureEmbeddings;
exports.ensureEmbeddingsWithDb = ensureEmbeddingsWithDb;
exports.ensureVectorIndex = ensureVectorIndex;
const sqlite_1 = require("@lethe/sqlite");
const chunker_1 = require("../chunker");
const dfidf_1 = require("../dfidf");
async function upsertMessages(sessionId, messages) {
    // Package contract signature - implementation needs DB injection
    throw new Error('upsertMessages needs DB instance - will be fixed in CLI integration');
}
async function upsertMessagesWithDb(db, sessionId, messages) {
    // Add session_id to all messages
    const messagesWithSession = messages.map(msg => ({ ...msg, session_id: sessionId }));
    // Insert messages
    (0, sqlite_1.insertMessages)(db, messagesWithSession);
    // Generate and insert chunks
    const allChunks = [];
    for (const message of messagesWithSession) {
        const chunks = (0, chunker_1.chunkMessage)(message);
        allChunks.push(...chunks);
    }
    if (allChunks.length > 0) {
        (0, sqlite_1.insertChunks)(db, allChunks);
        console.log(`Generated ${allChunks.length} chunks from ${messages.length} messages`);
    }
    // Rebuild DF/IDF for the session
    await (0, dfidf_1.rebuildForDb)(db, sessionId);
}
async function ensureEmbeddings(sessionId) {
    // Package contract signature - implementation needs DB injection
    throw new Error('ensureEmbeddings needs DB instance - will be fixed in CLI integration');
}
async function ensureEmbeddingsWithDb(db, sessionId) {
    // Get chunks that don't have embeddings yet
    const allChunks = (0, sqlite_1.getChunksBySession)(db, sessionId);
    if (allChunks.length === 0) {
        console.log('No chunks found for session');
        return;
    }
    const existingEmbeddings = db.prepare('SELECT chunk_id FROM embeddings WHERE chunk_id IN (' +
        allChunks.map(() => '?').join(',') + ')').all(...allChunks.map(c => c.id));
    const existingIds = new Set(existingEmbeddings.map(e => e.chunk_id));
    const chunksNeedingEmbeddings = allChunks.filter(c => !existingIds.has(c.id));
    if (chunksNeedingEmbeddings.length === 0) {
        console.log('No new chunks need embeddings.');
        return;
    }
    console.log(`Need to embed ${chunksNeedingEmbeddings.length} chunks`);
    // Import embeddings provider dynamically to avoid loading overhead
    try {
        const { getProvider } = await Promise.resolve().then(() => __importStar(require('@lethe/embeddings')));
        const provider = await getProvider('transformersjs');
        // Extract texts to embed
        const texts = chunksNeedingEmbeddings.map(chunk => chunk.text);
        // Generate embeddings
        const embeddings = await provider.embed(texts);
        // Convert to database format
        const embeddingRows = embeddings.map((embedding, index) => ({
            chunk_id: chunksNeedingEmbeddings[index].id,
            dim: provider.dim,
            vec: Buffer.from(embedding.buffer)
        }));
        // Insert into database
        (0, sqlite_1.insertEmbeddings)(db, embeddingRows);
        // Refresh vector index since we added new embeddings
        const { refreshVectorIndex } = await Promise.resolve().then(() => __importStar(require('@lethe/sqlite')));
        refreshVectorIndex();
        console.log(`Successfully embedded ${chunksNeedingEmbeddings.length} chunks`);
    }
    catch (error) {
        console.error('Failed to generate embeddings:', error);
        console.log('Creating placeholder embeddings for graceful degradation');
        // Fall back to placeholder embeddings
        const placeholderDim = 384; // bge-small-en-v1.5 dimension
        const placeholderEmbeddings = chunksNeedingEmbeddings.map(chunk => ({
            chunk_id: chunk.id,
            dim: placeholderDim,
            vec: Buffer.alloc(placeholderDim * 4) // 4 bytes per float32
        }));
        (0, sqlite_1.insertEmbeddings)(db, placeholderEmbeddings);
        console.log(`Created placeholder embeddings for ${chunksNeedingEmbeddings.length} chunks`);
    }
}
async function ensureVectorIndex() {
    // Package contract signature - will be implemented in M2
    console.log('Vector index creation will be implemented in M2');
}
//# sourceMappingURL=index.js.map