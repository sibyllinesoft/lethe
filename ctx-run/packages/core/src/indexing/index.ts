import { DB, Message, Chunk, insertMessages, insertChunks, insertEmbeddings, getChunksBySession } from '@lethe/sqlite';
import { chunkMessage } from '../chunker';
import { rebuildForDb } from '../dfidf';

export async function upsertMessages(sessionId: string, messages: Message[]): Promise<void> {
    // Package contract signature - implementation needs DB injection
    throw new Error('upsertMessages needs DB instance - will be fixed in CLI integration');
}

export async function upsertMessagesWithDb(db: DB, sessionId: string, messages: Message[]): Promise<void> {
    // Add session_id to all messages
    const messagesWithSession = messages.map(msg => ({ ...msg, session_id: sessionId }));
    
    // Insert messages
    insertMessages(db, messagesWithSession);
    
    // Generate and insert chunks
    const allChunks: Chunk[] = [];
    for (const message of messagesWithSession) {
        const chunks = chunkMessage(message);
        allChunks.push(...chunks);
    }
    
    if (allChunks.length > 0) {
        insertChunks(db, allChunks);
        console.log(`Generated ${allChunks.length} chunks from ${messages.length} messages`);
    }
    
    // Rebuild DF/IDF for the session
    await rebuildForDb(db, sessionId);
}

export async function ensureEmbeddings(sessionId: string): Promise<void> {
    // Package contract signature - implementation needs DB injection
    throw new Error('ensureEmbeddings needs DB instance - will be fixed in CLI integration');
}

export async function ensureEmbeddingsWithDb(db: DB, sessionId: string): Promise<void> {
    // Get chunks that don't have embeddings yet
    const allChunks = getChunksBySession(db, sessionId);
    if (allChunks.length === 0) {
        console.log('No chunks found for session');
        return;
    }
    
    const existingEmbeddings = db.prepare('SELECT chunk_id FROM embeddings WHERE chunk_id IN (' + 
        allChunks.map(() => '?').join(',') + ')').all(...allChunks.map(c => c.id)) as { chunk_id: string }[];
    
    const existingIds = new Set(existingEmbeddings.map(e => e.chunk_id));
    const chunksNeedingEmbeddings = allChunks.filter(c => !existingIds.has(c.id));
    
    if (chunksNeedingEmbeddings.length === 0) {
        console.log('No new chunks need embeddings.');
        return;
    }

    console.log(`Need to embed ${chunksNeedingEmbeddings.length} chunks`);
    
    // Import embeddings provider dynamically to avoid loading overhead
    try {
        const { getProvider } = await import('@lethe/embeddings');
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
        insertEmbeddings(db, embeddingRows);
        
        // Refresh vector index since we added new embeddings
        const { refreshVectorIndex } = await import('@lethe/sqlite');
        refreshVectorIndex();
        
        console.log(`Successfully embedded ${chunksNeedingEmbeddings.length} chunks`);
        
    } catch (error) {
        console.error('Failed to generate embeddings:', error);
        console.log('Creating placeholder embeddings for graceful degradation');
        
        // Fall back to placeholder embeddings
        const placeholderDim = 384; // bge-small-en-v1.5 dimension
        const placeholderEmbeddings = chunksNeedingEmbeddings.map(chunk => ({
            chunk_id: chunk.id,
            dim: placeholderDim,
            vec: Buffer.alloc(placeholderDim * 4) // 4 bytes per float32
        }));
        
        insertEmbeddings(db, placeholderEmbeddings);
        console.log(`Created placeholder embeddings for ${chunksNeedingEmbeddings.length} chunks`);
    }
}

export async function ensureVectorIndex(): Promise<void> {
    // Package contract signature - will be implemented in M2
    console.log('Vector index creation will be implemented in M2');
}
