export interface VectorDocument {
    id: string;
    vector: Float32Array;
}

// Simple cosine similarity calculation
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export class HnswWasm {
    private vectors: VectorDocument[] = [];
    private dimension: number = 0;

    constructor() {
        console.log("HnswWasm initialized (in-memory brute force fallback).");
    }

    addVector(id: string, vector: Float32Array): void {
        if (this.dimension === 0) {
            this.dimension = vector.length;
        } else if (vector.length !== this.dimension) {
            throw new Error(`Vector dimension mismatch: expected ${this.dimension}, got ${vector.length}`);
        }

        // Remove existing vector with same ID
        this.vectors = this.vectors.filter(v => v.id !== id);
        
        // Add new vector
        this.vectors.push({ id, vector });
    }

    addVectors(docs: VectorDocument[]): void {
        for (const doc of docs) {
            this.addVector(doc.id, doc.vector);
        }
    }

    search(queryVector: Float32Array, k: number): { id: string; score: number }[] {
        if (this.vectors.length === 0) {
            return [];
        }

        if (queryVector.length !== this.dimension) {
            throw new Error(`Query vector dimension mismatch: expected ${this.dimension}, got ${queryVector.length}`);
        }

        // Compute similarities for all vectors
        const similarities = this.vectors.map(doc => ({
            id: doc.id,
            score: cosineSimilarity(queryVector, doc.vector)
        }));

        // Sort by similarity (descending) and take top k
        similarities.sort((a, b) => b.score - a.score);
        return similarities.slice(0, k);
    }

    size(): number {
        return this.vectors.length;
    }

    clear(): void {
        this.vectors = [];
        this.dimension = 0;
    }

    // Serialize index for persistence (optional)
    serialize(): string {
        return JSON.stringify({
            dimension: this.dimension,
            vectors: this.vectors.map(doc => ({
                id: doc.id,
                vector: Array.from(doc.vector)
            }))
        });
    }

    // Deserialize index from persistence (optional)
    deserialize(data: string): void {
        const parsed = JSON.parse(data);
        this.dimension = parsed.dimension;
        this.vectors = parsed.vectors.map((doc: any) => ({
            id: doc.id,
            vector: new Float32Array(doc.vector)
        }));
    }
}
