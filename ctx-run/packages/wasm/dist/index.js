"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.HnswWasm = void 0;
// Simple cosine similarity calculation
function cosineSimilarity(a, b) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    if (normA === 0 || normB === 0)
        return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
class HnswWasm {
    vectors = [];
    dimension = 0;
    constructor() {
        console.log("HnswWasm initialized (in-memory brute force fallback).");
    }
    addVector(id, vector) {
        if (this.dimension === 0) {
            this.dimension = vector.length;
        }
        else if (vector.length !== this.dimension) {
            throw new Error(`Vector dimension mismatch: expected ${this.dimension}, got ${vector.length}`);
        }
        // Remove existing vector with same ID
        this.vectors = this.vectors.filter(v => v.id !== id);
        // Add new vector
        this.vectors.push({ id, vector });
    }
    addVectors(docs) {
        for (const doc of docs) {
            this.addVector(doc.id, doc.vector);
        }
    }
    search(queryVector, k) {
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
    size() {
        return this.vectors.length;
    }
    clear() {
        this.vectors = [];
        this.dimension = 0;
    }
    // Serialize index for persistence (optional)
    serialize() {
        return JSON.stringify({
            dimension: this.dimension,
            vectors: this.vectors.map(doc => ({
                id: doc.id,
                vector: Array.from(doc.vector)
            }))
        });
    }
    // Deserialize index from persistence (optional)
    deserialize(data) {
        const parsed = JSON.parse(data);
        this.dimension = parsed.dimension;
        this.vectors = parsed.vectors.map((doc) => ({
            id: doc.id,
            vector: new Float32Array(doc.vector)
        }));
    }
}
exports.HnswWasm = HnswWasm;
//# sourceMappingURL=index.js.map