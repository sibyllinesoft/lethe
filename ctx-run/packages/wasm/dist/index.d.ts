export interface VectorDocument {
    id: string;
    vector: Float32Array;
}
export declare class HnswWasm {
    private vectors;
    private dimension;
    constructor();
    addVector(id: string, vector: Float32Array): void;
    addVectors(docs: VectorDocument[]): void;
    search(queryVector: Float32Array, k: number): {
        id: string;
        score: number;
    }[];
    size(): number;
    clear(): void;
    serialize(): string;
    deserialize(data: string): void;
}
//# sourceMappingURL=index.d.ts.map