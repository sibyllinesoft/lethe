export interface Embeddings {
    name: string;
    dim: number;
    embed(texts: string[]): Promise<Float32Array[]>;
}
export declare class TransformersJsEmbeddings implements Embeddings {
    name: string;
    dim: number;
    private extractor;
    constructor(modelId?: string);
    init(): Promise<void>;
    embed(texts: string[]): Promise<Float32Array[]>;
}
export declare class OllamaEmbeddings implements Embeddings {
    name: string;
    dim: number;
    private url;
    private model;
    constructor(url?: string, model?: string);
    embed(texts: string[]): Promise<Float32Array[]>;
}
export declare class FallbackEmbeddings implements Embeddings {
    name: string;
    dim: number;
    embed(texts: string[]): Promise<Float32Array[]>;
}
export declare function getProvider(pref?: "transformersjs" | "ollama"): Promise<Embeddings>;
//# sourceMappingURL=index.d.ts.map