import type { Candidate } from '../retrieval/index.js';
export interface Diversifier {
    name: string;
    diversify(candidates: Candidate[], k: number): Promise<Candidate[]>;
}
export declare class EntityCoverageDiversifier implements Diversifier {
    name: string;
    private entityExtractorCache;
    diversify(candidates: Candidate[], k: number): Promise<Candidate[]>;
    private extractEntities;
    private isCommonWord;
    private calculateContentDiversity;
    private tokenize;
}
export declare class SemanticDiversifier implements Diversifier {
    private embeddings?;
    private seed;
    name: string;
    constructor(embeddings?: any | undefined, seed?: number);
    diversify(candidates: Candidate[], k: number): Promise<Candidate[]>;
    private autoK;
    private kmeansFast;
    private cosineDistance;
}
export declare class NoOpDiversifier implements Diversifier {
    name: string;
    diversify(candidates: Candidate[], k: number): Promise<Candidate[]>;
}
export declare function getDiversifier(enabled?: boolean, method?: string): Promise<Diversifier>;
//# sourceMappingURL=index.d.ts.map