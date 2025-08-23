import type { DB } from '@lethe/sqlite';
export interface OllamaConfig {
    baseUrl: string;
    connectTimeoutMs: number;
    callTimeoutMs: number;
}
export interface OllamaRequest {
    model: string;
    prompt: string;
    stream?: boolean;
    temperature?: number;
    max_tokens?: number;
}
export interface OllamaResponse {
    response: string;
    done: boolean;
    context?: number[];
    total_duration?: number;
    load_duration?: number;
    prompt_eval_count?: number;
    prompt_eval_duration?: number;
    eval_count?: number;
    eval_duration?: number;
}
export interface OllamaBridge {
    generate(request: OllamaRequest): Promise<OllamaResponse>;
    isAvailable(): Promise<boolean>;
    getModels(): Promise<string[]>;
}
export declare const DEFAULT_OLLAMA_CONFIG: OllamaConfig;
export declare function getOllamaBridge(db?: DB): Promise<OllamaBridge>;
export declare function safeParseJSON<T>(text: string, fallback: T): T;
export declare function testOllamaConnection(db?: DB): Promise<{
    available: boolean;
    models: string[];
    testGeneration?: boolean;
    error?: string;
}>;
//# sourceMappingURL=index.d.ts.map