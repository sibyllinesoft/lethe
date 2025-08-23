"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DEFAULT_OLLAMA_CONFIG = void 0;
exports.getOllamaBridge = getOllamaBridge;
exports.safeParseJSON = safeParseJSON;
exports.testOllamaConnection = testOllamaConnection;
const sqlite_1 = require("@lethe/sqlite");
class OllamaBridgeImpl {
    config;
    constructor(config) {
        this.config = config;
    }
    async isAvailable() {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.config.connectTimeoutMs);
            const response = await fetch(`${this.config.baseUrl}/api/tags`, {
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            clearTimeout(timeoutId);
            return response.ok;
        }
        catch (error) {
            console.debug(`Ollama not available: ${error}`);
            return false;
        }
    }
    async getModels() {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.config.connectTimeoutMs);
            const response = await fetch(`${this.config.baseUrl}/api/tags`, {
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            clearTimeout(timeoutId);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            const data = await response.json();
            return (data.models || []).map((model) => model.name);
        }
        catch (error) {
            console.warn(`Failed to get Ollama models: ${error?.message || error}`);
            return [];
        }
    }
    async generate(request) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.callTimeoutMs);
        try {
            const response = await fetch(`${this.config.baseUrl}/api/generate`, {
                method: 'POST',
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    ...request,
                    stream: false // Force non-streaming for simplicity
                })
            });
            clearTimeout(timeoutId);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            const data = await response.json();
            // Validate response structure
            if (typeof data.response !== 'string') {
                throw new Error('Invalid Ollama response: missing response field');
            }
            return data;
        }
        catch (error) {
            clearTimeout(timeoutId);
            if (error instanceof Error && error.name === 'AbortError') {
                throw new Error(`Ollama request timeout after ${this.config.callTimeoutMs}ms`);
            }
            throw error;
        }
    }
}
// Default configuration
exports.DEFAULT_OLLAMA_CONFIG = {
    baseUrl: 'http://localhost:11434',
    connectTimeoutMs: 500,
    callTimeoutMs: 10000
};
// Cache for bridge instances
const bridgeCache = new Map();
async function getOllamaBridge(db) {
    let config = exports.DEFAULT_OLLAMA_CONFIG;
    // Override with database config if available
    if (db) {
        try {
            const timeoutConfig = (0, sqlite_1.getConfig)(db, 'timeouts');
            if (timeoutConfig) {
                config = {
                    ...config,
                    connectTimeoutMs: timeoutConfig.ollama_connect_ms || config.connectTimeoutMs,
                    callTimeoutMs: Math.max(timeoutConfig.hyde_ms || config.callTimeoutMs, timeoutConfig.summarize_ms || config.callTimeoutMs)
                };
            }
        }
        catch (error) {
            console.debug(`Could not load timeout config: ${error?.message || error}`);
        }
    }
    const cacheKey = JSON.stringify(config);
    if (!bridgeCache.has(cacheKey)) {
        bridgeCache.set(cacheKey, new OllamaBridgeImpl(config));
    }
    return bridgeCache.get(cacheKey);
}
// Helper function to safely parse JSON from Ollama responses
function safeParseJSON(text, fallback) {
    try {
        // Clean up common JSON formatting issues from LLMs
        let cleaned = text.trim();
        // Remove markdown code blocks
        cleaned = cleaned.replace(/^```json?\s*|\s*```$/gm, '');
        // Find JSON object boundaries
        const start = cleaned.indexOf('{');
        const end = cleaned.lastIndexOf('}');
        if (start !== -1 && end !== -1 && end > start) {
            cleaned = cleaned.substring(start, end + 1);
        }
        return JSON.parse(cleaned);
    }
    catch (error) {
        console.warn(`Failed to parse JSON from Ollama response: ${error}`);
        console.debug(`Raw text: ${text}`);
        return fallback;
    }
}
// Test function for CLI diagnostics
async function testOllamaConnection(db) {
    try {
        const bridge = await getOllamaBridge(db);
        const available = await bridge.isAvailable();
        if (!available) {
            return { available: false, models: [], error: 'Ollama service not reachable' };
        }
        const models = await bridge.getModels();
        // Test basic generation with a simple model if available
        let testGeneration = false;
        if (models.length > 0) {
            try {
                const testModel = models.find(m => m.includes('llama')) || models[0];
                const response = await bridge.generate({
                    model: testModel,
                    prompt: 'Hello, respond with just "OK"',
                    temperature: 0,
                    max_tokens: 10
                });
                testGeneration = response.response.includes('OK');
            }
            catch (error) {
                console.debug(`Test generation failed: ${error}`);
            }
        }
        return { available, models, testGeneration };
    }
    catch (error) {
        return { available: false, models: [], error: error?.message || String(error) };
    }
}
//# sourceMappingURL=index.js.map