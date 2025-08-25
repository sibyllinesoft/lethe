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
exports.NoOpDiversifier = exports.SemanticDiversifier = exports.EntityCoverageDiversifier = void 0;
exports.getDiversifier = getDiversifier;
class EntityCoverageDiversifier {
    name = "entity-coverage";
    entityExtractorCache = new Map();
    async diversify(candidates, k) {
        if (candidates.length <= k) {
            return candidates; // No need to diversify
        }
        console.log(`Diversifying ${candidates.length} candidates to top ${k} with entity coverage`);
        // Extract entities from all candidates
        const candidateEntitiesMap = new Map();
        for (let i = 0; i < candidates.length; i++) {
            const candidate = candidates[i];
            if (candidate.text) {
                candidateEntitiesMap.set(i, await this.extractEntities(candidate.text));
            }
            else {
                candidateEntitiesMap.set(i, new Set());
            }
        }
        // Maximal Marginal Relevance (MMR) with entity coverage
        const selected = [];
        const selectedEntities = new Set();
        const remaining = new Set(Array.from({ length: candidates.length }, (_, i) => i));
        // Always select the highest-scoring candidate first
        if (remaining.size > 0) {
            const firstIndex = 0; // Candidates are already sorted by score
            selected.push(firstIndex);
            remaining.delete(firstIndex);
            const firstEntities = candidateEntitiesMap.get(firstIndex) || new Set();
            for (const entity of firstEntities) {
                selectedEntities.add(entity);
            }
        }
        // Select remaining candidates using MMR with entity diversity
        while (selected.length < k && remaining.size > 0) {
            let bestCandidate = -1;
            let bestScore = -Infinity;
            for (const candidateIndex of remaining) {
                const candidate = candidates[candidateIndex];
                const candidateEntities = candidateEntitiesMap.get(candidateIndex) || new Set();
                // Relevance score (already normalized 0-1)
                const relevanceScore = candidate.score;
                // Entity diversity score
                const newEntities = new Set([...candidateEntities].filter(e => !selectedEntities.has(e)));
                const diversityScore = newEntities.size / Math.max(candidateEntities.size, 1);
                // Content diversity score (simple text overlap)
                const contentDiversityScore = this.calculateContentDiversity(candidate.text || '', selected.map(i => candidates[i].text || ''));
                // Combined score: balance relevance and diversity
                const lambda = 0.6; // Weight for relevance vs diversity
                const alpha = 0.7; // Weight for entity diversity vs content diversity
                const combinedDiversityScore = alpha * diversityScore + (1 - alpha) * contentDiversityScore;
                const mmrScore = lambda * relevanceScore + (1 - lambda) * combinedDiversityScore;
                if (mmrScore > bestScore) {
                    bestScore = mmrScore;
                    bestCandidate = candidateIndex;
                }
            }
            if (bestCandidate !== -1) {
                selected.push(bestCandidate);
                remaining.delete(bestCandidate);
                // Update selected entities
                const newEntities = candidateEntitiesMap.get(bestCandidate) || new Set();
                for (const entity of newEntities) {
                    selectedEntities.add(entity);
                }
            }
            else {
                break; // No more candidates
            }
        }
        const diversifiedCandidates = selected.map(i => candidates[i]);
        console.log(`Diversification complete - selected ${diversifiedCandidates.length} candidates covering ${selectedEntities.size} unique entities`);
        return diversifiedCandidates;
    }
    async extractEntities(text) {
        // Use cache to avoid re-processing same text
        if (this.entityExtractorCache.has(text)) {
            return this.entityExtractorCache.get(text);
        }
        const entities = new Set();
        // Simple entity extraction patterns
        // 1. Capitalized words (potential proper nouns)
        const capitalizedWords = text.match(/\b[A-Z][a-zA-Z]{2,}\b/g) || [];
        for (const word of capitalizedWords) {
            if (!this.isCommonWord(word)) {
                entities.add(word.toLowerCase());
            }
        }
        // 2. File paths and extensions
        const filePaths = text.match(/[a-zA-Z0-9_-]+\.[a-zA-Z0-9]{1,4}/g) || [];
        for (const path of filePaths) {
            entities.add(path);
        }
        // 3. Function/method names in code
        const functionNames = text.match(/\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(/g) || [];
        for (const func of functionNames) {
            const name = func.replace(/\s*\($/, '');
            entities.add(name);
        }
        // 4. Package/module names
        const imports = text.match(/(?:import|from|require)\s+['""]([^'"]+)['"]/g) || [];
        for (const imp of imports) {
            const match = imp.match(/['""]([^'"]+)['"]/);
            if (match) {
                entities.add(match[1]);
            }
        }
        // 5. URLs and domains
        const urls = text.match(/https?:\/\/[^\s]+/g) || [];
        for (const url of urls) {
            try {
                const domain = new URL(url).hostname;
                entities.add(domain);
            }
            catch {
                // Invalid URL, skip
            }
        }
        // 6. Technical terms (APIs, frameworks, etc.)
        const techTerms = text.match(/\b(API|HTTP|JSON|XML|SQL|React|TypeScript|JavaScript|Python|Node\.js|Docker|Git|AWS|API)\b/gi) || [];
        for (const term of techTerms) {
            entities.add(term.toLowerCase());
        }
        // Cache the result
        this.entityExtractorCache.set(text, entities);
        return entities;
    }
    isCommonWord(word) {
        // Common English words that aren't likely to be entities
        const commonWords = new Set([
            'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'Or', 'But',
            'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By', 'From', 'Up', 'About',
            'Into', 'Through', 'During', 'Before', 'After', 'Above', 'Below', 'Between',
            'Among', 'All', 'Any', 'Both', 'Each', 'Few', 'More', 'Most', 'Other',
            'Some', 'Such', 'Only', 'Own', 'Same', 'So', 'Than', 'Too', 'Very',
            'Can', 'Will', 'Just', 'Should', 'Now', 'Here', 'There', 'When', 'Where',
            'Why', 'How', 'What', 'Which', 'Who', 'Whom', 'Whose', 'Whether', 'While'
        ]);
        return commonWords.has(word);
    }
    calculateContentDiversity(candidateText, selectedTexts) {
        if (selectedTexts.length === 0) {
            return 1.0; // Maximum diversity if no previous selections
        }
        const candidateTerms = this.tokenize(candidateText.toLowerCase());
        if (candidateTerms.length === 0) {
            return 0.5; // Neutral diversity for empty text
        }
        // Calculate minimum similarity with any selected document
        let minSimilarity = 1.0;
        for (const selectedText of selectedTexts) {
            const selectedTerms = this.tokenize(selectedText.toLowerCase());
            if (selectedTerms.length === 0)
                continue;
            // Jaccard similarity
            const intersection = candidateTerms.filter(term => selectedTerms.includes(term));
            const union = new Set([...candidateTerms, ...selectedTerms]);
            const similarity = intersection.length / union.size;
            minSimilarity = Math.min(minSimilarity, similarity);
        }
        // Return diversity (1 - similarity)
        return 1.0 - minSimilarity;
    }
    tokenize(text) {
        return text
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(term => term.length > 1);
    }
}
exports.EntityCoverageDiversifier = EntityCoverageDiversifier;
class SemanticDiversifier {
    embeddings;
    seed;
    name = "semantic";
    constructor(embeddings, seed = 11) {
        this.embeddings = embeddings;
        this.seed = seed;
    }
    async diversify(candidates, k) {
        if (candidates.length <= k) {
            return candidates; // No need to diversify
        }
        console.log(`Semantic diversifying ${candidates.length} candidates to top ${k}`);
        // Extract texts for embedding
        const texts = candidates.map(c => c.text || '').filter(t => t.length > 0);
        if (texts.length === 0) {
            return candidates.slice(0, k); // Fallback to top-k
        }
        try {
            // Get embeddings for all candidate texts
            const embeddings = this.embeddings || (await Promise.resolve().then(() => __importStar(require('@lethe/embeddings')))).getProvider();
            const vectors = await embeddings.embed(texts);
            if (vectors.length !== texts.length) {
                console.warn('Embedding length mismatch, falling back to entity-based diversification');
                return candidates.slice(0, k);
            }
            // Perform k-means clustering
            const K = this.autoK(k, vectors.length);
            const { clusters, medoids } = this.kmeansFast(vectors, K, this.seed);
            // Score clusters by max candidate score
            const clusterScores = clusters.map((cluster, idx) => {
                const maxScore = Math.max(...cluster.map(candidateIdx => candidates[candidateIdx].score));
                return { clusterIdx: idx, maxScore, cluster };
            });
            // Sort clusters by score descending
            clusterScores.sort((a, b) => b.maxScore - a.maxScore);
            // Greedily pick medoids from clusters until we reach k candidates
            const selected = new Set();
            const result = [];
            for (const { clusterIdx, cluster } of clusterScores) {
                if (result.length >= k)
                    break;
                const medoidIdx = medoids[clusterIdx];
                if (!selected.has(medoidIdx)) {
                    selected.add(medoidIdx);
                    result.push(candidates[medoidIdx]);
                }
            }
            // Fill remaining slots with highest-scoring candidates not yet selected
            for (let i = 0; i < candidates.length && result.length < k; i++) {
                if (!selected.has(i)) {
                    result.push(candidates[i]);
                    selected.add(i);
                }
            }
            console.log(`Semantic diversification complete - selected ${result.length} candidates from ${K} clusters`);
            return result;
        }
        catch (error) {
            console.warn(`Semantic diversification failed: ${error}, falling back to top-k`);
            return candidates.slice(0, k);
        }
    }
    autoK(targetK, n) {
        // Auto-determine number of clusters
        if (targetK === 0) {
            return Math.min(Math.ceil(n / 3), 24);
        }
        return Math.min(targetK * 2, Math.max(8, Math.ceil(n / 3)));
    }
    kmeansFast(vectors, k, seed) {
        // Simple k-means implementation with seeded random initialization
        const n = vectors.length;
        if (n <= k) {
            return {
                clusters: vectors.map((_, i) => [i]),
                medoids: vectors.map((_, i) => i)
            };
        }
        // Seed random number generator
        let seedState = seed;
        const random = () => {
            seedState = (seedState * 9301 + 49297) % 233280;
            return seedState / 233280;
        };
        // Initialize centroids randomly
        const centroids = [];
        const chosenIndices = new Set();
        for (let i = 0; i < k; i++) {
            let idx;
            do {
                idx = Math.floor(random() * n);
            } while (chosenIndices.has(idx));
            chosenIndices.add(idx);
            centroids.push(new Float32Array(vectors[idx]));
        }
        // Run k-means for a few iterations
        let clusters = [];
        for (let iter = 0; iter < 10; iter++) {
            // Assign points to clusters
            clusters = Array.from({ length: k }, () => []);
            for (let i = 0; i < n; i++) {
                let bestCluster = 0;
                let bestDist = this.cosineDistance(vectors[i], centroids[0]);
                for (let j = 1; j < k; j++) {
                    const dist = this.cosineDistance(vectors[i], centroids[j]);
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestCluster = j;
                    }
                }
                clusters[bestCluster].push(i);
            }
            // Update centroids
            for (let j = 0; j < k; j++) {
                if (clusters[j].length === 0)
                    continue;
                const newCentroid = new Float32Array(vectors[0].length);
                for (const idx of clusters[j]) {
                    for (let d = 0; d < newCentroid.length; d++) {
                        newCentroid[d] += vectors[idx][d];
                    }
                }
                for (let d = 0; d < newCentroid.length; d++) {
                    newCentroid[d] /= clusters[j].length;
                }
                centroids[j] = newCentroid;
            }
        }
        // Find medoids (closest point to centroid in each cluster)
        const medoids = clusters.map((cluster, clusterIdx) => {
            if (cluster.length === 0)
                return 0;
            let bestIdx = cluster[0];
            let bestDist = this.cosineDistance(vectors[bestIdx], centroids[clusterIdx]);
            for (const idx of cluster.slice(1)) {
                const dist = this.cosineDistance(vectors[idx], centroids[clusterIdx]);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }
            return bestIdx;
        });
        return { clusters, medoids };
    }
    cosineDistance(a, b) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
        return 1 - similarity; // Convert similarity to distance
    }
}
exports.SemanticDiversifier = SemanticDiversifier;
class NoOpDiversifier {
    name = "noop";
    async diversify(candidates, k) {
        return candidates.slice(0, k); // Just take top k
    }
}
exports.NoOpDiversifier = NoOpDiversifier;
async function getDiversifier(enabled = true, method = 'entity') {
    if (!enabled) {
        return new NoOpDiversifier();
    }
    if (method === 'semantic') {
        return new SemanticDiversifier();
    }
    return new EntityCoverageDiversifier();
}
//# sourceMappingURL=index.js.map