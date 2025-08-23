"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.NoOpDiversifier = exports.EntityCoverageDiversifier = void 0;
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
class NoOpDiversifier {
    name = "noop";
    async diversify(candidates, k) {
        return candidates.slice(0, k); // Just take top k
    }
}
exports.NoOpDiversifier = NoOpDiversifier;
async function getDiversifier(enabled = true) {
    if (!enabled) {
        return new NoOpDiversifier();
    }
    return new EntityCoverageDiversifier();
}
//# sourceMappingURL=index.js.map