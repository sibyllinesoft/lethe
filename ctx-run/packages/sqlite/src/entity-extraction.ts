/**
 * @fileoverview Entity extraction pipeline for Lethe agent context atoms
 * Supports regex patterns and optional NER with fallback
 */

import { Entity, EntityKind, EntityExtractionConfig, ExtractedEntities } from './atoms-types.js';

/**
 * Default entity extraction patterns
 */
export const DEFAULT_ENTITY_PATTERNS: EntityExtractionConfig['patterns'] = {
  // Code identifiers: variables, functions, classes, modules
  id: [
    /\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\b/g, // module.function
    /\b[A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)*\b/g,         // PascalCase
    /\b[a-z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)*\b/g,         // camelCase
    /\b[a-zA-Z_][a-zA-Z0-9_]*\(\)/g,                         // function()
    /\$\{[^}]+\}/g,                                          // ${variable}
    /\$[a-zA-Z_][a-zA-Z0-9_]*\b/g,                           // $variable
  ],
  
  // File paths and extensions
  file: [
    /(?:\.\/|\/|~\/)[a-zA-Z0-9._/-]+\.[a-zA-Z0-9]+/g,       // Relative/absolute paths with extensions
    /[a-zA-Z0-9._-]+\.[a-zA-Z0-9]{2,4}\b/g,                 // Simple filenames
    /\/[a-zA-Z0-9._/-]+/g,                                   // Unix paths
    /[A-Z]:\\[a-zA-Z0-9._\\-]+/g,                            // Windows paths
    /"[^"]*\.[a-zA-Z0-9]{2,4}"/g,                            // Quoted filenames
  ],
  
  // Error codes and patterns
  error: [
    /\b[A-Z]+_[A-Z0-9_]+\b/g,                                // ERROR_CODE
    /\b[Ee]rror\s*:\s*[A-Z][a-zA-Z0-9_\s]+/g,               // Error: Message
    /\b[Ee]xception\s*:\s*[A-Z][a-zA-Z0-9_\s]+/g,           // Exception: Message
    /\b\d{3,4}\b/g,                                          // HTTP status codes
    /0x[0-9A-Fa-f]+/g,                                       // Hex error codes
    /\b[A-Z][a-zA-Z]*Error\b/g,                             // SomeError classes
    /\bERR_[A-Z0-9_]+\b/g,                                   // Node.js error codes
  ],
  
  // API endpoints and URLs
  api: [
    /https?:\/\/[^\s<>"']+/g,                                // HTTP URLs
    /\b\/api\/[a-zA-Z0-9/_-]+/g,                            // API paths
    /\b\/v\d+\/[a-zA-Z0-9/_-]+/g,                           // Versioned API paths
    /\{[a-zA-Z0-9_-]+\}/g,                                  // {parameter} in URLs
    /:[a-zA-Z0-9_-]+\b/g,                                   // :parameter in URLs
  ],
  
  // Tool names and commands
  tool: [
    /\b(?:git|npm|node|python|pip|curl|grep|awk|sed|docker|kubectl)\b/gi,
    /\b[a-zA-Z0-9_-]+\.(?:sh|py|js|ts|exe|bat)\b/g,         // Executable files
    /\$\([^)]+\)/g,                                         // $(command)
    /`[^`]+`/g,                                             // `command`
  ],
  
  // Miscellaneous patterns
  misc: [
    /\b[A-Z]{2,}\b/g,                                       // CONSTANTS
    /\b\d+\.\d+\.\d+(?:\.\d+)?\b/g,                        // Version numbers / IPs
    /\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b/gi, // UUIDs
    /\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b/g, // Email addresses
  ],
};

/**
 * Default entity extraction configuration
 */
export const DEFAULT_ENTITY_CONFIG: EntityExtractionConfig = {
  patterns: DEFAULT_ENTITY_PATTERNS,
  ner: {
    enabled: false, // Start with regex-only, can add NER later
    fallbackToPatterns: true,
  },
  useSessionIdf: true,
  minWeight: 0.1,
};

/**
 * Simple NER interface - can be implemented with different models
 */
export interface NerModel {
  extractEntities(text: string): Promise<Array<{
    text: string;
    label: string;
    confidence: number;
    start: number;
    end: number;
  }>>;
}

/**
 * Entity extractor class
 */
export class EntityExtractor {
  private config: EntityExtractionConfig;
  private nerModel?: NerModel;

  constructor(config: EntityExtractionConfig = DEFAULT_ENTITY_CONFIG) {
    this.config = config;
  }

  /**
   * Set NER model for advanced entity recognition
   */
  setNerModel(model: NerModel): void {
    this.nerModel = model;
  }

  /**
   * Extract entities from text using regex patterns
   */
  private extractWithRegex(text: string): Array<{
    text: string;
    kind: EntityKind;
    confidence: number;
    source: 'regex';
  }> {
    const entities: Array<{
      text: string;
      kind: EntityKind;
      confidence: number;
      source: 'regex';
    }> = [];

    // Apply each pattern category
    for (const [kind, patterns] of Object.entries(this.config.patterns)) {
      for (const pattern of patterns) {
        const matches = text.matchAll(pattern);
        for (const match of matches) {
          if (match[0] && match[0].length > 1) { // Skip single chars
            entities.push({
              text: match[0].trim(),
              kind: kind as EntityKind,
              confidence: 0.8, // Fixed confidence for regex
              source: 'regex',
            });
          }
        }
      }
    }

    return entities;
  }

  /**
   * Extract entities using NER model
   */
  private async extractWithNer(text: string): Promise<Array<{
    text: string;
    kind: EntityKind;
    confidence: number;
    source: 'ner';
  }>> {
    if (!this.nerModel) {
      return [];
    }

    try {
      const nerResults = await this.nerModel.extractEntities(text);
      
      return nerResults.map(result => ({
        text: result.text,
        kind: this.mapNerLabelToKind(result.label),
        confidence: result.confidence,
        source: 'ner' as const,
      }));
    } catch (error) {
      console.warn('NER extraction failed:', error);
      return [];
    }
  }

  /**
   * Map NER labels to our entity kinds
   */
  private mapNerLabelToKind(nerLabel: string): EntityKind {
    const label = nerLabel.toUpperCase();
    
    // Map common NER labels to our kinds
    if (label.includes('PER')) return 'person';
    if (label.includes('ORG')) return 'org';
    if (label.includes('FILE') || label.includes('PATH')) return 'file';
    if (label.includes('CODE') || label.includes('FUNC')) return 'id';
    if (label.includes('ERROR')) return 'error';
    if (label.includes('API') || label.includes('URL')) return 'api';
    if (label.includes('TOOL')) return 'tool';
    
    return 'misc';
  }

  /**
   * Deduplicate and filter entities
   */
  private deduplicateEntities(entities: Array<{
    text: string;
    kind: EntityKind;
    confidence: number;
    source: 'regex' | 'ner';
  }>): Array<{
    text: string;
    kind: EntityKind;
    confidence: number;
    source: 'regex' | 'ner';
  }> {
    const entityMap = new Map<string, typeof entities[0]>();
    
    for (const entity of entities) {
      const key = `${entity.text.toLowerCase()}:${entity.kind}`;
      const existing = entityMap.get(key);
      
      if (!existing || entity.confidence > existing.confidence) {
        entityMap.set(key, entity);
      }
    }
    
    return Array.from(entityMap.values())
      .filter(e => e.confidence >= this.config.minWeight)
      .sort((a, b) => b.confidence - a.confidence);
  }

  /**
   * Extract entities from text
   */
  async extractEntities(text: string): Promise<ExtractedEntities> {
    const startTime = Date.now();
    
    // Extract with regex patterns
    const regexEntities = this.extractWithRegex(text);
    
    // Extract with NER if enabled
    let nerEntities: Array<{
      text: string;
      kind: EntityKind;
      confidence: number;
      source: 'ner';
    }> = [];
    
    if (this.config.ner?.enabled && this.nerModel) {
      nerEntities = await this.extractWithNer(text);
    }
    
    // Combine and deduplicate
    const allEntities = [...regexEntities, ...nerEntities];
    const dedupedEntities = this.deduplicateEntities(allEntities);
    
    const processingTime = Date.now() - startTime;
    
    return {
      entities: dedupedEntities,
      processingTime,
    };
  }

  /**
   * Create Entity objects for database insertion
   */
  async createEntitiesForAtom(
    atomId: string,
    text: string,
    sessionIdfWeights?: Map<string, number>
  ): Promise<Entity[]> {
    const extracted = await this.extractEntities(text);
    
    return extracted.entities.map(entity => {
      // Calculate weight using session-IDF if available
      let weight = entity.confidence;
      
      if (this.config.useSessionIdf && sessionIdfWeights) {
        const idfWeight = sessionIdfWeights.get(entity.text.toLowerCase()) || 1.0;
        weight = entity.confidence * Math.log(1 + idfWeight);
      }
      
      return {
        atom_id: atomId,
        entity: entity.text,
        kind: entity.kind,
        weight: Math.max(weight, this.config.minWeight),
      };
    });
  }
}

/**
 * Utility function to create a default entity extractor
 */
export function createEntityExtractor(config?: Partial<EntityExtractionConfig>): EntityExtractor {
  const fullConfig = {
    ...DEFAULT_ENTITY_CONFIG,
    ...config,
    patterns: {
      ...DEFAULT_ENTITY_CONFIG.patterns,
      ...config?.patterns,
    },
  };
  
  return new EntityExtractor(fullConfig);
}