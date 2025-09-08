import { describe, it, expect, vi } from 'vitest';

// Mock external dependencies to avoid complex setup
vi.mock('@lethe/embeddings', () => ({
  embeddings: {
    embed: vi.fn().mockResolvedValue([0.1, 0.2, 0.3])
  }
}));

describe('Diversifier Basic Tests', () => {
  it('should import diversifier module without errors', async () => {
    const diversifierModule = await import('./index.js');
    expect(diversifierModule).toBeDefined();
  });

  it('should have EntityCoverageDiversifier class', async () => {
    const { EntityCoverageDiversifier } = await import('./index.js');
    expect(EntityCoverageDiversifier).toBeDefined();
    expect(typeof EntityCoverageDiversifier).toBe('function');
  });

  it('should have SemanticDiversifier class', async () => {
    const { SemanticDiversifier } = await import('./index.js');
    expect(SemanticDiversifier).toBeDefined();
    expect(typeof SemanticDiversifier).toBe('function');
  });

  it('should instantiate EntityCoverageDiversifier', async () => {
    const { EntityCoverageDiversifier } = await import('./index.js');
    const diversifier = new EntityCoverageDiversifier();
    expect(diversifier).toBeDefined();
    expect(diversifier.name).toBe('entity-coverage');
  });

  it('should instantiate SemanticDiversifier', async () => {
    const { SemanticDiversifier } = await import('./index.js');
    const diversifier = new SemanticDiversifier();
    expect(diversifier).toBeDefined();
    expect(diversifier.name).toBe('semantic');
  });

  it('should handle empty candidates array', async () => {
    vi.spyOn(console, 'log').mockImplementation(() => {});
    
    const { EntityCoverageDiversifier } = await import('./index.js');
    const diversifier = new EntityCoverageDiversifier();
    
    const result = await diversifier.diversify([], 5);
    expect(result).toEqual([]);
  });

  it('should return same candidates when k >= length', async () => {
    vi.spyOn(console, 'log').mockImplementation(() => {});
    
    const { EntityCoverageDiversifier } = await import('./index.js');
    const diversifier = new EntityCoverageDiversifier();
    
    const candidates = [
      { id: '1', text: 'test', score: 0.9, metadata: { source: 'src', kind: 'text' } }
    ];
    
    const result = await diversifier.diversify(candidates, 5);
    expect(result).toEqual(candidates);
  });
});