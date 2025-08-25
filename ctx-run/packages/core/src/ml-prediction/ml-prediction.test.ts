import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { MLPredictor, MLConfig } from './index.js';

describe('MLPredictor HTTP Client', () => {
  let predictor: MLPredictor;

  beforeEach(() => {
    predictor = new MLPredictor({
      fusion_dynamic: true,
      plan_learned: true,
      service_url: 'http://127.0.0.1:8080',
      service_enabled: true,
      prediction_timeout_ms: 5000
    });
  });

  afterEach(() => {
    predictor.destroy();
  });

  describe('Configuration', () => {
    it('should use fallback configuration when service disabled', async () => {
      const fallbackPredictor = new MLPredictor({
        fusion_dynamic: false,
        plan_learned: false,
        service_enabled: false
      });

      const prediction = await fallbackPredictor.predictParameters('test query');
      
      expect(prediction.alpha).toBe(0.7);
      expect(prediction.beta).toBe(0.5);
      expect(prediction.plan).toBe('exploit');
      expect(prediction.model_loaded).toBe(false);
      
      fallbackPredictor.destroy();
    });

    it('should handle missing service gracefully', async () => {
      const unreachablePredictor = new MLPredictor({
        fusion_dynamic: true,
        plan_learned: true,
        service_url: 'http://127.0.0.1:9999', // Non-existent service
        service_enabled: true,
        prediction_timeout_ms: 1000
      });

      const prediction = await unreachablePredictor.predictParameters('test query');
      
      // Should fall back to defaults when service unavailable
      expect(prediction.alpha).toBe(0.7);
      expect(prediction.beta).toBe(0.5);
      expect(prediction.plan).toBe('exploit');
      expect(prediction.model_loaded).toBe(false);
      
      unreachablePredictor.destroy();
    });
  });

  describe('Caching', () => {
    it('should cache predictions', async () => {
      const query = 'test caching query';
      const context = { bm25_top1: 0.8 };

      // First call
      const start1 = Date.now();
      const prediction1 = await predictor.predictParameters(query, context);
      const time1 = Date.now() - start1;

      // Second call should be faster (cached)
      const start2 = Date.now();
      const prediction2 = await predictor.predictParameters(query, context);
      const time2 = Date.now() - start2;

      // Results should be identical
      expect(prediction2.alpha).toBe(prediction1.alpha);
      expect(prediction2.beta).toBe(prediction1.beta);
      expect(prediction2.plan).toBe(prediction1.plan);

      // Cache stats should show one entry
      const stats = predictor.getCacheStats();
      expect(stats.size).toBe(1);
    });

    it('should clear cache correctly', async () => {
      await predictor.predictParameters('test query 1');
      await predictor.predictParameters('test query 2');
      
      let stats = predictor.getCacheStats();
      expect(stats.size).toBe(2);
      
      predictor.clearCache();
      
      stats = predictor.getCacheStats();
      expect(stats.size).toBe(0);
    });
  });

  describe('Parameter Validation', () => {
    it('should return valid alpha and beta ranges', async () => {
      const prediction = await predictor.predictParameters('test query');
      
      expect(prediction.alpha).toBeGreaterThanOrEqual(0);
      expect(prediction.alpha).toBeLessThanOrEqual(1);
      expect(prediction.beta).toBeGreaterThanOrEqual(0);
      expect(prediction.beta).toBeLessThanOrEqual(1);
    });

    it('should return valid plan types', async () => {
      const queries = [
        'How to implement binary search?',
        'React component not rendering TypeError',
        'getUserById function implementation'
      ];

      for (const query of queries) {
        const prediction = await predictor.predictParameters(query);
        expect(['explore', 'verify', 'exploit']).toContain(prediction.plan);
      }
    });
  });

  describe('Context Handling', () => {
    it('should handle different context parameters', async () => {
      const contexts = [
        {},
        { bm25_top1: 0.9 },
        { ann_top1: 0.8, overlap_ratio: 0.6 },
        { hyde_k: 3, contradictions: 2 }
      ];

      for (const context of contexts) {
        const prediction = await predictor.predictParameters('test query', context);
        
        expect(prediction).toBeDefined();
        expect(typeof prediction.alpha).toBe('number');
        expect(typeof prediction.beta).toBe('number');
        expect(typeof prediction.plan).toBe('string');
      }
    });
  });
});