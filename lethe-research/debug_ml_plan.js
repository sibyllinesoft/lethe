#!/usr/bin/env node

import { getMLPredictor } from '../ctx-run/packages/core/dist/ml-prediction/index.js';

async function debugMLPlan() {
    console.log('🔍 Debug ML Plan Selection');
    
    // Test ML predictor directly
    const predictor = getMLPredictor({
        fusion_dynamic: true,
        plan_learned: true,
        models_dir: 'models',
        prediction_timeout_ms: 5000
    });
    
    console.log('🤖 Testing ML predictor...');
    try {
        const result = await predictor.predictParameters('TypeScript async error handling', {
            bm25_top1: 0.8,
            ann_top1: 0.7,
            overlap_ratio: 0.5
        });
        
        console.log('✅ Full ML Result:', JSON.stringify(result, null, 2));
        console.log('Plan loaded:', result.model_loaded);
        console.log('Plan value:', result.plan);
        console.log('Plan type:', typeof result.plan);
        
    } catch (error) {
        console.error('❌ ML Prediction failed:', error.message);
    }
    
    predictor.destroy();
}

debugMLPlan().catch(console.error);