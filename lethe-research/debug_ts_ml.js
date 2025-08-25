#!/usr/bin/env node

import { getMLPredictor } from '../ctx-run/packages/core/dist/ml-prediction/index.js';

async function debugMLIntegration() {
    console.log('🔍 Debug ML Integration');
    console.log('Current working directory:', process.cwd());
    
    // Test ML predictor
    const predictor = getMLPredictor({
        fusion_dynamic: true,
        plan_learned: true,
        models_dir: 'models',
        prediction_timeout_ms: 5000  // Increase timeout
    });
    
    console.log('🤖 Testing ML predictor initialization...');
    const initSuccess = await predictor.initialize();
    console.log('Initialization success:', initSuccess);
    
    console.log('🧪 Testing ML prediction...');
    try {
        const result = await predictor.predictParameters('TypeScript async error handling', {
            bm25_top1: 0.8,
            ann_top1: 0.7,
            overlap_ratio: 0.5
        });
        
        console.log('✅ ML Prediction successful:', result);
    } catch (error) {
        console.error('❌ ML Prediction failed:', error.message);
        console.error('Stack:', error.stack);
    }
    
    predictor.destroy();
}

debugMLIntegration().catch(console.error);