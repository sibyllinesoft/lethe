# Verification and Testing Infrastructure

"""
Comprehensive verification framework for Lethe research infrastructure.

This module provides:
- Property-based testing using Hypothesis
- Mutation testing for code quality assessment  
- Fuzzing infrastructure for robustness testing
- Oracle system for correctness verification
- Metamorphic testing for algorithmic validation

Components:
- properties/: Property-based and metamorphic testing
- mutation/: Mutation testing framework
- fuzzing/: Fuzzing infrastructure
- oracles/: Oracle verification system

Usage:
    # Property-based testing
    from verification.properties.test_suite import PropertyTestSuite
    suite = PropertyTestSuite()
    results = suite.run_all_tests()
    
    # Mutation testing
    from verification.mutation.test_mutations import MutationTestRunner
    runner = MutationTestRunner(['src'], 'tests')
    results = runner.run_mutation_tests()
    
    # Fuzzing
    from verification.fuzzing.test_fuzz import FuzzOrchestrator
    orchestrator = FuzzOrchestrator(target_function, generators)
    results = orchestrator.run_campaign()
    
    # Oracle verification
    from verification.oracles.test_oracles import OracleManager
    manager = OracleManager()
    results = manager.run_verification_batch(verifications)

Quality Thresholds:
- Mutation score: ≥ 0.80
- Property test coverage: ≥ 0.70  
- Oracle verification confidence: ≥ 0.85
"""

from pathlib import Path

# Export main components
__all__ = [
    'PropertyTestSuite',
    'MutationTestRunner', 
    'FuzzOrchestrator',
    'OracleManager'
]

# Version info
__version__ = "1.0.0"

# Verification thresholds (from NeurIPS requirements)
MUTATION_SCORE_THRESHOLD = 0.80
PROPERTY_COVERAGE_THRESHOLD = 0.70
ORACLE_CONFIDENCE_THRESHOLD = 0.85

def get_verification_summary() -> str:
    """Get summary of verification framework capabilities."""
    return """
Lethe Research Verification Framework v1.0.0

Components:
✓ Property-based testing (Hypothesis)
✓ Mutation testing (≥0.80 score required)
✓ Fuzzing infrastructure 
✓ Oracle verification system
✓ Metamorphic testing
✓ Statistical validation

Targets:
- Deterministic dataset builders
- Retrieval algorithms  
- Scoring functions
- Statistical analysis
- Experimental pipelines

Quality Gates:
- Mutation score ≥ 80%
- Property coverage ≥ 70% 
- Oracle confidence ≥ 85%
- Zero high/critical SAST findings
"""