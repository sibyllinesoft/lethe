# LetheBench Enhanced Dataset v3.0.0

## Overview
This enhanced dataset was constructed deterministically using seed 42 with comprehensive quality assurance to ensure perfect reproducibility and research-grade quality.

## Key Features
- **Domain-Specific Generation**: Realistic queries using specialized domain builders
- **Quality Assurance**: Comprehensive validation with IAA ≥0.7 κ
- **Stratified Splits**: Balanced train/dev/test splits maintaining domain distribution
- **Full Provenance**: Complete construction tracking and verification

## Statistics
- **Total Queries**: 436
- **Creation Time**: 2025-08-25 07:07:28.851106+00:00
- **Average Quality Score**: 0.921
- **Content Hash**: `530423b96ef33a88...`

## Domain Distribution
- **code_heavy**: 112 queries (25.7%)
- **chatty_prose**: 186 queries (42.7%)
- **tool_results**: 138 queries (31.7%)

## Dataset Splits
- **Train**: 260 queries (59.6%)
- **Dev**: 86 queries (19.7%)
- **Test**: 90 queries (20.6%)

## Quality Metrics
- **Average Quality Score**: 0.921
- **Quality Range**: 0.800 - 1.000
- **Standard Deviation**: 0.074
- **Validation Errors**: 0

## Files
- `queries.jsonl` - All queries in JSON Lines format
- `queries_structured.json` - All queries in structured JSON format  
- `splits/` - Train/dev/test splits in JSONL and CSV formats
- `domain_statistics.json` - Detailed domain statistics
- `quality_audit.json` - Comprehensive quality audit results
- `MANIFEST.json` - Complete dataset manifest with full provenance
- `manifest.csv` - Human-readable manifest summary

## Verification
To verify dataset integrity, check that the content hash matches: `530423b96ef33a88ff0db3ccc4398d4c776ebd33eaaf6ba07cb82a73fe945159`

## Reproducibility
This enhanced dataset can be exactly reproduced using:
```python
builder = DeterministicDatasetBuilder(
    target_queries=436, 
    seed=42,
    quality_threshold=0.8,
    iaa_threshold=0.7
)
manifest = builder.build_dataset()
```

## License
Apache-2.0 - Commercial use allowed, attribution required.
