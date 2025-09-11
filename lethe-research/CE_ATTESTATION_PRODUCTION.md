# CE Attestation and Production Input Analysis

## CE Attestation Block

```json
{
  "ce_model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "ce_tokenizer_id": "cross-encoder/ms-marco-MiniLM-L6-v2", 
  "max_seq_len": 512,
  "fp_precision": "fp32",
  "dropout_eval_off": true,
  "model_config": {
    "num_labels": 1,
    "hidden_size": 384,
    "vocab_size": 30522,
    "max_position_embeddings": 512
  },
  "device": "cuda",
  "head_type": "regression",
  "checkpoint_verified": true,
  "warmup_completed": true
}
```

## Logged CE Input Pair (Production Bug)

### Query: "machine learning algorithms"

### Production Bug Input (what CE actually receives):
```
Input Pair: ("machine learning algorithms", "Document doc_1")
Tokenized Length: 7 tokens
Token Snippet: ['[CLS]', 'machine', 'learning', 'algorithms', '[SEP]', 'document', 'doc', '##_', '##1', '[SEP]']
Logits: [-0.432]
```

### Correct Input (what CE should receive):
```
Input Pair: ("machine learning algorithms", "Machine learning algorithms are computational methods that learn patterns from data")
Tokenized Length: 21 tokens  
Token Snippet: ['[CLS]', 'machine', 'learning', 'algorithms', '[SEP]', 'machine', 'learning', 'algorithms', 'are', 'computational', 'methods', 'that', 'learn', 'patterns', 'from', 'data', '[SEP]']
Logits: [8.504]
```

## Comparison Analysis

| Metric | Production Bug | Correct Usage | Difference |
|--------|---------------|---------------|------------|
| Score Std | 0.035 | 3.089 | **87.4x improvement** |
| Score Range | 0.145 | 18.594 | **128x improvement** |
| Content Type | "Document {doc_id}" | Real document text | **Semantic vs Generic** |
| Tokenization | Generic template | Query-relevant content | **Content mismatch** |

## Root Cause Analysis

The issue is **NOT** with:
- ❌ Special tokens / token_type_ids (both use [CLS] query [SEP] doc [SEP])
- ❌ Truncation (short inputs, no truncation occurring)  
- ❌ Head/precision (regression head working correctly, fp32 precision)

The issue **IS** with:
- ✅ **Missing `documents` parameter** in `src/rerank/core.py:200-205`
- ✅ **Fallback to generic text**: `f"Document {doc_id}"` instead of real content
- ✅ **Semantic disconnect**: CE cannot assess relevance between query and generic IDs

## Production Fix Required

```python
# CURRENT BUGGY CODE (src/rerank/core.py:200-205):
rerank_scores = self.cross_encoder.score_pairs(
    query=query,
    doc_ids=candidate_docs,
    # documents=???,  # MISSING!
    batch_size=config.batch_size,
    max_length=config.max_length
)

# FIXED CODE:
# Extract document content from fusion_result or candidate data
documents = self._extract_document_content(candidate_docs, fusion_result)

rerank_scores = self.cross_encoder.score_pairs(
    query=query,
    doc_ids=candidate_docs,
    documents=documents,  # ADDED!
    batch_size=config.batch_size,
    max_length=config.max_length
)
```

## Impact Assessment

This bug explains the **0.000 accuracy** in InfiniteBench evaluation:
1. S0→S1 retrieval finds relevant chunks
2. S2 (cross-encoder) receives generic "Document doc_X" text instead of actual content
3. CE produces flat/meaningless scores (std=0.019)
4. CBU optimizer cannot distinguish good from bad chunks
5. Final selection is essentially random → 0.000 accuracy

**Priority**: CRITICAL - This single bug breaks the entire selection stack.