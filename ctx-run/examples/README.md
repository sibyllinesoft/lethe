# ctx-run Examples

This directory contains example files demonstrating various ctx-run features and evaluation suites.

## Files

### Sample Data
- **`sample-conversation.json`** - Example conversation data for testing ingestion and indexing

### Evaluation Suites  
- **`eval-code-heavy.json`** - Evaluation suite for technical, code-focused conversations
- **`eval-chatty-prose.json`** - Evaluation suite for conversational, prose-heavy discussions  
- **`eval-tool-results.json`** - Evaluation suite for conversations with tool outputs and data

## Using Evaluation Suites

### Basic Evaluation
Run an evaluation suite without parameter tuning:

```bash
npx ctx-run eval examples/eval-code-heavy.json -o results.json
```

### Evaluation with Parameter Tuning
Enable automatic parameter optimization:

```bash  
npx ctx-run eval examples/eval-code-heavy.json --tune --iterations 25 -o tuned-results.json
```

### Understanding Results

The evaluation command generates two output files:
- **JSON Report** (`results.json`) - Detailed results with per-query metrics
- **CSV Summary** (`results.csv`) - Tabular format for analysis in spreadsheets

#### Key Metrics Explained

**nDCG@k (Normalized Discounted Cumulative Gain)**
- Measures ranking quality of retrieved results
- Higher scores (0.0 to 1.0) indicate better relevance ordering
- @5 and @10 variants measure performance at different cutoff points

**Recall@k** 
- Measures what fraction of relevant chunks were retrieved
- Higher scores (0.0 to 1.0) indicate better coverage
- @5 and @10 variants for different result set sizes

**Latency Metrics**
- Mean, P50, and P90 latency in milliseconds
- Critical for understanding system responsiveness under load

### Parameter Tuning

When `--tune` is enabled, the system performs grid search over:
- **α (alpha)**: BM25 weighting parameter [0.5, 0.7, 1.0, 1.2, 1.5]
- **β (beta)**: Vector search weighting parameter [0.3, 0.5, 0.8, 1.0, 1.2]

The optimal configuration is automatically saved to your database.

## Creating Custom Evaluation Suites

### Suite Format

```json
{
  "name": "My Custom Evaluation Suite",
  "description": "Description of what this suite tests",
  "sessionId": "session-to-evaluate-against", 
  "queries": [
    {
      "id": "unique-query-id",
      "query": "Natural language question",
      "relevantChunks": ["chunk-id-1", "chunk-id-2"],
      "expectedSummary": "Optional expected summary", 
      "tags": ["tag1", "tag2"]
    }
  ]
}
```

### Best Practices

1. **Ground Truth**: Ensure `relevantChunks` accurately reflect chunks that should be retrieved
2. **Diversity**: Include queries of varying complexity and topics
3. **Realistic**: Use queries that match real user information needs
4. **Sufficient Size**: Include 10+ queries for meaningful statistics

### Query Categories

**Factual Queries**
- Direct information lookup
- Specific technical details
- Code examples and implementations

**Conceptual Queries**  
- Understanding relationships
- Comparing approaches
- Explaining principles

**Procedural Queries**
- Step-by-step processes
- How-to instructions  
- Troubleshooting guides

## Evaluation Workflow

1. **Prepare Data**: Ingest conversation data into a session
2. **Build Index**: Generate embeddings and search indexes
3. **Create Suite**: Define evaluation queries with ground truth
4. **Run Evaluation**: Execute suite with or without tuning
5. **Analyze Results**: Review metrics and identify improvements
6. **Iterate**: Refine configuration and re-evaluate

## Example Workflow

```bash
# Initialize workspace
npx ctx-run init

# Import conversation data  
npx ctx-run ingest -s tech-discussion --from conversation.json

# Build search indexes
npx ctx-run index -s tech-discussion

# Run evaluation with tuning
npx ctx-run eval examples/eval-code-heavy.json --tune -o results.json

# Start dev server to explore results
npx ctx-run serve --open
```

This workflow demonstrates the complete ctx-run pipeline from data ingestion to evaluation and optimization.