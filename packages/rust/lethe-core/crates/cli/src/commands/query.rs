use super::Command;
use crate::utils::AppContext;
use async_trait::async_trait;
use clap::Args;
use lethe_shared::Result;

#[derive(Debug, Args)]
pub struct QueryCommand {
    /// Query text
    #[arg(required = true)]
    query: String,

    /// Session ID to query within
    #[arg(long, short)]
    session_id: Option<String>,

    /// Number of results to return
    #[arg(long, short = 'n', default_value = "5")]
    limit: usize,

    /// Enable HyDE query expansion
    #[arg(long)]
    enable_hyde: bool,

    /// Search strategy to use
    #[arg(long, value_enum)]
    strategy: Option<SearchStrategy>,

    /// Minimum similarity threshold
    #[arg(long)]
    min_similarity: Option<f32>,

    /// Enable result reranking
    #[arg(long)]
    enable_rerank: bool,

    /// Show detailed scoring information
    #[arg(long)]
    show_scores: bool,

    /// Show additional metadata and diagnostics
    #[arg(long)]
    show_metadata: bool,
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum SearchStrategy {
    Vector,
    Bm25,
    Hybrid,
    Auto,
}

#[async_trait]
impl Command for QueryCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_domain::{
            corpus::ParquetCorpus, EmbeddingServiceFactory, EnhancedQueryOptions,
            EnhancedQueryResult, PipelineConfig, PipelineFactory, RetrievalStrategy,
        };
        use std::sync::Arc;

        if !context.quiet {
            println!("🔍 Executing query: \"{}\"", self.query);
        }
        let corpus = Arc::new(ParquetCorpus::new(&context.storage_root));
        corpus.health_check().await?;
        let embedding_config = super::to_domain_embedding_config(&context.config.embedding);
        let embedding_service = EmbeddingServiceFactory::create(&embedding_config).await?;

        let features = context
            .config
            .features
            .as_ref()
            .cloned()
            .unwrap_or_default();

        let pipeline_config = PipelineConfig {
            enable_hyde: self.enable_hyde || features.enable_hyde,
            enable_query_understanding: features.enable_query_understanding,
            enable_ml_prediction: features.enable_ml_prediction,
            max_candidates: self.limit.max(10),
            rerank_enabled: self.enable_rerank,
            rerank_top_k: self.limit.min(20),
            timeout_seconds: 30,
        };
        let repo: Arc<dyn lethe_domain::retrieval::DocumentRepository> = corpus.clone();
        let pipeline =
            PipelineFactory::create_pipeline(pipeline_config, repo, embedding_service, None, None);

        let options = EnhancedQueryOptions {
            session_id: self
                .session_id
                .clone()
                .unwrap_or_else(|| "default".to_string()),
            k: self.limit,
            include_metadata: self.show_metadata,
            enable_hyde: Some(self.enable_hyde),
            override_strategy: self.strategy.as_ref().map(|strategy| match strategy {
                SearchStrategy::Vector => RetrievalStrategy::VectorOnly,
                SearchStrategy::Bm25 => RetrievalStrategy::BM25Only,
                SearchStrategy::Hybrid => RetrievalStrategy::Hybrid,
                SearchStrategy::Auto => RetrievalStrategy::Adaptive,
            }),
            context: None,
        };

        let mut result: EnhancedQueryResult = pipeline.process_query(&self.query, &options).await?;

        if let Some(threshold) = self.min_similarity {
            let threshold = threshold as f64;
            result
                .candidates
                .retain(|candidate| candidate.score >= threshold);
        }

        self.display_results(&result, context)?;
        Ok(())
    }
}

impl QueryCommand {
    fn display_results(
        &self,
        result: &lethe_domain::EnhancedQueryResult,
        context: &AppContext,
    ) -> Result<()> {
        use crate::utils::OutputFormat;

        match context.output_format {
            OutputFormat::Json => {
                let json = serde_json::to_string_pretty(result)?;
                println!("{}", json);
            }
            OutputFormat::Yaml => {
                let yaml = serde_yaml::to_string(result).map_err(|e| {
                    lethe_shared::LetheError::internal(format!(
                        "Failed to serialize response: {}",
                        e
                    ))
                })?;
                println!("{}", yaml);
            }
            OutputFormat::Table => self.display_table_results(result)?,
            OutputFormat::Pretty => self.display_pretty_results(result)?,
        }

        Ok(())
    }

    fn display_table_results(&self, result: &lethe_domain::EnhancedQueryResult) -> Result<()> {
        use tabled::{Table, Tabled};

        #[derive(Tabled)]
        struct Row {
            #[tabled(rename = "Rank")]
            rank: usize,
            #[tabled(rename = "Score")]
            score: String,
            #[tabled(rename = "Kind")]
            kind: String,
            #[tabled(rename = "Snippet")]
            snippet: String,
        }

        let mut rows = Vec::new();
        for (idx, candidate) in result.candidates.iter().enumerate() {
            let snippet = candidate
                .text
                .as_deref()
                .map(|text| {
                    if text.len() > 100 {
                        format!("{}...", &text[..97])
                    } else {
                        text.to_string()
                    }
                })
                .unwrap_or_else(|| "<no text>".to_string());

            rows.push(Row {
                rank: idx + 1,
                score: if self.show_scores {
                    format!("{:.4}", candidate.score)
                } else {
                    "--".to_string()
                },
                kind: candidate.kind.as_deref().unwrap_or("unknown").to_string(),
                snippet,
            });
        }

        if rows.is_empty() {
            println!("No results found");
        } else {
            println!("🧠 Strategy: {:?}", result.strategy_used);
            println!("⏱️  Processing time: {}ms", result.processing_time_ms);
            println!(
                "📊 Total candidates considered: {}",
                result.total_candidates_found
            );
            println!(
                "{}
",
                Table::new(rows)
            );
        }

        Ok(())
    }

    fn display_pretty_results(&self, result: &lethe_domain::EnhancedQueryResult) -> Result<()> {
        use crate::utils::helpers::truncate_text;

        if result.candidates.is_empty() {
            println!("❌ No results found for query: \"{}\"", self.query);
            return Ok(());
        }

        println!(
            "✅ Retrieved {} result(s) in {}ms",
            result.candidates.len(),
            result.processing_time_ms
        );
        println!("🎯 Strategy used: {:?}", result.strategy_used);
        println!();

        for (idx, candidate) in result.candidates.iter().enumerate() {
            println!("{}️⃣  Result #{}", idx + 1, idx + 1);
            if self.show_scores {
                println!("   📊 Score: {:.4}", candidate.score);
            }
            println!("   📄 Document ID: {}", candidate.doc_id);
            if let Some(kind) = &candidate.kind {
                println!("   🏷️  Kind: {}", kind);
            }
            if let Some(text) = &candidate.text {
                println!("   📝 Snippet: {}", truncate_text(text, 200));
            }
            println!();
        }

        if self.show_metadata {
            println!("📦 Context Pack:");
            println!("   🆔 ID: {}", result.context_pack.id);
            println!("   🔍 Query: {}", result.context_pack.query);
            println!(
                "   📑 Summary: {}",
                truncate_text(&result.context_pack.summary, 200)
            );
            println!("   🧩 Key entities: {:?}", result.context_pack.key_entities);
            println!("   ✅ Claims: {}", result.context_pack.claims.len());
            println!(
                "   ⚔️  Contradictions: {}",
                result.context_pack.contradictions.len()
            );

            if let Some(understanding) = &result.query_understanding {
                println!("\n🧠 Query Understanding:");
                println!("   🧭 Type: {:?}", understanding.query_type);
                println!("   🎯 Intent: {:?}", understanding.intent);
                println!("   🧮 Complexity: {:?}", understanding.complexity);
                println!("   🗂️ Domain: {:?}", understanding.domain);
                println!("   🔑 Keywords: {:?}", understanding.keywords);
                println!("   📈 Confidence: {:.2}", understanding.confidence);
                if !understanding.entities.is_empty() {
                    println!("   🧵 Entities: {}", understanding.entities.len());
                }
            }

            if let Some(prediction) = &result.ml_prediction {
                println!("\n📈 ML Prediction:");
                println!(
                    "   🎯 Suggested strategy: {:?}",
                    prediction.prediction.strategy
                );
                println!("   🤝 Confidence: {:.2}", prediction.model_confidence);
                println!(
                    "   📝 Explanation: {}",
                    truncate_text(&prediction.explanation, 200)
                );
            }
        }

        Ok(())
    }
}
