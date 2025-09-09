use async_trait::async_trait;
use clap::Args;
use lethe_shared::Result;
use crate::utils::AppContext;
use super::Command;

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

    /// Show chunk metadata
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
        use lethe_domain::{EmbeddingServiceFactory, PipelineFactory, PipelineConfig};
        use lethe_infrastructure::{DatabaseManager, PgChunkRepository, PgEmbeddingRepository};
        use lethe_shared::QueryRequest;
        use std::sync::Arc;

        if !context.quiet {
            println!("🔍 Executing query: \"{}\"", self.query);
        }

        // Initialize database connection
        let db_url = context.database_url.as_ref()
            .ok_or("Database URL is required for querying")?;
        let db_manager = Arc::new(DatabaseManager::new(db_url).await?);

        // Initialize repositories
        let chunk_repo = Arc::new(PgChunkRepository::new(db_manager.pool().clone()));
        let embedding_repo = Arc::new(PgEmbeddingRepository::new(db_manager.pool().clone()));

        // Initialize services
        let embedding_service = Arc::new(EmbeddingServiceFactory::create_service(&context.config.embedding).await?);

        // Create pipeline configuration
        let pipeline_config = PipelineConfig {
            enable_hyde: self.enable_hyde || context.config.features.hyde_enabled,
            enable_query_understanding: true,
            enable_ml_prediction: true,
            max_candidates: context.config.retrieval.max_candidates.max(self.limit),
            rerank_enabled: self.enable_rerank || context.config.features.rerank_enabled,
            rerank_top_k: self.limit.min(20),
            timeout_seconds: context.config.timeouts.query_timeout as u64,
        };

        // Create query pipeline
        let pipeline = PipelineFactory::create_pipeline(
            pipeline_config,
            chunk_repo,
            embedding_service,
            None, // No LLM service for now
            None, // No reranking service for now
        );

        // Create query request
        let query_request = QueryRequest {
            query: self.query.clone(),
            session_id: self.session_id.clone(),
            limit: Some(self.limit),
            strategy: self.strategy.as_ref().map(|s| match s {
                SearchStrategy::Vector => lethe_shared::SearchStrategy::Vector,
                SearchStrategy::Bm25 => lethe_shared::SearchStrategy::BM25,
                SearchStrategy::Hybrid => lethe_shared::SearchStrategy::Hybrid,
                SearchStrategy::Auto => lethe_shared::SearchStrategy::Auto,
            }),
            min_similarity: self.min_similarity,
            enable_hyde: Some(self.enable_hyde),
            enable_rerank: Some(self.enable_rerank),
            context: None,
        };

        // Execute query
        let response = pipeline.query(&query_request).await?;

        // Display results
        self.display_results(&response, context)?;

        Ok(())
    }
}

impl QueryCommand {
    fn display_results(
        &self,
        response: &lethe_shared::QueryResponse,
        context: &AppContext,
    ) -> Result<()> {
        use crate::utils::OutputFormat;

        match context.output_format {
            OutputFormat::Json => {
                let json = serde_json::to_string_pretty(response)
                    .map_err(|e| format!("Failed to serialize response: {}", e))?;
                println!("{}", json);
            }
            OutputFormat::Yaml => {
                let yaml = serde_yaml::to_string(response)
                    .map_err(|e| format!("Failed to serialize response: {}", e))?;
                println!("{}", yaml);
            }
            OutputFormat::Table => {
                self.display_table_results(response)?;
            }
            OutputFormat::Pretty => {
                self.display_pretty_results(response)?;
            }
        }

        Ok(())
    }

    fn display_table_results(&self, response: &lethe_shared::QueryResponse) -> Result<()> {
        use tabled::{Table, Tabled};

        #[derive(Tabled)]
        struct ResultRow {
            #[tabled(rename = "Rank")]
            rank: usize,
            #[tabled(rename = "Score")]
            score: String,
            #[tabled(rename = "Strategy")]
            strategy: String,
            #[tabled(rename = "Text")]
            text: String,
        }

        let mut rows = Vec::new();
        for (i, candidate) in response.candidates.iter().enumerate() {
            rows.push(ResultRow {
                rank: i + 1,
                score: if self.show_scores {
                    format!("{:.4}", candidate.score)
                } else {
                    "---".to_string()
                },
                strategy: format!("{:?}", candidate.strategy),
                text: if candidate.chunk.text.len() > 100 {
                    format!("{}...", &candidate.chunk.text[..97])
                } else {
                    candidate.chunk.text.clone()
                },
            });
        }

        if rows.is_empty() {
            println!("No results found");
        } else {
            let table = Table::new(rows);
            println!("{}", table);
        }

        // Display metadata if requested
        if self.show_metadata && !response.candidates.is_empty() {
            println!("\n📊 Query Statistics:");
            if let Some(duration) = response.duration_ms {
                println!("   ⏱️  Query time: {}ms", duration);
            }
            if let Some(strategy) = &response.strategy_used {
                println!("   🎯 Strategy used: {:?}", strategy);
            }
            if response.hyde_expanded {
                println!("   🔄 HyDE expansion: enabled");
            }
        }

        Ok(())
    }

    fn display_pretty_results(&self, response: &lethe_shared::QueryResponse) -> Result<()> {
        if response.candidates.is_empty() {
            println!("❌ No results found for query: \"{}\"", self.query);
            return Ok(());
        }

        println!("✅ Found {} result(s):", response.candidates.len());
        println!();

        for (i, candidate) in response.candidates.iter().enumerate() {
            println!("🔍 Result #{}", i + 1);
            if self.show_scores {
                println!("   📊 Score: {:.4}", candidate.score);
            }
            println!("   🎯 Strategy: {:?}", candidate.strategy);
            println!("   📝 Text: {}", candidate.chunk.text);
            
            if self.show_metadata && candidate.chunk.meta.is_some() {
                println!("   🏷️  Metadata: {}", 
                    serde_json::to_string_pretty(candidate.chunk.meta.as_ref().unwrap())
                        .unwrap_or_else(|_| "Invalid JSON".to_string())
                );
            }
            
            println!();
        }

        // Display query statistics
        if self.show_metadata {
            println!("📊 Query Statistics:");
            if let Some(duration) = response.duration_ms {
                println!("   ⏱️  Query time: {}ms", duration);
            }
            if let Some(strategy) = &response.strategy_used {
                println!("   🎯 Strategy used: {:?}", strategy);
            }
            if response.hyde_expanded {
                println!("   🔄 HyDE expansion: enabled");
            }
        }

        Ok(())
    }
}