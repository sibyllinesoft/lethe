use super::Command;
use crate::utils::AppContext;
use async_trait::async_trait;
use clap::{Args, Subcommand};
use futures::stream::{self, StreamExt};
use lethe_shared::{LetheError, Result};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Args)]
pub struct BenchmarkCommand {
    #[command(subcommand)]
    action: BenchmarkAction,
}

#[derive(Debug, Subcommand)]
enum BenchmarkAction {
    /// Benchmark query performance
    Query {
        /// Number of queries to run
        #[arg(long, short, default_value = "100")]
        count: usize,
        /// Query text (or random if not provided)
        #[arg(long)]
        query: Option<String>,
        /// Enable concurrent execution
        #[arg(long)]
        concurrent: bool,
    },
    /// Benchmark embedding generation
    Embedding {
        /// Number of embeddings to generate
        #[arg(long, short, default_value = "100")]
        count: usize,
        /// Text length for test embeddings
        #[arg(long, default_value = "100")]
        text_length: usize,
    },
    /// Benchmark chunking performance
    Chunking {
        /// Test document size in KB
        #[arg(long, default_value = "10")]
        doc_size_kb: usize,
        /// Number of documents to process
        #[arg(long, short, default_value = "10")]
        count: usize,
    },
    /// Run all benchmarks
    All,
}

#[async_trait]
impl Command for BenchmarkCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        if !context.quiet {
            println!("🏁 Starting Lethe performance benchmarks...\n");
        }

        match &self.action {
            BenchmarkAction::Query {
                count,
                query,
                concurrent,
            } => {
                self.benchmark_queries(*count, query.clone(), *concurrent, context)
                    .await?;
            }
            BenchmarkAction::Embedding { count, text_length } => {
                self.benchmark_embeddings(*count, *text_length, context)
                    .await?;
            }
            BenchmarkAction::Chunking { doc_size_kb, count } => {
                self.benchmark_chunking(*doc_size_kb, *count, context)
                    .await?;
            }
            BenchmarkAction::All => {
                self.benchmark_embeddings(50, 100, context).await?;
                println!();
                self.benchmark_chunking(10, 10, context).await?;
                println!();
                self.benchmark_queries(50, None, false, context).await?;
            }
        }

        Ok(())
    }
}

impl BenchmarkCommand {
    async fn benchmark_queries(
        &self,
        count: usize,
        query: Option<String>,
        concurrent: bool,
        context: &AppContext,
    ) -> Result<()> {
        use lethe_domain::{
            EmbeddingRerankingService, EmbeddingServiceFactory, EnhancedQueryOptions,
            MLPredictionConfig, MLPredictionService, PipelineConfig, PipelineFactory,
        };
        use lethe_storage::ParquetCorpus;

        if count == 0 {
            return Err(LetheError::validation(
                "benchmark.count",
                "Query benchmark count must be greater than zero",
            ));
        }

        if !context.quiet {
            println!("🚀 Running query benchmark ({} requests)...", count);
        }

        let corpus = Arc::new(ParquetCorpus::new(&context.storage_root));
        corpus.health_check().await?;

        let embedding_config =
            super::to_domain_embedding_config(&context.resolved_config.embedding);
        let embedding_service = EmbeddingServiceFactory::create(&embedding_config).await?;

        let pipeline_config = PipelineConfig::from_resolved_config(&context.resolved_config);
        let enable_hyde_default = pipeline_config.enable_hyde;
        let rerank_enabled = pipeline_config.rerank_enabled;

        let ml_rules_path = context
            .resolved_config
            .ml
            .static_rules
            .path
            .as_ref()
            .map(|path| PathBuf::from(path));
        let ml_prediction_service = match MLPredictionService::from_rules_path(
            MLPredictionConfig::default(),
            ml_rules_path.as_deref(),
        ) {
            Ok(service) => service,
            Err(err) => {
                if !context.quiet {
                    eprintln!(
                        "⚠️  Failed to load ML strategy rules ({}). Using bundled defaults.",
                        err
                    );
                }
                MLPredictionService::default()
            }
        };

        let reranking_service = if rerank_enabled {
            Some(Arc::new(EmbeddingRerankingService::new(Arc::clone(
                &embedding_service,
            ))) as Arc<dyn lethe_domain::RerankingService>)
        } else {
            None
        };

        let document_repository: Arc<dyn lethe_domain::retrieval::DocumentRepository> =
            corpus.clone();
        let pipeline = Arc::new(PipelineFactory::create_pipeline(
            pipeline_config,
            document_repository,
            Arc::clone(&embedding_service),
            None,
            reranking_service,
            Some(ml_prediction_service),
        ));

        let sample_queries = vec![
            "Explain how vector search differs from keyword search.",
            "How does a bloom filter prevent false negatives?",
            "Walk me through indexing a repository with Lethe.",
            "What is HyDE query expansion and when should I use it?",
            "How can I improve retrieval quality for code snippets?",
        ];

        let queries: Vec<String> = (0..count)
            .map(|idx| match &query {
                Some(text) => text.clone(),
                None => sample_queries[idx % sample_queries.len()].to_string(),
            })
            .collect();

        let concurrency_level = if concurrent {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        } else {
            1
        };

        if concurrent && !context.quiet {
            println!(
                "⚙️  Executing with concurrency level {} (approximate)",
                concurrency_level
            );
        }

        let start_overall = Instant::now();
        let results = stream::iter(queries.into_iter().enumerate())
            .map(|(idx, text)| {
                let pipeline = Arc::clone(&pipeline);
                let session_id = format!("benchmark-{}", idx);
                async move {
                    let options = EnhancedQueryOptions {
                        session_id,
                        k: 5,
                        include_metadata: false,
                        enable_hyde: Some(enable_hyde_default),
                        override_strategy: None,
                        context: None,
                    };

                    let start = Instant::now();
                    match pipeline.process_query(&text, &options).await {
                        Ok(result) => {
                            let duration = start.elapsed();
                            Ok((duration, result.strategy_used))
                        }
                        Err(err) => Err((text, err.to_string())),
                    }
                }
            })
            .buffer_unordered(concurrency_level)
            .collect::<Vec<_>>()
            .await;

        let total_duration = start_overall.elapsed();

        let mut latencies = Vec::new();
        let mut strategies = Vec::new();
        let mut failures = Vec::new();

        for result in results {
            match result {
                Ok((duration, strategy)) => {
                    latencies.push(duration);
                    strategies.push(strategy);
                }
                Err((text, error)) => failures.push((text, error)),
            }
        }

        if latencies.is_empty() {
            return Err(LetheError::internal(
                "Query benchmark did not execute successfully; all attempts failed",
            ));
        }

        let avg_latency = average_duration(&latencies);
        let p95_latency = percentile_duration(&latencies, 0.95);
        let min_latency = latencies.iter().min().copied().unwrap_or_default();
        let max_latency = latencies.iter().max().copied().unwrap_or_default();
        let throughput = if total_duration.as_secs_f64() > 0.0 {
            latencies.len() as f64 / total_duration.as_secs_f64()
        } else {
            latencies.len() as f64
        };

        println!("📈 Query Benchmark Results:");
        println!(
            "   Total queries: {} (success: {}, failed: {})",
            count,
            latencies.len(),
            failures.len()
        );
        println!(
            "   Mode: {} (concurrency {})",
            if concurrent {
                "concurrent"
            } else {
                "sequential"
            },
            concurrency_level
        );
        println!("   Total time: {:?}", total_duration);
        println!(
            "   Avg latency: {:.2} ms",
            avg_latency.as_secs_f64() * 1000.0
        );
        println!(
            "   P95 latency: {:.2} ms",
            p95_latency.as_secs_f64() * 1000.0
        );
        println!(
            "   Min latency: {:.2} ms",
            min_latency.as_secs_f64() * 1000.0
        );
        println!(
            "   Max latency: {:.2} ms",
            max_latency.as_secs_f64() * 1000.0
        );
        println!("   Throughput: {:.2} req/s", throughput);

        if !strategies.is_empty() {
            let mut counts = std::collections::HashMap::new();
            for strategy in strategies {
                *counts.entry(strategy).or_insert(0usize) += 1;
            }
            println!("   Strategy usage:");
            for (strategy, occurrences) in counts {
                println!("     - {:?}: {}", strategy, occurrences);
            }
        }

        if !failures.is_empty() {
            println!("⚠️  {} queries failed during benchmarking", failures.len());
            if !context.quiet {
                for (idx, (text, error)) in failures.iter().take(3).enumerate() {
                    println!("     {}. '{}' → {}", idx + 1, text, error);
                }
                if failures.len() > 3 {
                    println!("     ... additional failures omitted ...");
                }
            }
        }

        Ok(())
    }

    async fn benchmark_embeddings(
        &self,
        count: usize,
        text_length: usize,
        context: &AppContext,
    ) -> Result<()> {
        use lethe_domain::EmbeddingServiceFactory;

        println!(
            "🧠 Benchmarking embedding generation ({} embeddings)...",
            count
        );

        let embedding_config =
            super::to_domain_embedding_config(&context.resolved_config.embedding);
        let embedding_service = EmbeddingServiceFactory::create(&embedding_config).await?;
        let test_text = "x".repeat(text_length);

        let start_time = std::time::Instant::now();

        for i in 0..count {
            if i % 10 == 0 && !context.quiet {
                println!("   Progress: {}/{}", i, count);
            }
            let _ = embedding_service.embed_single(&test_text).await?;
        }

        let duration = start_time.elapsed();
        let avg_time = duration.as_millis() as f64 / count as f64;

        println!("📊 Embedding Benchmark Results:");
        println!("   Total time: {:?}", duration);
        println!("   Average time per embedding: {:.2}ms", avg_time);
        println!("   Embeddings per second: {:.2}", 1000.0 / avg_time);

        Ok(())
    }

    async fn benchmark_chunking(
        &self,
        doc_size_kb: usize,
        count: usize,
        context: &AppContext,
    ) -> Result<()> {
        use chrono::Utc;
        use lethe_domain::{ChunkingConfig, ChunkingService};
        use lethe_shared::Message;
        use uuid::Uuid;

        println!(
            "📄 Benchmarking chunking performance ({} docs, {}KB each)...",
            count, doc_size_kb
        );

        let chunker = ChunkingService::new(ChunkingConfig {
            target_tokens: 1000,
            overlap: 200,
        });
        let test_doc = "This is a test document. ".repeat(doc_size_kb * 40); // ~1KB per 40 repetitions

        let start_time = std::time::Instant::now();

        for i in 0..count {
            if i % 5 == 0 && !context.quiet {
                println!("   Progress: {}/{}", i, count);
            }
            let message = Message {
                id: Uuid::new_v4(),
                session_id: "benchmark".to_string(),
                turn: i as i32,
                role: "system".to_string(),
                text: test_doc.clone(),
                ts: Utc::now(),
                meta: None,
            };

            let _ = chunker.chunk_message(&message)?;
        }

        let duration = start_time.elapsed();
        let avg_time = duration.as_millis() as f64 / count as f64;

        println!("📊 Chunking Benchmark Results:");
        println!("   Total time: {:?}", duration);
        println!("   Average time per document: {:.2}ms", avg_time);
        println!("   Documents per second: {:.2}", 1000.0 / avg_time);

        Ok(())
    }
}

fn average_duration(samples: &[Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::from_secs(0);
    }

    let total: f64 = samples.iter().map(|d| d.as_secs_f64()).sum();
    Duration::from_secs_f64(total / samples.len() as f64)
}

fn percentile_duration(samples: &[Duration], percentile: f64) -> Duration {
    if samples.is_empty() {
        return Duration::from_secs(0);
    }

    let mut values: Vec<f64> = samples.iter().map(|d| d.as_secs_f64()).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let clamped = percentile.clamp(0.0, 1.0);
    let idx = ((values.len() - 1) as f64 * clamped).round() as usize;
    Duration::from_secs_f64(values[idx])
}
