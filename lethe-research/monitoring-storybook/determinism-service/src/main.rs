use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod benchmark_execution;
mod benchmark_runner;
mod config;
mod delta_u_training;
mod determinism;
mod integration_tests;
mod json_canon;
mod lambda_mu_controller;
mod learning_loop;
mod monitoring;
mod performance;
mod testing;
mod types;
mod v2_features;

use config::Config;
use determinism::DeterminismSentinel;
use learning_loop::LearningLoopService;
use monitoring::DashboardState;
use benchmark_execution::*;

#[derive(Clone)]
pub struct AppState {
    sentinel: Arc<DeterminismSentinel>,
    dashboard: Arc<DashboardState>,
    learning_loop: Arc<LearningLoopService>,
    config: Arc<Config>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "determinism_service=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = Arc::new(Config::from_env()?);
    let sentinel = Arc::new(DeterminismSentinel::new(config.clone()).await?);
    let dashboard = Arc::new(DashboardState::new());
    let learning_loop = Arc::new(LearningLoopService::new(None));

    let app_state = AppState {
        sentinel,
        dashboard,
        learning_loop,
        config,
    };

    // Start background services
    let background_sentinel = app_state.sentinel.clone();
    tokio::spawn(async move {
        background_sentinel.run_background_validation().await;
    });

    // Initialize metrics
    metrics_exporter_prometheus::PrometheusBuilder::new()
        .install()
        .expect("failed to install prometheus recorder");

    let app = create_app(app_state);

    let listener = TcpListener::bind("0.0.0.0:3001").await?;
    tracing::info!("Determinism service starting on port 3001");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

fn create_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/determinism/replay/:slice_id", post(replay_slice))
        .route("/determinism/status", get(get_status))
        .route("/determinism/metrics", get(get_metrics))
        .route("/dashboard/data", get(get_dashboard_data))
        // Learning loop endpoints
        .route("/learning/process", post(process_changes))
        .route("/learning/metrics", get(get_learning_metrics))
        .route("/learning/ab-test/results", get(get_ab_test_results))
        .route("/learning/training", post(add_training_data))
        .route("/learning/performance", post(record_performance))
        // V2 Benchmark endpoints
        .route("/benchmark/v2/execute", post(execute_v2_benchmark))
        .route("/benchmark/v2/quick-test", post(execute_v2_quick_test))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive()),
        )
        .with_state(state)
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "healthy" }))
}

async fn replay_slice(
    State(state): State<AppState>,
    Path(slice_id): Path<String>,
) -> Result<Json<types::DeterminismReport>, StatusCode> {
    match state.sentinel.replay_slice_twice(&slice_id).await {
        Ok(report) => Ok(Json(report)),
        Err(e) => {
            tracing::error!("Failed to replay slice {}: {}", slice_id, e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_status(State(state): State<AppState>) -> Json<types::SystemStatus> {
    Json(state.sentinel.get_status().await)
}

async fn get_metrics(State(state): State<AppState>) -> Json<types::MetricsSnapshot> {
    Json(state.sentinel.get_metrics().await)
}

async fn get_dashboard_data(State(state): State<AppState>) -> Json<types::DashboardData> {
    Json(state.dashboard.get_data().await)
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received Ctrl+C, shutting down");
        },
        _ = terminate => {
            tracing::info!("Received terminate signal, shutting down");
        },
    }
}

// Learning Loop API handlers

async fn process_changes(
    State(state): State<AppState>,
    Json(request): Json<ProcessChangesRequest>,
) -> Result<Json<learning_loop::LearningLoopResult>, StatusCode> {
    match state.learning_loop.process_changes(request.changes, request.scenario_type).await {
        Ok(result) => Ok(Json(result)),
        Err(e) => {
            tracing::error!("Failed to process changes: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_learning_metrics(State(state): State<AppState>) -> Json<std::collections::HashMap<String, f64>> {
    Json(state.learning_loop.get_metrics().await)
}

async fn get_ab_test_results(State(state): State<AppState>) -> Json<learning_loop::ABTestResults> {
    Json(state.learning_loop.get_ab_test_results().await)
}

async fn add_training_data(
    State(state): State<AppState>,
    Json(request): Json<AddTrainingDataRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    state.learning_loop.add_training_datapoint(
        request.features,
        request.ground_truth_utility,
        request.scenario_type,
    ).await;
    
    Ok(Json(serde_json::json!({ "status": "training data added" })))
}

async fn record_performance(
    State(state): State<AppState>,
    Json(metric): Json<learning_loop::PerformanceMetric>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    state.learning_loop.record_performance(metric).await;
    
    Ok(Json(serde_json::json!({ "status": "performance recorded" })))
}

// Request types for learning loop APIs
#[derive(serde::Deserialize)]
struct ProcessChangesRequest {
    changes: Vec<types::TransformChangeV2>,
    scenario_type: types::ScenarioType,
}

#[derive(serde::Deserialize)]
struct AddTrainingDataRequest {
    features: types::V2Features,
    ground_truth_utility: f64,
    scenario_type: types::ScenarioType,
}

// V2 Benchmark endpoints

async fn execute_v2_benchmark(
    State(_state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match execute_v2_comprehensive_matrix().await {
        Ok(report) => {
            let summary = generate_benchmark_summary(&report);
            Ok(Json(serde_json::json!({
                "status": "completed",
                "report": report,
                "summary": summary
            })))
        },
        Err(e) => {
            tracing::error!("Failed to execute V2 benchmark: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn execute_v2_quick_test(
    State(_state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    match execute_quick_v2_test().await {
        Ok(()) => {
            Ok(Json(serde_json::json!({
                "status": "passed",
                "message": "Quick V2 test completed successfully"
            })))
        },
        Err(e) => {
            tracing::error!("Failed to execute quick V2 test: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}