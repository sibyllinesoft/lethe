pub mod benchmark_execution;
pub mod benchmark_runner;
pub mod config;
pub mod delta_u_training;
pub mod determinism;
pub mod json_canon;
pub mod lambda_mu_controller;
pub mod learning_loop;
pub mod monitoring;
pub mod performance;
pub mod testing;
pub mod types;
pub mod v2_features;

// Production Hardening Modules
pub mod auto_dim_k2_system;
pub mod blind_repro_system;
pub mod grouped_dpp_enhancement;
pub mod tenant_capacity_frontiers;
pub mod v1_retirement_control;
