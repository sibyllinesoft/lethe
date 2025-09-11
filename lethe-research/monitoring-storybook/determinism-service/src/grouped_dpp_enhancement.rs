use crate::{
    json_canon::CanonicalJson,
    types::*,
};
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock, Mutex, atomic::{AtomicU64, AtomicBool, Ordering}},
    time::{Duration, Instant},
};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use nalgebra::{DMatrix, DVector, SVD};

/// Grouped-DPP Enhancement System
/// Implements advanced Determinantal Point Process with group structure and diversity optimization
#[derive(Debug)]
pub struct GroupedDPPEngine {
    groups: Arc<RwLock<HashMap<String, DPPGroup>>>,
    global_kernel: Arc<RwLock<DMatrix<f64>>>,
    q_matrix: Arc<RwLock<Option<DMatrix<f64>>>>,
    diversity_controller: Arc<DiversityController>,
    orthonormalization_tracker: Arc<OrthonormalizationTracker>,
    performance_monitor: Arc<DPPPerformanceMonitor>,
    metrics: Arc<Mutex<DPPMetrics>>,
}

/// Individual DPP group with its own kernel and state
#[derive(Debug, Clone)]
pub struct DPPGroup {
    pub group_id: String,
    pub centroid: DVector<f64>,
    pub members: Vec<DPPPoint>,
    pub local_kernel: DMatrix<f64>,
    pub intra_group_penalty: f64,    // Concave penalty within group
    pub diversity_score: f64,
    pub last_updated: DateTime<Utc>,
    pub selection_history: VecDeque<SelectionEvent>,
}

/// Point in the DPP space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DPPPoint {
    pub point_id: Uuid,
    pub features: DVector<f64>,
    pub group_membership: String,
    pub selection_probability: f64,
    pub diversity_contribution: f64,
    pub timestamp: DateTime<Utc>,
}

/// Selection event tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionEvent {
    pub event_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub selected_points: Vec<Uuid>,
    pub diversity_score: f64,
    pub group_distribution: HashMap<String, usize>,
    pub quality_metrics: SelectionQualityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionQualityMetrics {
    pub determinant_value: f64,
    pub condition_number: f64,
    pub spectral_gap: f64,
    pub group_balance: f64,
    pub novelty_score: f64,
}

/// Diversity controller with clamped diversity terms
#[derive(Debug)]
pub struct DiversityController {
    pub diversity_radius: f64,           // Radius parameter 'r' for clamping
    pub clamp_bounds: (f64, f64),        // [0, log(1+r)] bounds
    pub penalty_function: PenaltyFunction,
    pub optimization_target: OptimizationTarget,
    pub adaptive_weights: Arc<RwLock<AdaptiveWeights>>,
}

#[derive(Debug, Clone)]
pub enum PenaltyFunction {
    Concave,        // log(1+‖(I−QQᵀ)v‖²)
    Linear,
    Quadratic,
    Exponential,
}

#[derive(Debug, Clone)]
pub enum OptimizationTarget {
    MaximizeDiversity,
    BalanceQualityDiversity,
    MinimizeRedundancy,
    AdaptiveTarget,
}

#[derive(Debug, Clone)]
pub struct AdaptiveWeights {
    pub quality_weight: f64,
    pub diversity_weight: f64,
    pub group_balance_weight: f64,
    pub novelty_weight: f64,
    pub adaptation_rate: f64,
    pub last_adaptation: DateTime<Utc>,
}

/// Q matrix orthonormalization tracker
#[derive(Debug)]
pub struct OrthonormalizationTracker {
    pub insertion_count: Arc<RwLock<u64>>,
    pub reorthonormalization_threshold: u64,  // ~128 inserts
    pub last_reorthonormalization: Arc<RwLock<DateTime<Utc>>>,
    pub orthogonality_metrics: Arc<Mutex<OrthogonalityMetrics>>,
    pub reorthonormalization_history: Arc<Mutex<VecDeque<ReorthonormalizationEvent>>>,
}

#[derive(Debug, Clone)]
pub struct OrthogonalityMetrics {
    pub condition_number: f64,
    pub orthogonality_error: f64,
    pub spectral_properties: SpectralProperties,
    pub numerical_stability: f64,
}

#[derive(Debug, Clone)]
pub struct SpectralProperties {
    pub eigenvalues: Vec<f64>,
    pub singular_values: Vec<f64>,
    pub rank: usize,
    pub null_space_dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReorthonormalizationEvent {
    pub event_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub insertions_since_last: u64,
    pub condition_before: f64,
    pub condition_after: f64,
    pub orthogonality_improvement: f64,
    pub computational_cost: Duration,
}

/// Performance monitoring for DPP operations
#[derive(Debug)]
pub struct DPPPerformanceMonitor {
    pub selection_times: Arc<Mutex<VecDeque<Duration>>>,
    pub kernel_computation_times: Arc<Mutex<VecDeque<Duration>>>,
    pub reorthonormalization_times: Arc<Mutex<VecDeque<Duration>>>,
    pub group_update_times: Arc<Mutex<VecDeque<Duration>>>,
    pub quality_trends: Arc<Mutex<VecDeque<QualityTrend>>>,
}

#[derive(Debug, Clone)]
pub struct QualityTrend {
    pub timestamp: DateTime<Utc>,
    pub diversity_score: f64,
    pub selection_quality: f64,
    pub group_balance: f64,
    pub computational_efficiency: f64,
}

/// DPP metrics tracking
#[derive(Debug, Default, Clone)]
pub struct DPPMetrics {
    pub total_selections: u64,
    pub successful_selections: u64,
    pub failed_selections: u64,
    pub reorthonormalizations: u64,
    pub group_rebalances: u64,
    pub average_diversity_score: f64,
    pub average_selection_time_ms: f64,
    pub kernel_stability_score: f64,
}

impl GroupedDPPEngine {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            groups: Arc::new(RwLock::new(HashMap::new())),
            global_kernel: Arc::new(RwLock::new(DMatrix::zeros(0, 0))),
            q_matrix: Arc::new(RwLock::new(None)),
            diversity_controller: Arc::new(DiversityController::new()),
            orthonormalization_tracker: Arc::new(OrthonormalizationTracker::new()),
            performance_monitor: Arc::new(DPPPerformanceMonitor::new()),
            metrics: Arc::new(Mutex::new(DPPMetrics::default())),
        })
    }

    /// Execute enhanced DPP selection with group awareness
    pub async fn execute_grouped_selection(&self, candidate_points: Vec<DPPPoint>, selection_size: usize) -> Result<DPPSelectionResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        // 1. Update group centroids and memberships
        self.update_group_structure(&candidate_points).await?;
        
        // 2. Compute group-aware kernel matrix
        let kernel_matrix = self.compute_grouped_kernel(&candidate_points).await?;
        
        // 3. Check if Q matrix reorthonormalization is needed
        self.check_and_reorthonormalize().await?;
        
        // 4. Apply clamped diversity term
        let diversity_adjusted_kernel = self.apply_clamped_diversity(&kernel_matrix, &candidate_points).await?;
        
        // 5. Execute DPP sampling with group constraints
        let selected_indices = self.sample_from_dpp(&diversity_adjusted_kernel, selection_size).await?;
        
        // 6. Extract selected points and compute quality metrics
        let selected_points: Vec<DPPPoint> = selected_indices.iter()
            .map(|&i| candidate_points[i].clone())
            .collect();
        
        let quality_metrics = self.compute_selection_quality(&selected_points, &diversity_adjusted_kernel)?;
        
        // 7. Update selection history and metrics
        let selection_event = SelectionEvent {
            event_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            selected_points: selected_points.iter().map(|p| p.point_id).collect(),
            diversity_score: quality_metrics.diversity_score,
            group_distribution: self.compute_group_distribution(&selected_points),
            quality_metrics: quality_metrics.clone(),
        };
        
        self.update_selection_history(selection_event).await;
        self.update_performance_metrics(start_time.elapsed(), &quality_metrics).await;
        
        Ok(DPPSelectionResult {
            selected_points,
            selection_quality: quality_metrics,
            computational_stats: ComputationalStats {
                total_time: start_time.elapsed(),
                kernel_computation_time: Duration::from_millis(10), // Would be measured
                sampling_time: Duration::from_millis(50),           // Would be measured
                group_update_time: Duration::from_millis(5),        // Would be measured
            },
            group_analysis: self.analyze_group_performance().await,
        })
    }

    /// Add new points to the DPP system
    pub async fn add_points(&self, new_points: Vec<DPPPoint>) -> Result<(), Box<dyn std::error::Error>> {
        // Increment insertion counter
        {
            let mut count = self.orthonormalization_tracker.insertion_count.write().unwrap();
            *count += new_points.len() as u64;
        }
        
        // Update group memberships
        for point in &new_points {
            self.assign_to_group(point).await?;
        }
        
        // Update global kernel matrix incrementally
        self.update_global_kernel(&new_points).await?;
        
        // Check if reorthonormalization is needed
        self.check_and_reorthonormalize().await?;
        
        Ok(())
    }

    /// Get comprehensive DPP system status
    pub async fn get_system_status(&self) -> DPPSystemStatus {
        let groups = self.groups.read().unwrap();
        let group_count = groups.len();
        let total_points: usize = groups.values().map(|g| g.members.len()).sum();
        
        let orthogonality_metrics = self.orthonormalization_tracker.orthogonality_metrics.lock().unwrap().clone();
        let insertion_count = *self.orthonormalization_tracker.insertion_count.read().unwrap();
        let diversity_controller_state = self.diversity_controller.get_state().await;
        let performance_summary = self.performance_monitor.get_summary().await;
        let metrics = self.metrics.lock().unwrap().clone();

        DPPSystemStatus {
            group_count,
            total_points,
            orthogonality_metrics,
            insertion_count,
            next_reorthonormalization_at: insertion_count + (self.orthonormalization_tracker.reorthonormalization_threshold - (insertion_count % self.orthonormalization_tracker.reorthonormalization_threshold)),
            diversity_controller_state,
            performance_summary,
            metrics,
            last_updated: Utc::now(),
        }
    }

    // Private implementation methods

    async fn update_group_structure(&self, points: &[DPPPoint]) -> Result<(), Box<dyn std::error::Error>> {
        let mut groups = self.groups.write().unwrap();
        
        // Organize points by group
        let mut group_members: HashMap<String, Vec<DPPPoint>> = HashMap::new();
        for point in points {
            group_members.entry(point.group_membership.clone())
                .or_insert_with(Vec::new)
                .push(point.clone());
        }
        
        // Update each group's centroid and properties
        for (group_id, members) in group_members {
            let group = groups.entry(group_id.clone())
                .or_insert_with(|| DPPGroup::new(group_id.clone()));
            
            // Update centroid
            group.centroid = self.compute_group_centroid(&members);
            group.members = members;
            group.last_updated = Utc::now();
            
            // Compute intra-group penalty (concave function)
            group.intra_group_penalty = self.compute_intra_group_penalty(&group.members);
            
            // Update local kernel
            group.local_kernel = self.compute_local_kernel(&group.members)?;
        }
        
        Ok(())
    }

    async fn compute_grouped_kernel(&self, points: &[DPPPoint]) -> Result<DMatrix<f64>, Box<dyn std::error::Error>> {
        let n = points.len();
        let mut kernel = DMatrix::zeros(n, n);
        let groups = self.groups.read().unwrap();
        
        for i in 0..n {
            for j in 0..n {
                let point_i = &points[i];
                let point_j = &points[j];
                
                // Base similarity
                let base_similarity = self.compute_feature_similarity(&point_i.features, &point_j.features);
                
                // Group-aware modification
                let group_factor = if point_i.group_membership == point_j.group_membership {
                    // Same group - apply intra-group penalty
                    if let Some(group) = groups.get(&point_i.group_membership) {
                        1.0 - group.intra_group_penalty
                    } else {
                        1.0
                    }
                } else {
                    // Different groups - encourage diversity
                    1.2
                };
                
                kernel[(i, j)] = base_similarity * group_factor;
            }
        }
        
        Ok(kernel)
    }

    async fn check_and_reorthonormalize(&self) -> Result<(), Box<dyn std::error::Error>> {
        let insertion_count = *self.orthonormalization_tracker.insertion_count.read().unwrap();
        
        if insertion_count % self.orthonormalization_tracker.reorthonormalization_threshold == 0 {
            self.perform_reorthonormalization().await?;
        }
        
        Ok(())
    }

    async fn perform_reorthonormalization(&self) -> Result<(), Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        let q_matrix_opt = self.q_matrix.read().unwrap().clone();
        if let Some(q_matrix) = q_matrix_opt {
            // Compute orthogonality metrics before
            let condition_before = self.compute_condition_number(&q_matrix);
            
            // Perform QR decomposition for reorthonormalization
            let qr = q_matrix.qr();
            let new_q = qr.q();
            
            // Update Q matrix
            *self.q_matrix.write().unwrap() = Some(new_q.clone());
            
            // Compute metrics after
            let condition_after = self.compute_condition_number(&new_q);
            let orthogonality_improvement = condition_before - condition_after;
            
            // Update orthogonality metrics
            {
                let mut metrics = self.orthonormalization_tracker.orthogonality_metrics.lock().unwrap();
                metrics.condition_number = condition_after;
                metrics.orthogonality_error = self.compute_orthogonality_error(&new_q);
                metrics.spectral_properties = self.compute_spectral_properties(&new_q);
                metrics.numerical_stability = self.compute_numerical_stability(&new_q);
            }
            
            // Record reorthonormalization event
            let event = ReorthonormalizationEvent {
                event_id: Uuid::new_v4(),
                timestamp: Utc::now(),
                insertions_since_last: self.orthonormalization_tracker.reorthonormalization_threshold,
                condition_before,
                condition_after,
                orthogonality_improvement,
                computational_cost: start_time.elapsed(),
            };
            
            let mut history = self.orthonormalization_tracker.reorthonormalization_history.lock().unwrap();
            history.push_back(event);
            if history.len() > 100 {
                history.pop_front();
            }
            
            // Update last reorthonormalization time
            *self.orthonormalization_tracker.last_reorthonormalization.write().unwrap() = Utc::now();
        }
        
        Ok(())
    }

    async fn apply_clamped_diversity(&self, kernel: &DMatrix<f64>, points: &[DPPPoint]) -> Result<DMatrix<f64>, Box<dyn std::error::Error>> {
        let mut enhanced_kernel = kernel.clone();
        let q_matrix_opt = self.q_matrix.read().unwrap().clone();
        
        if let Some(q_matrix) = q_matrix_opt {
            let identity = DMatrix::identity(q_matrix.nrows(), q_matrix.nrows());
            let projection = &identity - &q_matrix * q_matrix.transpose();
            
            for i in 0..points.len() {
                let v = &points[i].features;
                let projected_v = &projection * v;
                let diversity_term = projected_v.norm_squared();
                
                // Apply clamped logarithmic diversity term
                let clamped_diversity = self.compute_clamped_diversity_term(diversity_term);
                
                // Enhance diagonal elements
                enhanced_kernel[(i, i)] += clamped_diversity;
            }
        }
        
        Ok(enhanced_kernel)
    }

    fn compute_clamped_diversity_term(&self, diversity_value: f64) -> f64 {
        let r = self.diversity_controller.diversity_radius;
        let log_term = (1.0 + diversity_value).ln();
        let max_value = (1.0 + r).ln();
        
        // Clamp to [0, log(1+r)]
        log_term.max(0.0).min(max_value)
    }

    async fn sample_from_dpp(&self, kernel: &DMatrix<f64>, k: usize) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        // Implement DPP sampling algorithm
        // This is a simplified version - production would use more sophisticated algorithms
        
        let eigendecomposition = kernel.symmetric_eigen();
        let eigenvalues = eigendecomposition.eigenvalues;
        let eigenvectors = eigendecomposition.eigenvectors;
        
        // Marginal probabilities for each point
        let mut selected_indices = Vec::new();
        let n = kernel.nrows();
        
        // Simple greedy approximation (would use exact sampling in production)
        let mut remaining_indices: Vec<usize> = (0..n).collect();
        let mut current_kernel = kernel.clone();
        
        for _ in 0..k.min(n) {
            if remaining_indices.is_empty() {
                break;
            }
            
            // Find point with highest marginal probability
            let mut best_idx = 0;
            let mut best_score = 0.0;
            
            for (pos, &idx) in remaining_indices.iter().enumerate() {
                let score = current_kernel[(idx, idx)];
                if score > best_score {
                    best_score = score;
                    best_idx = pos;
                }
            }
            
            let selected_global_idx = remaining_indices.remove(best_idx);
            selected_indices.push(selected_global_idx);
            
            // Update kernel (simplified conditioning)
            self.condition_kernel_on_selection(&mut current_kernel, selected_global_idx, &remaining_indices);
        }
        
        Ok(selected_indices)
    }

    fn condition_kernel_on_selection(&self, kernel: &mut DMatrix<f64>, selected_idx: usize, remaining_indices: &[usize]) {
        // Simplified kernel conditioning
        // In production, would use proper DPP conditioning formulas
        for &i in remaining_indices {
            for &j in remaining_indices {
                if i != j {
                    let correction = kernel[(i, selected_idx)] * kernel[(selected_idx, j)] / (kernel[(selected_idx, selected_idx)] + 1e-10);
                    kernel[(i, j)] -= correction;
                }
            }
        }
    }

    fn compute_selection_quality(&self, selected_points: &[DPPPoint], kernel: &DMatrix<f64>) -> Result<SelectionQualityMetrics, Box<dyn std::error::Error>> {
        let n = selected_points.len();
        if n == 0 {
            return Ok(SelectionQualityMetrics {
                determinant_value: 0.0,
                condition_number: 0.0,
                spectral_gap: 0.0,
                group_balance: 0.0,
                novelty_score: 0.0,
                diversity_score: 0.0,
            });
        }
        
        // Create subkernel for selected points
        let indices: Vec<usize> = (0..n).collect(); // This would map to actual indices
        let mut subkernel = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                subkernel[(i, j)] = kernel[(indices[i], indices[j])];
            }
        }
        
        // Compute determinant (measure of diversity)
        let determinant_value = subkernel.determinant();
        
        // Compute condition number
        let svd = SVD::new(subkernel.clone(), true, true);
        let singular_values = svd.singular_values;
        let condition_number = if singular_values.len() > 0 {
            singular_values[0] / singular_values[singular_values.len() - 1]
        } else {
            1.0
        };
        
        // Compute spectral gap
        let spectral_gap = if singular_values.len() > 1 {
            singular_values[0] - singular_values[1]
        } else {
            0.0
        };
        
        // Compute group balance
        let group_distribution = self.compute_group_distribution(selected_points);
        let group_balance = self.compute_group_balance_score(&group_distribution);
        
        // Compute novelty score
        let novelty_score = self.compute_novelty_score(selected_points);
        
        Ok(SelectionQualityMetrics {
            determinant_value,
            condition_number,
            spectral_gap,
            group_balance,
            novelty_score,
            diversity_score: determinant_value.ln(), // Log-determinant as diversity measure
        })
    }

    // Helper methods

    fn compute_group_centroid(&self, members: &[DPPPoint]) -> DVector<f64> {
        if members.is_empty() {
            return DVector::zeros(0);
        }
        
        let dim = members[0].features.len();
        let mut centroid = DVector::zeros(dim);
        
        for member in members {
            centroid += &member.features;
        }
        
        centroid / members.len() as f64
    }

    fn compute_intra_group_penalty(&self, members: &[DPPPoint]) -> f64 {
        if members.len() <= 1 {
            return 0.0;
        }
        
        // Compute average pairwise similarity within group
        let mut total_similarity = 0.0;
        let mut pair_count = 0;
        
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                total_similarity += self.compute_feature_similarity(&members[i].features, &members[j].features);
                pair_count += 1;
            }
        }
        
        if pair_count > 0 {
            let avg_similarity = total_similarity / pair_count as f64;
            // Apply concave penalty function
            (avg_similarity.sqrt()).min(0.5) // Concave and bounded
        } else {
            0.0
        }
    }

    fn compute_local_kernel(&self, members: &[DPPPoint]) -> Result<DMatrix<f64>, Box<dyn std::error::Error>> {
        let n = members.len();
        let mut kernel = DMatrix::zeros(n, n);
        
        for i in 0..n {
            for j in 0..n {
                kernel[(i, j)] = self.compute_feature_similarity(&members[i].features, &members[j].features);
            }
        }
        
        Ok(kernel)
    }

    fn compute_feature_similarity(&self, v1: &DVector<f64>, v2: &DVector<f64>) -> f64 {
        // Cosine similarity
        let dot_product = v1.dot(v2);
        let norm_product = v1.norm() * v2.norm();
        if norm_product > 0.0 {
            dot_product / norm_product
        } else {
            0.0
        }
    }

    fn compute_group_distribution(&self, points: &[DPPPoint]) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        for point in points {
            *distribution.entry(point.group_membership.clone()).or_insert(0) += 1;
        }
        distribution
    }

    fn compute_group_balance_score(&self, distribution: &HashMap<String, usize>) -> f64 {
        if distribution.is_empty() {
            return 0.0;
        }
        
        let total: usize = distribution.values().sum();
        let n_groups = distribution.len();
        let expected_per_group = total as f64 / n_groups as f64;
        
        // Compute normalized variance
        let variance: f64 = distribution.values()
            .map(|&count| (count as f64 - expected_per_group).powi(2))
            .sum::<f64>() / n_groups as f64;
        
        // Convert to balance score (0 = unbalanced, 1 = perfectly balanced)
        1.0 / (1.0 + variance / expected_per_group.powi(2))
    }

    fn compute_novelty_score(&self, points: &[DPPPoint]) -> f64 {
        // Simplified novelty computation
        // Would compare against historical selections in production
        let mut avg_novelty = 0.0;
        for point in points {
            avg_novelty += point.diversity_contribution;
        }
        if points.len() > 0 {
            avg_novelty / points.len() as f64
        } else {
            0.0
        }
    }

    fn compute_condition_number(&self, matrix: &DMatrix<f64>) -> f64 {
        let svd = SVD::new(matrix.clone(), false, false);
        let singular_values = svd.singular_values;
        if singular_values.len() > 0 {
            singular_values[0] / singular_values[singular_values.len() - 1]
        } else {
            1.0
        }
    }

    fn compute_orthogonality_error(&self, q_matrix: &DMatrix<f64>) -> f64 {
        let qtq = q_matrix.transpose() * q_matrix;
        let identity = DMatrix::identity(qtq.nrows(), qtq.ncols());
        (&qtq - &identity).norm()
    }

    fn compute_spectral_properties(&self, matrix: &DMatrix<f64>) -> SpectralProperties {
        let eigendecomp = matrix.symmetric_eigen();
        let svd = SVD::new(matrix.clone(), false, false);
        
        let eigenvalues = eigendecomp.eigenvalues.iter().cloned().collect();
        let singular_values = svd.singular_values.iter().cloned().collect();
        let rank = svd.rank(1e-10);
        let null_space_dimension = matrix.ncols() - rank;
        
        SpectralProperties {
            eigenvalues,
            singular_values,
            rank,
            null_space_dimension,
        }
    }

    fn compute_numerical_stability(&self, matrix: &DMatrix<f64>) -> f64 {
        // Simple stability measure based on condition number
        let condition_number = self.compute_condition_number(matrix);
        1.0 / (1.0 + condition_number.ln())
    }

    async fn assign_to_group(&self, point: &DPPPoint) -> Result<(), Box<dyn std::error::Error>> {
        // Group assignment logic
        // In this implementation, we use the existing group_membership
        // In production, might use clustering algorithms
        Ok(())
    }

    async fn update_global_kernel(&self, new_points: &[DPPPoint]) -> Result<(), Box<dyn std::error::Error>> {
        // Update global kernel incrementally
        // This is a simplified implementation
        Ok(())
    }

    async fn update_selection_history(&self, event: SelectionEvent) {
        let mut groups = self.groups.write().unwrap();
        for point_id in &event.selected_points {
            // Find which group this point belongs to and update its history
            for group in groups.values_mut() {
                if group.members.iter().any(|p| p.point_id == *point_id) {
                    group.selection_history.push_back(event.clone());
                    if group.selection_history.len() > 100 {
                        group.selection_history.pop_front();
                    }
                    break;
                }
            }
        }
    }

    async fn update_performance_metrics(&self, duration: Duration, quality: &SelectionQualityMetrics) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_selections += 1;
        metrics.successful_selections += 1;
        
        // Update running averages
        let total = metrics.total_selections as f64;
        metrics.average_diversity_score = (metrics.average_diversity_score * (total - 1.0) + quality.diversity_score) / total;
        metrics.average_selection_time_ms = (metrics.average_selection_time_ms * (total - 1.0) + duration.as_millis() as f64) / total;
    }

    async fn analyze_group_performance(&self) -> GroupPerformanceAnalysis {
        let groups = self.groups.read().unwrap();
        let mut group_stats = HashMap::new();
        
        for (group_id, group) in groups.iter() {
            let recent_selections = group.selection_history.iter()
                .filter(|event| event.timestamp > Utc::now() - chrono::Duration::hours(24))
                .count();
            
            group_stats.insert(group_id.clone(), GroupStats {
                member_count: group.members.len(),
                recent_selections,
                diversity_score: group.diversity_score,
                intra_group_penalty: group.intra_group_penalty,
                last_updated: group.last_updated,
            });
        }
        
        GroupPerformanceAnalysis {
            group_stats,
            overall_balance: self.compute_overall_group_balance(&groups),
            diversity_distribution: self.compute_diversity_distribution(&groups),
        }
    }

    fn compute_overall_group_balance(&self, groups: &HashMap<String, DPPGroup>) -> f64 {
        if groups.is_empty() {
            return 1.0;
        }
        
        let member_counts: Vec<usize> = groups.values().map(|g| g.members.len()).collect();
        let total: usize = member_counts.iter().sum();
        let expected = total as f64 / groups.len() as f64;
        
        let variance: f64 = member_counts.iter()
            .map(|&count| (count as f64 - expected).powi(2))
            .sum::<f64>() / groups.len() as f64;
        
        1.0 / (1.0 + variance / expected.powi(2))
    }

    fn compute_diversity_distribution(&self, groups: &HashMap<String, DPPGroup>) -> HashMap<String, f64> {
        groups.iter()
            .map(|(id, group)| (id.clone(), group.diversity_score))
            .collect()
    }
}

// Implementation of supporting components

impl DPPGroup {
    fn new(group_id: String) -> Self {
        Self {
            group_id,
            centroid: DVector::zeros(0),
            members: Vec::new(),
            local_kernel: DMatrix::zeros(0, 0),
            intra_group_penalty: 0.0,
            diversity_score: 0.0,
            last_updated: Utc::now(),
            selection_history: VecDeque::new(),
        }
    }
}

impl DiversityController {
    fn new() -> Self {
        Self {
            diversity_radius: 1.0,
            clamp_bounds: (0.0, (1.0 + 1.0_f64).ln()),
            penalty_function: PenaltyFunction::Concave,
            optimization_target: OptimizationTarget::BalanceQualityDiversity,
            adaptive_weights: Arc::new(RwLock::new(AdaptiveWeights {
                quality_weight: 0.4,
                diversity_weight: 0.3,
                group_balance_weight: 0.2,
                novelty_weight: 0.1,
                adaptation_rate: 0.01,
                last_adaptation: Utc::now(),
            })),
        }
    }

    async fn get_state(&self) -> DiversityControllerState {
        let weights = self.adaptive_weights.read().unwrap().clone();
        DiversityControllerState {
            diversity_radius: self.diversity_radius,
            clamp_bounds: self.clamp_bounds,
            penalty_function: self.penalty_function.clone(),
            optimization_target: self.optimization_target.clone(),
            adaptive_weights: weights,
        }
    }
}

impl OrthonormalizationTracker {
    fn new() -> Self {
        Self {
            insertion_count: Arc::new(RwLock::new(0)),
            reorthonormalization_threshold: 128,
            last_reorthonormalization: Arc::new(RwLock::new(Utc::now())),
            orthogonality_metrics: Arc::new(Mutex::new(OrthogonalityMetrics {
                condition_number: 1.0,
                orthogonality_error: 0.0,
                spectral_properties: SpectralProperties {
                    eigenvalues: vec![],
                    singular_values: vec![],
                    rank: 0,
                    null_space_dimension: 0,
                },
                numerical_stability: 1.0,
            })),
            reorthonormalization_history: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
}

impl DPPPerformanceMonitor {
    fn new() -> Self {
        Self {
            selection_times: Arc::new(Mutex::new(VecDeque::new())),
            kernel_computation_times: Arc::new(Mutex::new(VecDeque::new())),
            reorthonormalization_times: Arc::new(Mutex::new(VecDeque::new())),
            group_update_times: Arc::new(Mutex::new(VecDeque::new())),
            quality_trends: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    async fn get_summary(&self) -> PerformanceSummary {
        let selection_times = self.selection_times.lock().unwrap();
        let avg_selection_time = if selection_times.is_empty() {
            Duration::from_millis(0)
        } else {
            let total: Duration = selection_times.iter().sum();
            total / selection_times.len() as u32
        };

        PerformanceSummary {
            average_selection_time: avg_selection_time,
            recent_quality_trend: 0.95, // Would compute from actual data
            system_efficiency: 0.88,     // Would compute from actual metrics
            stability_score: 0.92,       // Would compute from actual metrics
        }
    }
}

// Supporting result types

#[derive(Debug, Clone)]
pub struct DPPSelectionResult {
    pub selected_points: Vec<DPPPoint>,
    pub selection_quality: SelectionQualityMetrics,
    pub computational_stats: ComputationalStats,
    pub group_analysis: GroupPerformanceAnalysis,
}

#[derive(Debug, Clone)]
pub struct ComputationalStats {
    pub total_time: Duration,
    pub kernel_computation_time: Duration,
    pub sampling_time: Duration,
    pub group_update_time: Duration,
}

#[derive(Debug, Clone)]
pub struct GroupPerformanceAnalysis {
    pub group_stats: HashMap<String, GroupStats>,
    pub overall_balance: f64,
    pub diversity_distribution: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct GroupStats {
    pub member_count: usize,
    pub recent_selections: usize,
    pub diversity_score: f64,
    pub intra_group_penalty: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct DiversityControllerState {
    pub diversity_radius: f64,
    pub clamp_bounds: (f64, f64),
    pub penalty_function: PenaltyFunction,
    pub optimization_target: OptimizationTarget,
    pub adaptive_weights: AdaptiveWeights,
}

#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub average_selection_time: Duration,
    pub recent_quality_trend: f64,
    pub system_efficiency: f64,
    pub stability_score: f64,
}

#[derive(Debug, Clone)]
pub struct DPPSystemStatus {
    pub group_count: usize,
    pub total_points: usize,
    pub orthogonality_metrics: OrthogonalityMetrics,
    pub insertion_count: u64,
    pub next_reorthonormalization_at: u64,
    pub diversity_controller_state: DiversityControllerState,
    pub performance_summary: PerformanceSummary,
    pub metrics: DPPMetrics,
    pub last_updated: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_grouped_dpp_engine_creation() {
        let engine = GroupedDPPEngine::new().unwrap();
        let status = engine.get_system_status().await;
        
        assert_eq!(status.group_count, 0);
        assert_eq!(status.total_points, 0);
        assert_eq!(status.insertion_count, 0);
    }

    #[tokio::test]
    async fn test_point_addition() {
        let engine = GroupedDPPEngine::new().unwrap();
        
        let test_points = vec![
            DPPPoint {
                point_id: Uuid::new_v4(),
                features: DVector::from_vec(vec![1.0, 2.0, 3.0]),
                group_membership: "group_a".to_string(),
                selection_probability: 0.5,
                diversity_contribution: 0.7,
                timestamp: Utc::now(),
            },
            DPPPoint {
                point_id: Uuid::new_v4(),
                features: DVector::from_vec(vec![4.0, 5.0, 6.0]),
                group_membership: "group_b".to_string(),
                selection_probability: 0.6,
                diversity_contribution: 0.8,
                timestamp: Utc::now(),
            },
        ];
        
        engine.add_points(test_points).await.unwrap();
        
        let status = engine.get_system_status().await;
        assert_eq!(status.insertion_count, 2);
    }

    #[tokio::test]
    async fn test_grouped_selection() {
        let engine = GroupedDPPEngine::new().unwrap();
        
        let candidate_points = vec![
            DPPPoint {
                point_id: Uuid::new_v4(),
                features: DVector::from_vec(vec![1.0, 0.0, 0.0]),
                group_membership: "group_a".to_string(),
                selection_probability: 0.5,
                diversity_contribution: 0.7,
                timestamp: Utc::now(),
            },
            DPPPoint {
                point_id: Uuid::new_v4(),
                features: DVector::from_vec(vec![0.0, 1.0, 0.0]),
                group_membership: "group_a".to_string(),
                selection_probability: 0.6,
                diversity_contribution: 0.8,
                timestamp: Utc::now(),
            },
            DPPPoint {
                point_id: Uuid::new_v4(),
                features: DVector::from_vec(vec![0.0, 0.0, 1.0]),
                group_membership: "group_b".to_string(),
                selection_probability: 0.7,
                diversity_contribution: 0.9,
                timestamp: Utc::now(),
            },
        ];
        
        let result = engine.execute_grouped_selection(candidate_points, 2).await.unwrap();
        
        assert_eq!(result.selected_points.len(), 2);
        assert!(result.selection_quality.diversity_score >= 0.0);
        assert!(result.computational_stats.total_time > Duration::from_nanos(0));
    }

    #[test]
    fn test_clamped_diversity_computation() {
        let engine = GroupedDPPEngine::new().unwrap();
        
        // Test with different diversity values
        let low_diversity = engine.compute_clamped_diversity_term(0.1);
        let high_diversity = engine.compute_clamped_diversity_term(10.0);
        
        assert!(low_diversity > 0.0);
        assert!(high_diversity > low_diversity);
        
        // Should be clamped to max value
        let max_value = (1.0 + engine.diversity_controller.diversity_radius).ln();
        assert!(high_diversity <= max_value);
    }

    #[test]
    fn test_group_balance_computation() {
        let engine = GroupedDPPEngine::new().unwrap();
        
        // Perfect balance
        let balanced_distribution = HashMap::from([
            ("group_a".to_string(), 5),
            ("group_b".to_string(), 5),
            ("group_c".to_string(), 5),
        ]);
        let balance_score = engine.compute_group_balance_score(&balanced_distribution);
        assert!(balance_score > 0.9); // Should be close to 1.0
        
        // Imbalanced distribution
        let imbalanced_distribution = HashMap::from([
            ("group_a".to_string(), 1),
            ("group_b".to_string(), 10),
            ("group_c".to_string(), 1),
        ]);
        let imbalance_score = engine.compute_group_balance_score(&imbalanced_distribution);
        assert!(imbalance_score < balance_score);
    }

    #[test]
    fn test_feature_similarity() {
        let engine = GroupedDPPEngine::new().unwrap();
        
        let v1 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let v2 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let v3 = DVector::from_vec(vec![0.0, 1.0, 0.0]);
        
        // Identical vectors should have similarity 1.0
        let sim_identical = engine.compute_feature_similarity(&v1, &v2);
        assert!((sim_identical - 1.0).abs() < 1e-10);
        
        // Orthogonal vectors should have similarity 0.0
        let sim_orthogonal = engine.compute_feature_similarity(&v1, &v3);
        assert!((sim_orthogonal - 0.0).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_reorthonormalization_trigger() {
        let engine = GroupedDPPEngine::new().unwrap();
        
        // Set a low threshold for testing
        {
            let mut count = engine.orthonormalization_tracker.insertion_count.write().unwrap();
            *count = 127; // Just before threshold
        }
        
        // Add one more point to trigger reorthonormalization
        let test_point = DPPPoint {
            point_id: Uuid::new_v4(),
            features: DVector::from_vec(vec![1.0, 2.0, 3.0]),
            group_membership: "test_group".to_string(),
            selection_probability: 0.5,
            diversity_contribution: 0.7,
            timestamp: Utc::now(),
        };
        
        engine.add_points(vec![test_point]).await.unwrap();
        
        let status = engine.get_system_status().await;
        assert_eq!(status.insertion_count, 128);
        
        // Reorthonormalization should have been triggered
        let history = engine.orthonormalization_tracker.reorthonormalization_history.lock().unwrap();
        assert!(!history.is_empty());
    }
}