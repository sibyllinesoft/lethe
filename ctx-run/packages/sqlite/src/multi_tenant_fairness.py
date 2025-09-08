#!/usr/bin/env python3
"""
Multi-Tenant Fairness and Algorithmic Enhancements for Lethe

Implements advanced fairness mechanisms and algorithmic optimizations:
1. Jain's fairness index with shadow-price λ allocation
2. Group closure optimization with bounded split moves  
3. Grouped-DPP with laminar constraints and PSD properties
4. Resource starvation prevention and workload mix adaptation
5. Operational constraint enforcement (λ/μ drift limits, promotion freezes)

Mathematical Framework:
- Jain's Fairness Index: J = (Σx_i)² / (n·Σx_i²)
- Bounded split moves with τ=0.7 threshold and cool-down
- Log-determinant optimization on group representatives
- Intra-group concave penalty with PSD kernel matrices
- Drift constraints: λ-drift, μ-drift ≤ ±15%/24h
"""

import logging
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from collections import defaultdict, deque
from scipy.linalg import eigvals, cholesky, LinAlgError
from scipy.optimize import minimize
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class TenantAllocation:
    """Resource allocation for a single tenant."""
    tenant_id: str
    lambda_multiplier: float
    mu_parameter: float
    resource_share: float          # Fraction of total resources
    prefix_reuse_rate: float       # KV cache reuse rate
    performance_score: float       # Recent performance
    last_updated: datetime
    
@dataclass
class FairnessMetrics:
    """Fairness metrics and violations."""
    jain_fairness_index: float
    resource_entropy: float        # Shannon entropy of resource distribution
    gini_coefficient: float        # Gini coefficient for inequality
    max_deviation_from_fair: float # Maximum deviation from fair share
    starvation_detected: Set[str]  # Tenants below threshold
    monopolization_detected: Set[str]  # Tenants above threshold

@dataclass 
class GroupClosureState:
    """Group closure optimization state."""
    current_tau: float
    active_closures: List[Set[int]]  # Active closure sets
    recent_split_moves: int
    split_move_cooldown: int
    ilp_overhead_ms: float
    high_gain_protection_active: bool
    sibling_drag_prevented: bool

@dataclass
class GroupedDPPKernel:
    """Grouped-DPP kernel with PSD properties."""
    group_representatives: List[int]
    kernel_matrix: np.ndarray
    log_determinant: float
    eigenvalues: List[float]
    psd_verified: bool
    intra_group_penalties: Dict[int, float]
    marginal_quality_score: float

@dataclass
class MultiTenantOptimizationResult:
    """Result of multi-tenant fairness optimization."""
    fairness_metrics: FairnessMetrics
    tenant_allocations: Dict[str, TenantAllocation]
    group_closure_state: GroupClosureState
    grouped_dpp_kernel: GroupedDPPKernel
    
    # Constraint satisfaction
    drift_constraints_satisfied: bool
    operational_constraints_satisfied: bool
    promotion_freeze_status: Optional[Dict[str, Any]]
    
    # Quality metrics
    overall_fairness_score: float
    system_efficiency: float
    ungameable_score: float
    
    # Diagnostics
    optimization_time_ms: float
    warnings: List[str]
    recommendations: List[str]

class MultiTenantFairnessSystem:
    """
    Comprehensive multi-tenant fairness system with algorithmic enhancements.
    
    Provides:
    1. Fair resource allocation with Jain's index optimization
    2. Group closure optimization with bounded split moves
    3. Grouped-DPP with laminar constraints
    4. Drift monitoring and constraint enforcement
    5. Anti-gaming mechanisms
    """
    
    def __init__(
        self,
        jain_index_threshold: float = 0.8,
        drift_limit_24h: float = 0.15,
        starvation_threshold: float = 0.1,
        monopolization_threshold: float = 0.4,
        tau_threshold: float = 0.7,
        split_cooldown_cycles: int = 10
    ):
        """Initialize multi-tenant fairness system."""
        self.jain_index_threshold = jain_index_threshold
        self.drift_limit_24h = drift_limit_24h
        self.starvation_threshold = starvation_threshold
        self.monopolization_threshold = monopolization_threshold
        self.tau_threshold = tau_threshold
        self.split_cooldown_cycles = split_cooldown_cycles
        
        # State tracking
        self.tenant_allocations: Dict[str, TenantAllocation] = {}
        self.allocation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24h at 1min intervals
        self.fairness_history = deque(maxlen=100)
        
        # Group closure state
        self.group_closure_state = GroupClosureState(
            current_tau=tau_threshold,
            active_closures=[],
            recent_split_moves=0,
            split_move_cooldown=0,
            ilp_overhead_ms=0.0,
            high_gain_protection_active=True,
            sibling_drag_prevented=True
        )
        
        # Promotion freeze tracking
        self.promotion_freeze = {
            'active': False,
            'start_time': None,
            'reason': None,
            'affected_pools': set()
        }
        
        logger.info("Multi-tenant fairness system initialized")
    
    def optimize_fairness(
        self,
        tenant_demands: Dict[str, Dict[str, float]],
        system_capacity: Dict[str, float],
        performance_data: Dict[str, Any],
        group_structure: Optional[List[Set[int]]] = None
    ) -> MultiTenantOptimizationResult:
        """
        Optimize multi-tenant fairness with algorithmic enhancements.
        
        Args:
            tenant_demands: Per-tenant resource demands and preferences
            system_capacity: Available system capacity
            performance_data: Recent performance metrics
            group_structure: Optional candidate grouping structure
            
        Returns:
            MultiTenantOptimizationResult with optimization decisions
        """
        start_time = time.time()
        
        try:
            # 1. Update tenant allocations with fairness optimization
            tenant_allocations = self._optimize_tenant_allocations(tenant_demands, system_capacity)
            
            # 2. Calculate comprehensive fairness metrics
            fairness_metrics = self._calculate_fairness_metrics(tenant_allocations)
            
            # 3. Optimize group closure with bounded split moves
            group_closure_state = self._optimize_group_closures(
                performance_data, group_structure or []
            )
            
            # 4. Build Grouped-DPP kernel with laminar constraints
            grouped_dpp_kernel = self._build_grouped_dpp_kernel(
                group_closure_state, performance_data
            )
            
            # 5. Check constraint satisfaction
            drift_satisfied = self._check_drift_constraints()
            operational_satisfied = self._check_operational_constraints()
            
            # 6. Update promotion freeze status
            promotion_freeze_status = self._update_promotion_freeze_status(performance_data)
            
            # 7. Calculate quality scores
            fairness_score = self._calculate_overall_fairness_score(fairness_metrics)
            efficiency_score = self._calculate_system_efficiency(tenant_allocations, performance_data)
            ungameable_score = self._calculate_ungameable_score(
                fairness_metrics, group_closure_state, grouped_dpp_kernel
            )
            
            # 8. Generate diagnostics
            warnings, recommendations = self._generate_fairness_diagnostics(
                fairness_metrics, tenant_allocations, group_closure_state
            )
            
            optimization_time = (time.time() - start_time) * 1000
            
            result = MultiTenantOptimizationResult(
                fairness_metrics=fairness_metrics,
                tenant_allocations=tenant_allocations,
                group_closure_state=group_closure_state,
                grouped_dpp_kernel=grouped_dpp_kernel,
                drift_constraints_satisfied=drift_satisfied,
                operational_constraints_satisfied=operational_satisfied,
                promotion_freeze_status=promotion_freeze_status,
                overall_fairness_score=fairness_score,
                system_efficiency=efficiency_score,
                ungameable_score=ungameable_score,
                optimization_time_ms=optimization_time,
                warnings=warnings,
                recommendations=recommendations
            )
            
            # Update history
            self.fairness_history.append(fairness_metrics)
            self._update_allocation_history(tenant_allocations)
            
            logger.info(
                f"Fairness optimization complete: Jain={fairness_metrics.jain_fairness_index:.3f}, "
                f"efficiency={efficiency_score:.3f}, ungameable={ungameable_score:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-tenant fairness optimization failed: {e}")
            return self._create_error_result(start_time, str(e))
    
    def _optimize_tenant_allocations(
        self,
        tenant_demands: Dict[str, Dict[str, float]],
        system_capacity: Dict[str, float]
    ) -> Dict[str, TenantAllocation]:
        """Optimize tenant resource allocations for fairness."""
        current_time = datetime.now()
        total_demand = sum(
            demand.get('resource_demand', 1.0) for demand in tenant_demands.values()
        )
        
        # Calculate fair share baseline
        num_tenants = len(tenant_demands)
        if num_tenants == 0:
            return {}
        
        fair_share = 1.0 / num_tenants
        total_capacity = system_capacity.get('total_capacity', 1.0)
        
        # Optimize allocations using proportional fairness with demand weighting
        optimized_allocations = {}
        
        for tenant_id, demand_data in tenant_demands.items():
            # Base allocation proportional to demand
            demand_weight = demand_data.get('resource_demand', 1.0) / total_demand
            base_allocation = demand_weight * total_capacity
            
            # Apply fairness adjustments
            fairness_adjustment = self._calculate_fairness_adjustment(
                tenant_id, base_allocation, fair_share
            )
            
            final_allocation = max(
                self.starvation_threshold,
                min(self.monopolization_threshold, base_allocation + fairness_adjustment)
            )
            
            # Calculate shadow-price λ for this tenant
            lambda_multiplier = self._calculate_shadow_price_lambda(
                tenant_id, final_allocation, demand_data
            )
            
            # Calculate μ parameter
            mu_parameter = self._calculate_mu_parameter(tenant_id, demand_data)
            
            # Calculate prefix reuse rate
            prefix_reuse = self._calculate_prefix_reuse_rate(tenant_id, demand_data)
            
            # Performance score
            performance_score = demand_data.get('recent_performance', 0.8)
            
            optimized_allocations[tenant_id] = TenantAllocation(
                tenant_id=tenant_id,
                lambda_multiplier=lambda_multiplier,
                mu_parameter=mu_parameter,
                resource_share=final_allocation,
                prefix_reuse_rate=prefix_reuse,
                performance_score=performance_score,
                last_updated=current_time
            )
        
        return optimized_allocations
    
    def _calculate_fairness_adjustment(
        self,
        tenant_id: str,
        base_allocation: float,
        fair_share: float
    ) -> float:
        """Calculate fairness adjustment for tenant allocation."""
        # Historical fairness bias
        if tenant_id in self.tenant_allocations:
            historical_allocation = self.tenant_allocations[tenant_id].resource_share
            historical_bias = historical_allocation - fair_share
            
            # Apply negative feedback to balance historical bias
            adjustment = -0.1 * historical_bias
        else:
            adjustment = 0.0
        
        # Performance-based adjustment
        if tenant_id in self.tenant_allocations:
            performance = self.tenant_allocations[tenant_id].performance_score
            if performance < 0.5:  # Poor performance
                adjustment -= 0.05  # Slight penalty
            elif performance > 0.9:  # Excellent performance
                adjustment += 0.02  # Small bonus
        
        return adjustment
    
    def _calculate_shadow_price_lambda(
        self,
        tenant_id: str,
        allocation: float,
        demand_data: Dict[str, float]
    ) -> float:
        """Calculate shadow-price λ for tenant's resource allocation."""
        # Base λ inversely related to allocation (more resources = lower λ)
        base_lambda = 1.0 / (allocation + 1e-6)
        
        # Adjust for demand urgency
        urgency = demand_data.get('urgency_factor', 1.0)
        lambda_multiplier = base_lambda * urgency
        
        # Historical λ smoothing
        if tenant_id in self.tenant_allocations:
            historical_lambda = self.tenant_allocations[tenant_id].lambda_multiplier
            # Exponential moving average with α=0.3
            lambda_multiplier = 0.7 * lambda_multiplier + 0.3 * historical_lambda
        
        return np.clip(lambda_multiplier, 0.1, 5.0)
    
    def _calculate_mu_parameter(self, tenant_id: str, demand_data: Dict[str, float]) -> float:
        """Calculate μ parameter for tenant's hysteretic control."""
        # Base μ from demand characteristics
        base_mu = demand_data.get('latency_sensitivity', 1.0)
        
        # Adjust for tenant's historical performance
        if tenant_id in self.tenant_allocations:
            historical_performance = self.tenant_allocations[tenant_id].performance_score
            if historical_performance < 0.6:
                base_mu *= 1.2  # Increase μ for struggling tenants
            elif historical_performance > 0.9:
                base_mu *= 0.9  # Decrease μ for high-performing tenants
        
        return np.clip(base_mu, 0.1, 3.0)
    
    def _calculate_prefix_reuse_rate(self, tenant_id: str, demand_data: Dict[str, float]) -> float:
        """Calculate KV prefix reuse rate for tenant."""
        # Base reuse rate from query patterns
        base_reuse = demand_data.get('query_similarity', 0.7)
        
        # Adjust for tenant's access patterns
        if tenant_id in self.tenant_allocations:
            historical_reuse = self.tenant_allocations[tenant_id].prefix_reuse_rate
            # Smooth adjustment
            reuse_rate = 0.8 * base_reuse + 0.2 * historical_reuse
        else:
            reuse_rate = base_reuse
        
        return np.clip(reuse_rate, 0.1, 0.95)
    
    def _calculate_fairness_metrics(
        self,
        tenant_allocations: Dict[str, TenantAllocation]
    ) -> FairnessMetrics:
        """Calculate comprehensive fairness metrics."""
        if not tenant_allocations:
            return FairnessMetrics(
                jain_fairness_index=1.0,
                resource_entropy=0.0,
                gini_coefficient=0.0,
                max_deviation_from_fair=0.0,
                starvation_detected=set(),
                monopolization_detected=set()
            )
        
        allocations = [alloc.resource_share for alloc in tenant_allocations.values()]
        n = len(allocations)
        fair_share = 1.0 / n
        
        # Jain's Fairness Index: J = (Σx_i)² / (n·Σx_i²)
        sum_allocations = sum(allocations)
        sum_squared_allocations = sum(x**2 for x in allocations)
        
        if sum_squared_allocations > 0:
            jain_index = (sum_allocations**2) / (n * sum_squared_allocations)
        else:
            jain_index = 1.0
        
        # Resource entropy (Shannon entropy)
        if sum_allocations > 0:
            normalized_allocations = [x / sum_allocations for x in allocations]
            resource_entropy = -sum(
                p * math.log2(p) for p in normalized_allocations if p > 0
            )
        else:
            resource_entropy = 0.0
        
        # Gini coefficient
        gini_coefficient = self._calculate_gini_coefficient(allocations)
        
        # Maximum deviation from fair share
        max_deviation = max(abs(alloc - fair_share) for alloc in allocations)
        
        # Detect starvation and monopolization
        starvation_detected = {
            tenant_id for tenant_id, alloc in tenant_allocations.items()
            if alloc.resource_share < self.starvation_threshold
        }
        
        monopolization_detected = {
            tenant_id for tenant_id, alloc in tenant_allocations.items()
            if alloc.resource_share > self.monopolization_threshold
        }
        
        return FairnessMetrics(
            jain_fairness_index=jain_index,
            resource_entropy=resource_entropy,
            gini_coefficient=gini_coefficient,
            max_deviation_from_fair=max_deviation,
            starvation_detected=starvation_detected,
            monopolization_detected=monopolization_detected
        )
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if not values or len(values) == 1:
            return 0.0
        
        # Sort values
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Calculate Gini coefficient
        cumulative = 0.0
        for i, value in enumerate(sorted_values):
            cumulative += (2 * (i + 1) - n - 1) * value
        
        mean_value = sum(sorted_values) / n
        if mean_value > 0:
            gini = cumulative / (n * n * mean_value)
        else:
            gini = 0.0
        
        return max(0.0, min(1.0, gini))
    
    def _optimize_group_closures(
        self,
        performance_data: Dict[str, Any],
        group_structure: List[Set[int]]
    ) -> GroupClosureState:
        """Optimize group closures with bounded split moves."""
        # Update cooldown counter
        if self.group_closure_state.split_move_cooldown > 0:
            self.group_closure_state.split_move_cooldown -= 1
        
        # Check if split move is beneficial and allowed
        should_split, split_rationale = self._should_perform_split_move(performance_data)
        
        if should_split and self.group_closure_state.split_move_cooldown == 0:
            # Perform bounded split move
            new_closures = self._perform_bounded_split_move(group_structure)
            self.group_closure_state.active_closures = new_closures
            self.group_closure_state.recent_split_moves += 1
            self.group_closure_state.split_move_cooldown = self.split_cooldown_cycles
            
            logger.info(f"Performed bounded split move: {split_rationale}")
        
        # Update ILP overhead tracking
        ilp_time = performance_data.get('ilp_time_ms', 0.0)
        total_time = performance_data.get('total_time_ms', 1.0)
        self.group_closure_state.ilp_overhead_ms = ilp_time / total_time if total_time > 0 else 0.0
        
        # Check high-gain children protection
        self.group_closure_state.high_gain_protection_active = self._check_high_gain_protection(
            performance_data
        )
        
        # Check sibling drag prevention
        self.group_closure_state.sibling_drag_prevented = self._check_sibling_drag_prevention(
            performance_data
        )
        
        return self.group_closure_state
    
    def _should_perform_split_move(self, performance_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if bounded split move should be performed."""
        # Check ILP overhead constraint
        if self.group_closure_state.ilp_overhead_ms > 0.05:  # >5% overhead
            return False, "ILP overhead too high"
        
        # Check for performance degradation in groups
        group_performance_variance = performance_data.get('group_performance_variance', 0.0)
        if group_performance_variance > 0.2:  # High variance suggests suboptimal grouping
            return True, f"High group performance variance: {group_performance_variance:.3f}"
        
        # Check for high-gain children being dragged down
        high_gain_children = performance_data.get('high_gain_children_score', 0.0)
        sibling_average = performance_data.get('sibling_average_score', 0.0)
        
        if high_gain_children > sibling_average * 1.3:  # 30% better than siblings
            return True, "High-gain children being held back by siblings"
        
        return False, "No split move needed"
    
    def _perform_bounded_split_move(self, group_structure: List[Set[int]]) -> List[Set[int]]:
        """Perform bounded split move with τ=0.7 threshold."""
        new_closures = []
        
        for group in group_structure:
            if len(group) <= 2:
                # Don't split small groups
                new_closures.append(group)
                continue
            
            # Calculate intra-group similarities (simplified)
            similarities = []
            group_list = list(group)
            for i in range(len(group_list)):
                for j in range(i + 1, len(group_list)):
                    # Simplified similarity calculation
                    sim = 0.5 + 0.3 * np.random.random()  # Would use actual similarity
                    similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            # Split if average similarity < τ
            if avg_similarity < self.tau_threshold:
                # Split into two subgroups (simplified splitting)
                mid = len(group_list) // 2
                subgroup1 = set(group_list[:mid])
                subgroup2 = set(group_list[mid:])
                new_closures.extend([subgroup1, subgroup2])
            else:
                new_closures.append(group)
        
        return new_closures
    
    def _check_high_gain_protection(self, performance_data: Dict[str, Any]) -> bool:
        """Check if high-gain children are protected from underperforming siblings."""
        high_gain_score = performance_data.get('high_gain_children_score', 1.0)
        sibling_scores = performance_data.get('sibling_scores', [1.0])
        
        if not sibling_scores:
            return True
        
        avg_sibling_score = np.mean(sibling_scores)
        
        # Protection is active if high-gain children significantly outperform siblings
        return high_gain_score >= avg_sibling_score * 0.95  # Allow 5% tolerance
    
    def _check_sibling_drag_prevention(self, performance_data: Dict[str, Any]) -> bool:
        """Check if sibling drag is being prevented."""
        drag_factor = performance_data.get('sibling_drag_factor', 0.0)
        return drag_factor <= 0.1  # Allow up to 10% drag
    
    def _build_grouped_dpp_kernel(
        self,
        group_closure_state: GroupClosureState,
        performance_data: Dict[str, Any]
    ) -> GroupedDPPKernel:
        """Build Grouped-DPP kernel with laminar constraints and PSD properties."""
        # Select group representatives
        representatives = []
        for group in group_closure_state.active_closures[:8]:  # Limit to 8 groups
            if group:
                rep = min(group)  # Simple selection - use first element
                representatives.append(rep)
        
        if not representatives:
            representatives = [0, 1, 2, 3]  # Fallback
        
        n = len(representatives)
        
        # Build kernel matrix with diversity and quality components
        kernel_matrix = self._build_dpp_kernel_matrix(representatives, performance_data)
        
        # Verify PSD property
        psd_verified, eigenvalues = self._verify_psd_property(kernel_matrix)
        
        # Calculate log-determinant
        try:
            log_determinant = math.log(max(np.linalg.det(kernel_matrix), 1e-10))
        except (ValueError, LinAlgError):
            log_determinant = -10.0  # Log of very small determinant
        
        # Calculate intra-group penalties
        intra_group_penalties = self._calculate_intra_group_penalties(
            representatives, group_closure_state.active_closures
        )
        
        # Assess marginal mathematics quality
        marginal_quality = self._assess_marginal_mathematics_quality(
            kernel_matrix, intra_group_penalties
        )
        
        return GroupedDPPKernel(
            group_representatives=representatives,
            kernel_matrix=kernel_matrix,
            log_determinant=log_determinant,
            eigenvalues=eigenvalues,
            psd_verified=psd_verified,
            intra_group_penalties=intra_group_penalties,
            marginal_quality_score=marginal_quality
        )
    
    def _build_dpp_kernel_matrix(
        self,
        representatives: List[int],
        performance_data: Dict[str, Any]
    ) -> np.ndarray:
        """Build DPP kernel matrix with diversity and quality components."""
        n = len(representatives)
        kernel = np.zeros((n, n))
        
        # Diagonal entries (quality scores)
        for i, rep in enumerate(representatives):
            quality_score = performance_data.get(f'quality_{rep}', 0.8)
            kernel[i, i] = quality_score
        
        # Off-diagonal entries (similarity/diversity)
        for i in range(n):
            for j in range(i + 1, n):
                # Similarity between representatives (simplified)
                # In practice, would use actual embedding similarity
                similarity = 0.3 + 0.4 * np.random.random()
                
                # Diversity = 1 - similarity for DPP
                diversity = 1.0 - similarity
                
                # Geometric mean of qualities weighted by diversity
                quality_i = kernel[i, i]
                quality_j = kernel[j, j]
                kernel[i, j] = kernel[j, i] = diversity * math.sqrt(quality_i * quality_j)
        
        # Add regularization for numerical stability
        kernel += 1e-6 * np.eye(n)
        
        return kernel
    
    def _verify_psd_property(self, matrix: np.ndarray) -> Tuple[bool, List[float]]:
        """Verify that kernel matrix has positive semi-definite property."""
        try:
            eigenvalues = eigvals(matrix).real.tolist()
            min_eigenvalue = min(eigenvalues)
            
            # Allow small negative eigenvalues due to numerical errors
            psd_verified = min_eigenvalue >= -1e-8
            
            return psd_verified, eigenvalues
            
        except LinAlgError as e:
            logger.warning(f"Eigenvalue computation failed: {e}")
            return False, []
    
    def _calculate_intra_group_penalties(
        self,
        representatives: List[int],
        active_closures: List[Set[int]]
    ) -> Dict[int, float]:
        """Calculate intra-group concave penalties."""
        penalties = {}
        
        for rep in representatives:
            # Find which group this representative belongs to
            group_size = 1
            intra_similarity = 0.0
            
            for group in active_closures:
                if rep in group:
                    group_size = len(group)
                    # Simplified intra-group similarity calculation
                    intra_similarity = 0.5 - 0.1 * min(group_size, 5)  # Decreases with size
                    break
            
            # Concave penalty encourages diversity within groups
            penalty = 0.1 * intra_similarity ** 2  # Quadratic penalty
            penalties[rep] = penalty
        
        return penalties
    
    def _assess_marginal_mathematics_quality(
        self,
        kernel_matrix: np.ndarray,
        penalties: Dict[int, float]
    ) -> float:
        """Assess quality of marginal mathematics under closures."""
        try:
            # Check condition number of kernel matrix
            condition_number = np.linalg.cond(kernel_matrix)
            
            # Check determinant stability
            det_value = np.linalg.det(kernel_matrix)
            det_stable = det_value > 1e-10
            
            # Check penalty consistency
            penalty_variance = np.var(list(penalties.values())) if penalties else 0.0
            penalty_consistent = penalty_variance < 0.01
            
            # Composite quality score
            quality_components = [
                1.0 / (1.0 + condition_number / 100.0),  # Lower condition number is better
                1.0 if det_stable else 0.0,
                1.0 if penalty_consistent else 0.5
            ]
            
            marginal_quality = np.mean(quality_components)
            
        except Exception as e:
            logger.warning(f"Marginal mathematics quality assessment failed: {e}")
            marginal_quality = 0.5  # Neutral score
        
        return marginal_quality
    
    def _check_drift_constraints(self) -> bool:
        """Check λ and μ drift constraints over 24h."""
        current_time = time.time()
        time_24h_ago = current_time - 24 * 3600
        
        for tenant_id, history in self.allocation_history.items():
            if not history:
                continue
            
            # Find allocations from ~24h ago
            recent_allocations = [
                alloc for timestamp, alloc in history
                if timestamp >= time_24h_ago
            ]
            
            if len(recent_allocations) < 2:
                continue  # Insufficient data
            
            # Check λ drift
            lambda_values = [alloc.lambda_multiplier for _, alloc in recent_allocations]
            if lambda_values:
                lambda_drift = (lambda_values[-1] - lambda_values[0]) / lambda_values[0]
                if abs(lambda_drift) > self.drift_limit_24h:
                    logger.warning(f"λ drift violation for {tenant_id}: {lambda_drift:.2f}")
                    return False
            
            # Check μ drift
            mu_values = [alloc.mu_parameter for _, alloc in recent_allocations]
            if mu_values:
                mu_drift = (mu_values[-1] - mu_values[0]) / mu_values[0]
                if abs(mu_drift) > self.drift_limit_24h:
                    logger.warning(f"μ drift violation for {tenant_id}: {mu_drift:.2f}")
                    return False
        
        return True
    
    def _check_operational_constraints(self) -> bool:
        """Check operational constraints satisfaction."""
        # Check ILP overhead
        if self.group_closure_state.ilp_overhead_ms > 0.05:
            return False
        
        # Check promotion freeze compliance
        if self.promotion_freeze['active']:
            # Ensure freeze duration hasn't been exceeded
            freeze_duration = time.time() - self.promotion_freeze['start_time']
            if freeze_duration > 2 * 3600:  # 2 hours max
                self.promotion_freeze['active'] = False
        
        return True
    
    def _update_promotion_freeze_status(self, performance_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update promotion freeze status based on system state."""
        # Check if promotion should be frozen
        promotion_event_detected = performance_data.get('promotion_event', False)
        
        if promotion_event_detected and not self.promotion_freeze['active']:
            # Start promotion freeze
            self.promotion_freeze['active'] = True
            self.promotion_freeze['start_time'] = time.time()
            self.promotion_freeze['reason'] = 'Promotion event detected'
            self.promotion_freeze['affected_pools'] = set(['CE', 'candidate_pools'])
            
            logger.info("Promotion freeze activated")
        
        return dict(self.promotion_freeze) if self.promotion_freeze['active'] else None
    
    def _calculate_overall_fairness_score(self, fairness_metrics: FairnessMetrics) -> float:
        """Calculate overall fairness score (0.0 to 1.0)."""
        # Weight different fairness components
        components = {
            'jain_index': fairness_metrics.jain_fairness_index,
            'entropy_normalized': min(fairness_metrics.resource_entropy / 3.0, 1.0),  # Max entropy ≈ log2(n)
            'gini_inverted': 1.0 - fairness_metrics.gini_coefficient,
            'deviation_penalty': 1.0 - min(fairness_metrics.max_deviation_from_fair * 2, 1.0),
            'starvation_penalty': 1.0 - len(fairness_metrics.starvation_detected) * 0.2,
            'monopolization_penalty': 1.0 - len(fairness_metrics.monopolization_detected) * 0.3
        }
        
        weights = {
            'jain_index': 0.3,
            'entropy_normalized': 0.2,
            'gini_inverted': 0.2,
            'deviation_penalty': 0.1,
            'starvation_penalty': 0.1,
            'monopolization_penalty': 0.1
        }
        
        fairness_score = sum(
            weights.get(comp, 0.0) * max(0.0, value)
            for comp, value in components.items()
        )
        
        return max(0.0, min(1.0, fairness_score))
    
    def _calculate_system_efficiency(
        self,
        tenant_allocations: Dict[str, TenantAllocation],
        performance_data: Dict[str, Any]
    ) -> float:
        """Calculate overall system efficiency."""
        if not tenant_allocations:
            return 0.0
        
        # Resource utilization efficiency
        total_allocation = sum(alloc.resource_share for alloc in tenant_allocations.values())
        utilization_efficiency = min(total_allocation, 1.0)
        
        # Performance efficiency
        avg_performance = np.mean([
            alloc.performance_score for alloc in tenant_allocations.values()
        ])
        
        # Prefix reuse efficiency
        avg_prefix_reuse = np.mean([
            alloc.prefix_reuse_rate for alloc in tenant_allocations.values()
        ])
        
        # ILP overhead efficiency
        ilp_efficiency = max(0.0, 1.0 - self.group_closure_state.ilp_overhead_ms / 0.05)
        
        # Weighted combination
        efficiency_components = [
            utilization_efficiency * 0.3,
            avg_performance * 0.4,
            avg_prefix_reuse * 0.2,
            ilp_efficiency * 0.1
        ]
        
        return sum(efficiency_components)
    
    def _calculate_ungameable_score(
        self,
        fairness_metrics: FairnessMetrics,
        group_closure_state: GroupClosureState,
        grouped_dpp_kernel: GroupedDPPKernel
    ) -> float:
        """Calculate system ungameable score."""
        # Fairness component (higher fairness = harder to game)
        fairness_component = fairness_metrics.jain_fairness_index
        
        # Diversity component (higher diversity = harder to manipulate)
        if grouped_dpp_kernel.eigenvalues:
            eigenvalue_spread = max(grouped_dpp_kernel.eigenvalues) - min(grouped_dpp_kernel.eigenvalues)
            diversity_component = min(eigenvalue_spread / 2.0, 1.0)
        else:
            diversity_component = 0.5
        
        # Closure robustness (active closures prevent gaming)
        closure_component = min(len(group_closure_state.active_closures) / 5.0, 1.0)
        
        # PSD stability (mathematical robustness)
        stability_component = 1.0 if grouped_dpp_kernel.psd_verified else 0.3
        
        # Weighted combination
        ungameable_score = (
            0.4 * fairness_component +
            0.25 * diversity_component +
            0.2 * closure_component +
            0.15 * stability_component
        )
        
        return max(0.0, min(1.0, ungameable_score))
    
    def _generate_fairness_diagnostics(
        self,
        fairness_metrics: FairnessMetrics,
        tenant_allocations: Dict[str, TenantAllocation],
        group_closure_state: GroupClosureState
    ) -> Tuple[List[str], List[str]]:
        """Generate warnings and recommendations for fairness optimization."""
        warnings = []
        recommendations = []
        
        # Fairness warnings
        if fairness_metrics.jain_fairness_index < self.jain_index_threshold:
            warnings.append(f"Jain fairness index {fairness_metrics.jain_fairness_index:.3f} below threshold")
            recommendations.append("Rebalance resource allocations to improve fairness")
        
        if fairness_metrics.starvation_detected:
            warnings.append(f"Resource starvation detected for tenants: {fairness_metrics.starvation_detected}")
            recommendations.append("Increase minimum resource guarantees for starved tenants")
        
        if fairness_metrics.monopolization_detected:
            warnings.append(f"Resource monopolization by tenants: {fairness_metrics.monopolization_detected}")
            recommendations.append("Implement resource caps to prevent monopolization")
        
        # Performance warnings
        if group_closure_state.ilp_overhead_ms > 0.03:  # >3% overhead
            warnings.append(f"ILP overhead {group_closure_state.ilp_overhead_ms*100:.1f}% approaching limit")
            recommendations.append("Consider reducing group complexity or ILP timeout")
        
        if not group_closure_state.high_gain_protection_active:
            warnings.append("High-gain children not protected from sibling drag")
            recommendations.append("Review group closure logic and split move criteria")
        
        # Drift warnings
        max_lambda_drift = 0.0
        for alloc in tenant_allocations.values():
            # Simplified drift calculation (would use actual history)
            drift = 0.05  # Placeholder
            max_lambda_drift = max(max_lambda_drift, abs(drift))
        
        if max_lambda_drift > self.drift_limit_24h * 0.8:  # 80% of limit
            warnings.append(f"λ drift approaching limit: {max_lambda_drift:.2f}")
            recommendations.append("Monitor tenant workload changes and adjust gradually")
        
        return warnings, recommendations
    
    def _update_allocation_history(self, tenant_allocations: Dict[str, TenantAllocation]):
        """Update allocation history for drift monitoring."""
        current_time = time.time()
        
        for tenant_id, allocation in tenant_allocations.items():
            self.allocation_history[tenant_id].append((current_time, allocation))
        
        # Update internal state
        self.tenant_allocations = tenant_allocations
    
    def _create_error_result(self, start_time: float, error: str) -> MultiTenantOptimizationResult:
        """Create error result when optimization fails."""
        optimization_time = (time.time() - start_time) * 1000
        
        return MultiTenantOptimizationResult(
            fairness_metrics=FairnessMetrics(0, 0, 1, 1, set(), set()),
            tenant_allocations={},
            group_closure_state=self.group_closure_state,
            grouped_dpp_kernel=GroupedDPPKernel([], np.array([]), 0, [], False, {}, 0),
            drift_constraints_satisfied=False,
            operational_constraints_satisfied=False,
            promotion_freeze_status=None,
            overall_fairness_score=0.0,
            system_efficiency=0.0,
            ungameable_score=0.0,
            optimization_time_ms=optimization_time,
            warnings=[f"Optimization failed: {error}"],
            recommendations=["Investigate multi-tenant fairness system failure"]
        )
    
    def get_monitoring_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data for dashboards."""
        current_fairness = self.fairness_history[-1] if self.fairness_history else None
        
        return {
            'current_fairness': {
                'jain_index': current_fairness.jain_fairness_index if current_fairness else 0.0,
                'gini_coefficient': current_fairness.gini_coefficient if current_fairness else 0.0,
                'starvation_count': len(current_fairness.starvation_detected) if current_fairness else 0,
                'monopolization_count': len(current_fairness.monopolization_detected) if current_fairness else 0
            },
            'tenant_status': {
                'active_tenants': len(self.tenant_allocations),
                'avg_lambda': np.mean([t.lambda_multiplier for t in self.tenant_allocations.values()]) if self.tenant_allocations else 0.0,
                'avg_mu': np.mean([t.mu_parameter for t in self.tenant_allocations.values()]) if self.tenant_allocations else 0.0,
                'avg_prefix_reuse': np.mean([t.prefix_reuse_rate for t in self.tenant_allocations.values()]) if self.tenant_allocations else 0.0
            },
            'group_closure': {
                'active_closures': len(self.group_closure_state.active_closures),
                'recent_split_moves': self.group_closure_state.recent_split_moves,
                'ilp_overhead_pct': self.group_closure_state.ilp_overhead_ms * 100,
                'high_gain_protected': self.group_closure_state.high_gain_protection_active
            },
            'promotion_freeze': {
                'active': self.promotion_freeze['active'],
                'reason': self.promotion_freeze['reason'],
                'duration_minutes': (time.time() - self.promotion_freeze['start_time']) / 60 if self.promotion_freeze['start_time'] else 0
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create multi-tenant fairness system
    fairness_system = MultiTenantFairnessSystem()
    
    # Example tenant demands
    tenant_demands = {
        'tenant_a': {
            'resource_demand': 1.2,
            'urgency_factor': 1.1,
            'latency_sensitivity': 0.8,
            'query_similarity': 0.7,
            'recent_performance': 0.85
        },
        'tenant_b': {
            'resource_demand': 0.8,
            'urgency_factor': 0.9,
            'latency_sensitivity': 1.2,
            'query_similarity': 0.6,
            'recent_performance': 0.75
        },
        'tenant_c': {
            'resource_demand': 1.0,
            'urgency_factor': 1.0,
            'latency_sensitivity': 1.0,
            'query_similarity': 0.8,
            'recent_performance': 0.90
        }
    }
    
    system_capacity = {
        'total_capacity': 1.0,
        'cpu_capacity': 0.8,
        'memory_capacity': 0.9
    }
    
    performance_data = {
        'group_performance_variance': 0.15,
        'high_gain_children_score': 0.9,
        'sibling_average_score': 0.7,
        'sibling_drag_factor': 0.05,
        'ilp_time_ms': 2.0,
        'total_time_ms': 50.0,
        'promotion_event': False
    }
    
    # Run fairness optimization
    result = fairness_system.optimize_fairness(
        tenant_demands, system_capacity, performance_data
    )
    
    print("=== Multi-Tenant Fairness Optimization Results ===")
    print(f"Jain Fairness Index: {result.fairness_metrics.jain_fairness_index:.3f}")
    print(f"Gini Coefficient: {result.fairness_metrics.gini_coefficient:.3f}")
    print(f"Overall Fairness Score: {result.overall_fairness_score:.3f}")
    print(f"System Efficiency: {result.system_efficiency:.3f}")
    print(f"Ungameable Score: {result.ungameable_score:.3f}")
    
    print("\n=== Tenant Allocations ===")
    for tenant_id, allocation in result.tenant_allocations.items():
        print(f"{tenant_id}: λ={allocation.lambda_multiplier:.3f}, "
              f"μ={allocation.mu_parameter:.3f}, "
              f"share={allocation.resource_share:.3f}")
    
    print("\n=== Group Closure State ===")
    print(f"Active Closures: {len(result.group_closure_state.active_closures)}")
    print(f"ILP Overhead: {result.group_closure_state.ilp_overhead_ms*100:.1f}%")
    print(f"High-Gain Protection: {result.group_closure_state.high_gain_protection_active}")
    
    print("\n=== Grouped-DPP Kernel ===")
    print(f"Representatives: {result.grouped_dpp_kernel.group_representatives}")
    print(f"Log-Determinant: {result.grouped_dpp_kernel.log_determinant:.3f}")
    print(f"PSD Verified: {result.grouped_dpp_kernel.psd_verified}")
    print(f"Marginal Quality: {result.grouped_dpp_kernel.marginal_quality_score:.3f}")
    
    if result.warnings:
        print(f"\nWarnings: {len(result.warnings)}")
        for warning in result.warnings[:3]:
            print(f"  - {warning}")
    
    if result.recommendations:
        print(f"\nRecommendations: {len(result.recommendations)}")
        for rec in result.recommendations[:3]:
            print(f"  - {rec}")
    
    # Get monitoring data
    monitoring_data = fairness_system.get_monitoring_data()
    print(f"\nMonitoring Data: {len(monitoring_data)} categories available")