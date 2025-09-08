/**
 * Risk Budget Management System
 * Shadow-price consistency + CBU elasticity monitoring for production systems
 */

export interface RiskBudget {
  id: string;
  name: string;
  allocated_budget: number; // Total risk budget (error budget)
  consumed_budget: number; // Risk consumed so far
  remaining_budget: number; // Risk budget remaining
  shadow_price: number; // Cost of consuming additional risk
  elasticity_coefficient: number; // CBU elasticity measure
  period_start: number; // Budget period start timestamp
  period_end: number; // Budget period end timestamp
  thresholds: {
    warning: number; // Warning when budget consumption exceeds this %
    critical: number; // Critical when budget consumption exceeds this %
    emergency: number; // Emergency stop when budget consumption exceeds this %
  };
}

export interface CBUElasticityMetrics {
  timestamp: number;
  cbu_demand: number; // Current CBU demand
  cbu_price: number; // Current CBU shadow price
  elasticity: number; // Price elasticity of demand
  cross_elasticity: Record<string, number>; // Cross-elasticity with other resources
  substitution_effects: Record<string, number>; // Resource substitution effects
}

export interface ShadowPriceConsistency {
  resource: string;
  current_shadow_price: number;
  historical_shadow_price: number;
  consistency_score: number; // 0-1, where 1 is perfectly consistent
  arbitrage_opportunities: Array<{
    resource_pair: [string, string];
    price_differential: number;
    arbitrage_profit: number;
  }>;
}

export interface RiskBudgetConfig {
  budget_refresh_period_hours: number; // How often to refresh budgets
  shadow_price_window_hours: number; // Window for shadow price calculation
  elasticity_calculation_window_hours: number; // Window for elasticity calculation
  consistency_threshold: number; // Minimum consistency score (0.8)
  max_arbitrage_differential: number; // Max allowed arbitrage differential
}

/**
 * Shadow Price Calculator
 * Calculates and validates shadow prices for consistency across resources
 */
export class ShadowPriceCalculator {
  private config: RiskBudgetConfig;
  private priceHistory: Map<string, Array<{ timestamp: number; price: number; demand: number }>> = new Map();

  constructor(config: RiskBudgetConfig) {
    this.config = config;
  }

  /**
   * Calculate shadow price for a resource based on demand and constraints
   */
  calculateShadowPrice(
    resource: string,
    currentDemand: number,
    capacity: number,
    historicalData: Array<{ timestamp: number; demand: number; cost: number }>
  ): number {
    // Shadow price is the marginal cost of the most constraining resource
    const utilizationRatio = currentDemand / capacity;
    
    // Get historical price trend
    const historyKey = resource;
    const history = this.priceHistory.get(historyKey) || [];
    
    // Calculate base shadow price using lagrange multipliers approach
    let baseShadowPrice = 0;
    
    if (utilizationRatio > 0.8) {
      // High utilization - exponential pricing
      baseShadowPrice = Math.pow(utilizationRatio / 0.8, 3);
    } else if (utilizationRatio > 0.6) {
      // Medium utilization - quadratic pricing
      baseShadowPrice = Math.pow((utilizationRatio - 0.6) / 0.2, 2);
    } else {
      // Low utilization - linear pricing
      baseShadowPrice = utilizationRatio / 0.6;
    }
    
    // Adjust based on historical trends
    const trendAdjustment = this.calculateTrendAdjustment(history);
    const adjustedPrice = baseShadowPrice * (1 + trendAdjustment);
    
    // Store in history
    history.push({
      timestamp: Date.now(),
      price: adjustedPrice,
      demand: currentDemand
    });
    
    // Keep only recent history
    const cutoffTime = Date.now() - (this.config.shadow_price_window_hours * 60 * 60 * 1000);
    const recentHistory = history.filter(h => h.timestamp >= cutoffTime);
    this.priceHistory.set(historyKey, recentHistory);
    
    return adjustedPrice;
  }

  /**
   * Check shadow price consistency across resources
   */
  checkShadowPriceConsistency(
    resourcePrices: Record<string, number>
  ): ShadowPriceConsistency[] {
    const consistencyResults: ShadowPriceConsistency[] = [];
    
    for (const [resource, currentPrice] of Object.entries(resourcePrices)) {
      const history = this.priceHistory.get(resource) || [];
      const historicalPrice = history.length > 0 
        ? history.reduce((sum, h) => sum + h.price, 0) / history.length 
        : currentPrice;
      
      // Calculate consistency score
      const priceVariation = Math.abs(currentPrice - historicalPrice) / historicalPrice;
      const consistencyScore = Math.max(0, 1 - priceVariation);
      
      // Find arbitrage opportunities
      const arbitrageOpportunities = this.findArbitrageOpportunities(resource, currentPrice, resourcePrices);
      
      consistencyResults.push({
        resource,
        current_shadow_price: currentPrice,
        historical_shadow_price: historicalPrice,
        consistency_score: consistencyScore,
        arbitrage_opportunities: arbitrageOpportunities
      });
    }
    
    return consistencyResults;
  }

  private calculateTrendAdjustment(history: Array<{ timestamp: number; price: number; demand: number }>): number {
    if (history.length < 2) return 0;
    
    // Calculate price trend over recent history
    const recent = history.slice(-10); // Last 10 data points
    let trendSum = 0;
    
    for (let i = 1; i < recent.length; i++) {
      const priceChange = (recent[i].price - recent[i-1].price) / recent[i-1].price;
      trendSum += priceChange;
    }
    
    const averageTrend = trendSum / (recent.length - 1);
    
    // Limit trend adjustment to ±20%
    return Math.max(-0.2, Math.min(0.2, averageTrend));
  }

  private findArbitrageOpportunities(
    baseResource: string,
    basePrice: number,
    allPrices: Record<string, number>
  ): Array<{ resource_pair: [string, string]; price_differential: number; arbitrage_profit: number }> {
    const arbitrageOps: Array<{ resource_pair: [string, string]; price_differential: number; arbitrage_profit: number }> = [];
    
    for (const [otherResource, otherPrice] of Object.entries(allPrices)) {
      if (otherResource === baseResource) continue;
      
      const priceDifferential = Math.abs(basePrice - otherPrice) / Math.min(basePrice, otherPrice);
      
      if (priceDifferential > this.config.max_arbitrage_differential) {
        const arbitrageProfit = Math.abs(basePrice - otherPrice) * 0.1; // Assume 10% volume
        
        arbitrageOps.push({
          resource_pair: [baseResource, otherResource],
          price_differential: priceDifferential,
          arbitrage_profit: arbitrageProfit
        });
      }
    }
    
    return arbitrageOps;
  }
}

/**
 * CBU Elasticity Monitor
 * Monitors Context Budget Unit elasticity and substitution effects
 */
export class CBUElasticityMonitor {
  private config: RiskBudgetConfig;
  private elasticityHistory: Array<CBUElasticityMetrics> = [];

  constructor(config: RiskBudgetConfig) {
    this.config = config;
  }

  /**
   * Calculate CBU price elasticity of demand
   */
  calculateElasticity(
    demandHistory: Array<{ timestamp: number; cbu_demand: number; cbu_price: number }>,
    complementaryResources: Record<string, Array<{ timestamp: number; demand: number; price: number }>>
  ): CBUElasticityMetrics {
    if (demandHistory.length < 2) {
      throw new Error('Insufficient data for elasticity calculation');
    }

    // Sort by timestamp
    demandHistory.sort((a, b) => a.timestamp - b.timestamp);
    
    // Calculate price elasticity of demand
    const elasticity = this.calculatePriceElasticity(demandHistory);
    
    // Calculate cross-elasticities with complementary resources
    const crossElasticity: Record<string, number> = {};
    for (const [resource, data] of Object.entries(complementaryResources)) {
      if (data.length >= 2) {
        crossElasticity[resource] = this.calculateCrossElasticity(demandHistory, data);
      }
    }
    
    // Calculate substitution effects
    const substitutionEffects = this.calculateSubstitutionEffects(demandHistory, complementaryResources);
    
    const latest = demandHistory[demandHistory.length - 1];
    const metrics: CBUElasticityMetrics = {
      timestamp: Date.now(),
      cbu_demand: latest.cbu_demand,
      cbu_price: latest.cbu_price,
      elasticity,
      cross_elasticity: crossElasticity,
      substitution_effects: substitutionEffects
    };
    
    // Store in history
    this.elasticityHistory.push(metrics);
    
    // Keep only recent history
    const cutoffTime = Date.now() - (this.config.elasticity_calculation_window_hours * 60 * 60 * 1000);
    this.elasticityHistory = this.elasticityHistory.filter(h => h.timestamp >= cutoffTime);
    
    return metrics;
  }

  /**
   * Get elasticity trends over time
   */
  getElasticityTrends(): {
    current_elasticity: number;
    trend: number; // Positive = becoming more elastic, Negative = becoming more inelastic
    volatility: number; // Standard deviation of elasticity
    stability_score: number; // 0-1, where 1 is perfectly stable
  } {
    if (this.elasticityHistory.length < 2) {
      return {
        current_elasticity: 0,
        trend: 0,
        volatility: 0,
        stability_score: 1
      };
    }

    const elasticities = this.elasticityHistory.map(h => h.elasticity);
    const current = elasticities[elasticities.length - 1];
    
    // Calculate trend using linear regression
    const trend = this.calculateLinearTrend(elasticities);
    
    // Calculate volatility (standard deviation)
    const mean = elasticities.reduce((sum, e) => sum + e, 0) / elasticities.length;
    const variance = elasticities.reduce((sum, e) => sum + Math.pow(e - mean, 2), 0) / elasticities.length;
    const volatility = Math.sqrt(variance);
    
    // Stability score (inverse of coefficient of variation)
    const coefficientOfVariation = Math.abs(mean) > 0.001 ? volatility / Math.abs(mean) : 0;
    const stabilityScore = Math.max(0, 1 - coefficientOfVariation);
    
    return {
      current_elasticity: current,
      trend,
      volatility,
      stability_score: stabilityScore
    };
  }

  private calculatePriceElasticity(data: Array<{ cbu_demand: number; cbu_price: number }>): number {
    if (data.length < 2) return 0;
    
    let elasticitySum = 0;
    let validPairs = 0;
    
    for (let i = 1; i < data.length; i++) {
      const prev = data[i - 1];
      const curr = data[i];
      
      const demandChange = (curr.cbu_demand - prev.cbu_demand) / prev.cbu_demand;
      const priceChange = (curr.cbu_price - prev.cbu_price) / prev.cbu_price;
      
      if (Math.abs(priceChange) > 0.001) { // Avoid division by zero
        const elasticity = demandChange / priceChange;
        elasticitySum += elasticity;
        validPairs++;
      }
    }
    
    return validPairs > 0 ? elasticitySum / validPairs : 0;
  }

  private calculateCrossElasticity(
    cbuData: Array<{ cbu_demand: number; cbu_price: number }>,
    resourceData: Array<{ demand: number; price: number }>
  ): number {
    // Align data points by finding matching time periods (simplified)
    const minLength = Math.min(cbuData.length, resourceData.length);
    if (minLength < 2) return 0;
    
    let crossElasticitySum = 0;
    let validPairs = 0;
    
    for (let i = 1; i < minLength; i++) {
      const cbuDemandChange = (cbuData[i].cbu_demand - cbuData[i-1].cbu_demand) / cbuData[i-1].cbu_demand;
      const resourcePriceChange = (resourceData[i].price - resourceData[i-1].price) / resourceData[i-1].price;
      
      if (Math.abs(resourcePriceChange) > 0.001) {
        const crossElasticity = cbuDemandChange / resourcePriceChange;
        crossElasticitySum += crossElasticity;
        validPairs++;
      }
    }
    
    return validPairs > 0 ? crossElasticitySum / validPairs : 0;
  }

  private calculateSubstitutionEffects(
    cbuData: Array<{ cbu_demand: number; cbu_price: number }>,
    resources: Record<string, Array<{ demand: number; price: number }>>
  ): Record<string, number> {
    const substitutionEffects: Record<string, number> = {};
    
    for (const [resource, data] of Object.entries(resources)) {
      // Substitution effect is the change in demand for substitute good
      // when price of original good changes
      const minLength = Math.min(cbuData.length, data.length);
      if (minLength < 2) continue;
      
      let substitutionSum = 0;
      let validPairs = 0;
      
      for (let i = 1; i < minLength; i++) {
        const cbuPriceChange = (cbuData[i].cbu_price - cbuData[i-1].cbu_price) / cbuData[i-1].cbu_price;
        const resourceDemandChange = (data[i].demand - data[i-1].demand) / data[i-1].demand;
        
        if (Math.abs(cbuPriceChange) > 0.001) {
          const substitutionEffect = resourceDemandChange / cbuPriceChange;
          substitutionSum += substitutionEffect;
          validPairs++;
        }
      }
      
      substitutionEffects[resource] = validPairs > 0 ? substitutionSum / validPairs : 0;
    }
    
    return substitutionEffects;
  }

  private calculateLinearTrend(values: number[]): number {
    if (values.length < 2) return 0;
    
    const n = values.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    
    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += values[i];
      sumXY += i * values[i];
      sumXX += i * i;
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    return slope;
  }
}

/**
 * Risk Budget Manager
 * Manages risk budgets with shadow price consistency and CBU elasticity
 */
export class RiskBudgetManager {
  private config: RiskBudgetConfig;
  private shadowPriceCalculator: ShadowPriceCalculator;
  private cbuElasticityMonitor: CBUElasticityMonitor;
  private budgets: Map<string, RiskBudget> = new Map();
  private isRunning: boolean = false;

  constructor(config: RiskBudgetConfig) {
    this.config = config;
    this.shadowPriceCalculator = new ShadowPriceCalculator(config);
    this.cbuElasticityMonitor = new CBUElasticityMonitor(config);
  }

  /**
   * Create a new risk budget
   */
  createRiskBudget(
    id: string,
    name: string,
    allocatedBudget: number,
    durationHours: number = 24
  ): RiskBudget {
    const now = Date.now();
    const budget: RiskBudget = {
      id,
      name,
      allocated_budget: allocatedBudget,
      consumed_budget: 0,
      remaining_budget: allocatedBudget,
      shadow_price: 1.0, // Initial shadow price
      elasticity_coefficient: -1.0, // Initial elasticity (normal goods are negative)
      period_start: now,
      period_end: now + (durationHours * 60 * 60 * 1000),
      thresholds: {
        warning: 0.7, // 70%
        critical: 0.9, // 90%
        emergency: 0.95 // 95%
      }
    };
    
    this.budgets.set(id, budget);
    console.log(`📊 Created risk budget: ${name} (${allocatedBudget} units)`);
    
    return budget;
  }

  /**
   * Consume risk budget
   */
  consumeRiskBudget(budgetId: string, amount: number): {
    success: boolean;
    remaining: number;
    threshold_exceeded?: 'warning' | 'critical' | 'emergency';
    shadow_price: number;
  } {
    const budget = this.budgets.get(budgetId);
    if (!budget) {
      throw new Error(`Risk budget not found: ${budgetId}`);
    }
    
    // Check if budget period is still valid
    if (Date.now() > budget.period_end) {
      throw new Error(`Risk budget expired: ${budgetId}`);
    }
    
    // Update consumption
    const newConsumed = budget.consumed_budget + amount;
    const newRemaining = budget.allocated_budget - newConsumed;
    
    // Calculate new shadow price based on scarcity
    const utilizationRatio = newConsumed / budget.allocated_budget;
    const newShadowPrice = this.shadowPriceCalculator.calculateShadowPrice(
      budgetId,
      newConsumed,
      budget.allocated_budget,
      [] // Historical data would be passed here in a real implementation
    );
    
    // Check thresholds
    let thresholdExceeded: 'warning' | 'critical' | 'emergency' | undefined;
    if (utilizationRatio >= budget.thresholds.emergency) {
      thresholdExceeded = 'emergency';
    } else if (utilizationRatio >= budget.thresholds.critical) {
      thresholdExceeded = 'critical';
    } else if (utilizationRatio >= budget.thresholds.warning) {
      thresholdExceeded = 'warning';
    }
    
    // Update budget
    budget.consumed_budget = newConsumed;
    budget.remaining_budget = newRemaining;
    budget.shadow_price = newShadowPrice;
    
    const success = newRemaining >= 0;
    
    if (thresholdExceeded) {
      console.log(`⚠️ Risk budget threshold exceeded: ${budgetId} - ${thresholdExceeded} (${(utilizationRatio * 100).toFixed(1)}%)`);
    }
    
    return {
      success,
      remaining: newRemaining,
      threshold_exceeded: thresholdExceeded,
      shadow_price: newShadowPrice
    };
  }

  /**
   * Check shadow price consistency across all budgets
   */
  checkShadowPriceConsistency(): {
    overall_consistency: number;
    inconsistencies: ShadowPriceConsistency[];
    arbitrage_opportunities: number;
    recommendations: string[];
  } {
    const resourcePrices: Record<string, number> = {};
    for (const [id, budget] of this.budgets) {
      resourcePrices[id] = budget.shadow_price;
    }
    
    const consistencyResults = this.shadowPriceCalculator.checkShadowPriceConsistency(resourcePrices);
    
    // Calculate overall consistency
    const overallConsistency = consistencyResults.length > 0
      ? consistencyResults.reduce((sum, r) => sum + r.consistency_score, 0) / consistencyResults.length
      : 1.0;
    
    // Count arbitrage opportunities
    const arbitrageOpportunities = consistencyResults.reduce(
      (sum, r) => sum + r.arbitrage_opportunities.length, 0
    );
    
    // Generate recommendations
    const recommendations: string[] = [];
    
    if (overallConsistency < this.config.consistency_threshold) {
      recommendations.push('Shadow price consistency below threshold - review pricing model');
    }
    
    if (arbitrageOpportunities > 0) {
      recommendations.push(`${arbitrageOpportunities} arbitrage opportunities detected - rebalance prices`);
    }
    
    const inconsistentBudgets = consistencyResults.filter(r => r.consistency_score < 0.8);
    if (inconsistentBudgets.length > 0) {
      recommendations.push(`Review pricing for: ${inconsistentBudgets.map(r => r.resource).join(', ')}`);
    }
    
    return {
      overall_consistency: overallConsistency,
      inconsistencies: consistencyResults,
      arbitrage_opportunities: arbitrageOpportunities,
      recommendations
    };
  }

  /**
   * Update CBU elasticity monitoring
   */
  updateCBUElasticity(
    demandHistory: Array<{ timestamp: number; cbu_demand: number; cbu_price: number }>,
    complementaryResources: Record<string, Array<{ timestamp: number; demand: number; price: number }>>
  ): CBUElasticityMetrics {
    const elasticityMetrics = this.cbuElasticityMonitor.calculateElasticity(demandHistory, complementaryResources);
    
    // Update elasticity coefficient in budgets
    for (const budget of this.budgets.values()) {
      budget.elasticity_coefficient = elasticityMetrics.elasticity;
    }
    
    return elasticityMetrics;
  }

  /**
   * Get elasticity trends
   */
  getElasticityTrends(): ReturnType<CBUElasticityMonitor['getElasticityTrends']> {
    return this.cbuElasticityMonitor.getElasticityTrends();
  }

  /**
   * Refresh expired budgets
   */
  refreshExpiredBudgets(): string[] {
    const refreshedBudgets: string[] = [];
    const now = Date.now();
    
    for (const [id, budget] of this.budgets) {
      if (now > budget.period_end) {
        // Reset budget for new period
        budget.consumed_budget = 0;
        budget.remaining_budget = budget.allocated_budget;
        budget.period_start = now;
        budget.period_end = now + (this.config.budget_refresh_period_hours * 60 * 60 * 1000);
        budget.shadow_price = 1.0; // Reset to base price
        
        refreshedBudgets.push(id);
        console.log(`🔄 Refreshed risk budget: ${budget.name}`);
      }
    }
    
    return refreshedBudgets;
  }

  /**
   * Get all budgets status
   */
  getAllBudgets(): RiskBudget[] {
    return Array.from(this.budgets.values());
  }

  /**
   * Get budget by ID
   */
  getBudget(id: string): RiskBudget | undefined {
    return this.budgets.get(id);
  }

  /**
   * Delete budget
   */
  deleteBudget(id: string): boolean {
    return this.budgets.delete(id);
  }

  /**
   * Health check for risk budget system
   */
  healthCheck(): {
    healthy: boolean;
    issues: string[];
    metrics: {
      active_budgets: number;
      expired_budgets: number;
      budgets_over_warning: number;
      budgets_over_critical: number;
      overall_consistency: number;
      arbitrage_opportunities: number;
    };
  } {
    const issues: string[] = [];
    const now = Date.now();
    
    let activeBudgets = 0;
    let expiredBudgets = 0;
    let budgetsOverWarning = 0;
    let budgetsOverCritical = 0;
    
    for (const budget of this.budgets.values()) {
      if (now > budget.period_end) {
        expiredBudgets++;
      } else {
        activeBudgets++;
        
        const utilizationRatio = budget.consumed_budget / budget.allocated_budget;
        if (utilizationRatio >= budget.thresholds.critical) {
          budgetsOverCritical++;
        } else if (utilizationRatio >= budget.thresholds.warning) {
          budgetsOverWarning++;
        }
      }
    }
    
    const consistencyCheck = this.checkShadowPriceConsistency();
    
    // Check for issues
    if (expiredBudgets > 0) {
      issues.push(`${expiredBudgets} expired budgets need refresh`);
    }
    
    if (budgetsOverCritical > 0) {
      issues.push(`${budgetsOverCritical} budgets over critical threshold`);
    }
    
    if (consistencyCheck.overall_consistency < this.config.consistency_threshold) {
      issues.push('Shadow price consistency below threshold');
    }
    
    if (consistencyCheck.arbitrage_opportunities > 5) {
      issues.push('High number of arbitrage opportunities detected');
    }
    
    return {
      healthy: issues.length === 0,
      issues,
      metrics: {
        active_budgets: activeBudgets,
        expired_budgets: expiredBudgets,
        budgets_over_warning: budgetsOverWarning,
        budgets_over_critical: budgetsOverCritical,
        overall_consistency: consistencyCheck.overall_consistency,
        arbitrage_opportunities: consistencyCheck.arbitrage_opportunities
      }
    };
  }

  /**
   * Export risk budget report
   */
  exportReport(): {
    summary: {
      total_budgets: number;
      total_allocated: number;
      total_consumed: number;
      total_remaining: number;
      average_utilization: number;
      average_shadow_price: number;
    };
    budgets: Array<RiskBudget & {
      utilization_percentage: number;
      time_remaining_hours: number;
      status: 'healthy' | 'warning' | 'critical' | 'emergency' | 'expired';
    }>;
    consistency_analysis: ReturnType<RiskBudgetManager['checkShadowPriceConsistency']>;
    elasticity_trends: ReturnType<CBUElasticityMonitor['getElasticityTrends']>;
  } {
    const budgets = Array.from(this.budgets.values());
    const now = Date.now();
    
    const totalAllocated = budgets.reduce((sum, b) => sum + b.allocated_budget, 0);
    const totalConsumed = budgets.reduce((sum, b) => sum + b.consumed_budget, 0);
    const totalRemaining = budgets.reduce((sum, b) => sum + b.remaining_budget, 0);
    const averageUtilization = totalAllocated > 0 ? totalConsumed / totalAllocated : 0;
    const averageShadowPrice = budgets.length > 0
      ? budgets.reduce((sum, b) => sum + b.shadow_price, 0) / budgets.length
      : 0;
    
    const detailedBudgets = budgets.map(budget => {
      const utilizationPercentage = (budget.consumed_budget / budget.allocated_budget) * 100;
      const timeRemainingHours = Math.max(0, (budget.period_end - now) / (60 * 60 * 1000));
      
      let status: 'healthy' | 'warning' | 'critical' | 'emergency' | 'expired';
      if (now > budget.period_end) {
        status = 'expired';
      } else if (utilizationPercentage >= budget.thresholds.emergency * 100) {
        status = 'emergency';
      } else if (utilizationPercentage >= budget.thresholds.critical * 100) {
        status = 'critical';
      } else if (utilizationPercentage >= budget.thresholds.warning * 100) {
        status = 'warning';
      } else {
        status = 'healthy';
      }
      
      return {
        ...budget,
        utilization_percentage: utilizationPercentage,
        time_remaining_hours: timeRemainingHours,
        status
      };
    });
    
    return {
      summary: {
        total_budgets: budgets.length,
        total_allocated: totalAllocated,
        total_consumed: totalConsumed,
        total_remaining: totalRemaining,
        average_utilization: averageUtilization,
        average_shadow_price: averageShadowPrice
      },
      budgets: detailedBudgets,
      consistency_analysis: this.checkShadowPriceConsistency(),
      elasticity_trends: this.getElasticityTrends()
    };
  }
}