#!/usr/bin/env python3
"""
Centralized SQLite Database Schema for Lethe Experiments
======================================================

Consolidates all experimental data into a single SQLite database for
simplified data aggregation and analysis. Replaces multiple JSON/CSV
output formats with a unified relational schema.

Tables:
- experiments: Top-level experiment metadata
- configurations: Parameter configurations tested
- runs: Individual experimental runs
- queries: Query execution results
- metrics: Computed evaluation metrics
- statistical_tests: Hypothesis testing results

Benefits:
- Single source of truth for all experimental data
- Efficient querying and aggregation 
- Simplified analysis pipeline
- Better data integrity and consistency
"""

import sqlite3
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Database schema version for migration support
SCHEMA_VERSION = "1.0.0"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_path: str
    enable_foreign_keys: bool = True
    enable_wal_mode: bool = True
    timeout_seconds: int = 30

class ExperimentDatabase:
    """Centralized SQLite database for experiment results"""
    
    def __init__(self, db_path: str, config: Optional[DatabaseConfig] = None):
        self.db_path = Path(db_path)
        self.config = config or DatabaseConfig(str(db_path))
        self.logger = logging.getLogger(__name__)
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize database with complete schema"""
        with sqlite3.connect(self.db_path, timeout=self.config.timeout_seconds) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            if self.config.enable_wal_mode:
                conn.execute("PRAGMA journal_mode = WAL")
            
            # Create all tables
            self._create_schema(conn)
            self._create_indices(conn)
            self._create_views(conn)
            
            # Insert schema metadata
            conn.execute("""
                INSERT OR REPLACE INTO schema_metadata (key, value)
                VALUES ('version', ?)
            """, (SCHEMA_VERSION,))
            
            conn.commit()
            
    def _create_schema(self, conn: sqlite3.Connection):
        """Create complete database schema"""
        
        # Schema metadata
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Experiments table - top level experiment metadata
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                description TEXT,
                config_path TEXT,
                git_commit_sha TEXT,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                status TEXT CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
                total_runs INTEGER DEFAULT 0,
                completed_runs INTEGER DEFAULT 0,
                failed_runs INTEGER DEFAULT 0,
                artifacts_path TEXT,
                mlflow_run_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Configurations table - parameter combinations tested
        conn.execute("""
            CREATE TABLE IF NOT EXISTS configurations (
                config_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                config_name TEXT NOT NULL,
                config_type TEXT CHECK(config_type IN ('lethe', 'baseline')),
                parameters_json TEXT NOT NULL,
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        """)
        
        # Runs table - individual experimental runs
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                config_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                complexity TEXT NOT NULL,
                session_length TEXT NOT NULL,
                replication INTEGER NOT NULL,
                status TEXT CHECK(status IN ('pending', 'running', 'completed', 'timeout', 'error')),
                start_time DATETIME,
                end_time DATETIME,
                runtime_seconds REAL,
                peak_memory_mb REAL,
                timeout_seconds INTEGER,
                error_message TEXT,
                artifacts_path TEXT,
                mlflow_run_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id),
                FOREIGN KEY (config_id) REFERENCES configurations (config_id)
            )
        """)
        
        # Queries table - individual query execution results
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                query_execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                query_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                domain TEXT NOT NULL,
                complexity TEXT NOT NULL,
                ground_truth_docs_json TEXT NOT NULL, -- JSON array
                retrieved_docs_json TEXT NOT NULL,    -- JSON array  
                relevance_scores_json TEXT NOT NULL,  -- JSON array
                latency_ms REAL NOT NULL,
                memory_mb REAL NOT NULL,
                entities_covered_json TEXT,           -- JSON array
                contradictions_json TEXT,             -- JSON array
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """)
        
        # Metrics table - computed evaluation metrics per run
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                metric_category TEXT NOT NULL, -- 'quality', 'efficiency', 'robustness', 'adaptivity'
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                confidence_interval_lower REAL,
                confidence_interval_upper REAL,
                sample_size INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """)
        
        # Metric aggregates table - pre-computed aggregated metrics
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metric_aggregates (
                aggregate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                config_id TEXT,
                domain TEXT,
                complexity TEXT,
                grouping_keys TEXT, -- JSON object for flexible grouping
                metric_category TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                aggregate_function TEXT NOT NULL, -- 'mean', 'median', 'std', 'min', 'max'
                aggregate_value REAL NOT NULL,
                sample_size INTEGER NOT NULL,
                confidence_interval_lower REAL,
                confidence_interval_upper REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id),
                FOREIGN KEY (config_id) REFERENCES configurations (config_id)
            )
        """)
        
        # Statistical tests table - hypothesis testing results
        conn.execute("""
            CREATE TABLE IF NOT EXISTS statistical_tests (
                test_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                hypothesis_id TEXT NOT NULL,
                test_name TEXT NOT NULL,
                baseline_config TEXT NOT NULL,
                treatment_config TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                test_statistic REAL,
                p_value REAL NOT NULL,
                adjusted_p_value REAL,
                effect_size REAL,
                effect_size_ci_lower REAL,
                effect_size_ci_upper REAL,
                significant BOOLEAN NOT NULL,
                power_analysis REAL,
                sample_size_baseline INTEGER,
                sample_size_treatment INTEGER,
                test_metadata_json TEXT, -- Additional test parameters
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        """)
        
        # Data quality table - validation and integrity checks
        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_checks (
                check_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                check_type TEXT NOT NULL, -- 'fraud_detection', 'outlier_analysis', 'consistency'
                check_name TEXT NOT NULL,
                target_table TEXT,
                target_run_id TEXT,
                status TEXT CHECK(status IN ('passed', 'failed', 'warning')),
                message TEXT,
                details_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        """)
        
    def _create_indices(self, conn: sqlite3.Connection):
        """Create performance indices"""
        
        indices = [
            # Experiments
            "CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name)",
            "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)",
            "CREATE INDEX IF NOT EXISTS idx_experiments_start_time ON experiments(start_time)",
            
            # Configurations
            "CREATE INDEX IF NOT EXISTS idx_configurations_experiment ON configurations(experiment_id)",
            "CREATE INDEX IF NOT EXISTS idx_configurations_type ON configurations(config_type)",
            "CREATE INDEX IF NOT EXISTS idx_configurations_name ON configurations(config_name)",
            
            # Runs
            "CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id)",
            "CREATE INDEX IF NOT EXISTS idx_runs_config ON runs(config_id)",
            "CREATE INDEX IF NOT EXISTS idx_runs_domain ON runs(domain)",
            "CREATE INDEX IF NOT EXISTS idx_runs_complexity ON runs(complexity)",
            "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)",
            "CREATE INDEX IF NOT EXISTS idx_runs_start_time ON runs(start_time)",
            
            # Queries
            "CREATE INDEX IF NOT EXISTS idx_queries_run ON queries(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_queries_query_id ON queries(query_id)",
            "CREATE INDEX IF NOT EXISTS idx_queries_domain ON queries(domain)",
            "CREATE INDEX IF NOT EXISTS idx_queries_latency ON queries(latency_ms)",
            
            # Metrics
            "CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_category ON metrics(metric_category)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_value ON metrics(metric_value)",
            
            # Metric aggregates
            "CREATE INDEX IF NOT EXISTS idx_aggregates_experiment ON metric_aggregates(experiment_id)",
            "CREATE INDEX IF NOT EXISTS idx_aggregates_config ON metric_aggregates(config_id)",
            "CREATE INDEX IF NOT EXISTS idx_aggregates_metric ON metric_aggregates(metric_category, metric_name)",
            
            # Statistical tests
            "CREATE INDEX IF NOT EXISTS idx_tests_experiment ON statistical_tests(experiment_id)",
            "CREATE INDEX IF NOT EXISTS idx_tests_hypothesis ON statistical_tests(hypothesis_id)",
            "CREATE INDEX IF NOT EXISTS idx_tests_configs ON statistical_tests(baseline_config, treatment_config)",
            "CREATE INDEX IF NOT EXISTS idx_tests_metric ON statistical_tests(metric_name)",
            
            # Data quality
            "CREATE INDEX IF NOT EXISTS idx_quality_experiment ON data_quality_checks(experiment_id)",
            "CREATE INDEX IF NOT EXISTS idx_quality_type ON data_quality_checks(check_type)",
            "CREATE INDEX IF NOT EXISTS idx_quality_status ON data_quality_checks(status)"
        ]
        
        for index_sql in indices:
            conn.execute(index_sql)
            
    def _create_views(self, conn: sqlite3.Connection):
        """Create convenience views for common queries"""
        
        # Run results summary view
        conn.execute("""
            CREATE VIEW IF NOT EXISTS run_results_summary AS
            SELECT 
                r.run_id,
                r.experiment_id,
                c.config_name,
                c.config_type,
                r.domain,
                r.complexity,
                r.session_length,
                r.status,
                r.runtime_seconds,
                r.peak_memory_mb,
                COUNT(q.query_execution_id) as query_count,
                AVG(q.latency_ms) as avg_latency_ms,
                AVG(q.memory_mb) as avg_memory_mb,
                r.start_time,
                r.end_time
            FROM runs r
            LEFT JOIN configurations c ON r.config_id = c.config_id
            LEFT JOIN queries q ON r.run_id = q.run_id
            GROUP BY r.run_id, r.experiment_id, c.config_name, c.config_type,
                     r.domain, r.complexity, r.session_length, r.status,
                     r.runtime_seconds, r.peak_memory_mb, r.start_time, r.end_time
        """)
        
        # Experiment progress view
        conn.execute("""
            CREATE VIEW IF NOT EXISTS experiment_progress AS
            SELECT
                e.experiment_id,
                e.name,
                e.status,
                e.start_time,
                e.total_runs,
                e.completed_runs,
                e.failed_runs,
                ROUND(100.0 * e.completed_runs / e.total_runs, 2) as completion_percentage,
                COUNT(DISTINCT c.config_id) as unique_configurations,
                COUNT(DISTINCT r.domain) as domains_tested,
                AVG(r.runtime_seconds) as avg_run_time_seconds
            FROM experiments e
            LEFT JOIN configurations c ON e.experiment_id = c.experiment_id
            LEFT JOIN runs r ON e.experiment_id = r.experiment_id AND r.status = 'completed'
            GROUP BY e.experiment_id, e.name, e.status, e.start_time,
                     e.total_runs, e.completed_runs, e.failed_runs
        """)
        
        # Metrics summary view
        conn.execute("""
            CREATE VIEW IF NOT EXISTS metrics_summary AS
            SELECT
                m.run_id,
                r.experiment_id,
                c.config_name,
                r.domain,
                r.complexity,
                m.metric_category,
                m.metric_name,
                m.metric_value,
                m.metric_unit,
                m.confidence_interval_lower,
                m.confidence_interval_upper
            FROM metrics m
            JOIN runs r ON m.run_id = r.run_id
            JOIN configurations c ON r.config_id = c.config_id
        """)
        
    def insert_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """Insert new experiment record"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiments (
                    experiment_id, name, version, description, config_path,
                    git_commit_sha, start_time, status, artifacts_path, mlflow_run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_data['experiment_id'],
                experiment_data['name'],
                experiment_data['version'],
                experiment_data.get('description'),
                experiment_data.get('config_path'),
                experiment_data.get('git_commit_sha'),
                experiment_data['start_time'],
                experiment_data.get('status', 'running'),
                experiment_data.get('artifacts_path'),
                experiment_data.get('mlflow_run_id')
            ))
            
        return experiment_data['experiment_id']
        
    def insert_configuration(self, config_data: Dict[str, Any]) -> str:
        """Insert configuration record"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO configurations (
                    config_id, experiment_id, config_name, config_type,
                    parameters_json, description
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                config_data['config_id'],
                config_data['experiment_id'],
                config_data['config_name'],
                config_data['config_type'],
                json.dumps(config_data['parameters']),
                config_data.get('description')
            ))
            
        return config_data['config_id']
        
    def insert_run(self, run_data: Dict[str, Any]) -> str:
        """Insert run record"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO runs (
                    run_id, experiment_id, config_id, domain, complexity,
                    session_length, replication, status, start_time, end_time,
                    runtime_seconds, peak_memory_mb, timeout_seconds,
                    error_message, artifacts_path, mlflow_run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_data['run_id'],
                run_data['experiment_id'],
                run_data['config_id'],
                run_data['domain'],
                run_data['complexity'],
                run_data['session_length'],
                run_data['replication'],
                run_data.get('status', 'pending'),
                run_data.get('start_time'),
                run_data.get('end_time'),
                run_data.get('runtime_seconds'),
                run_data.get('peak_memory_mb'),
                run_data.get('timeout_seconds'),
                run_data.get('error_message'),
                run_data.get('artifacts_path'),
                run_data.get('mlflow_run_id')
            ))
            
        return run_data['run_id']
        
    def insert_query_result(self, query_data: Dict[str, Any]) -> int:
        """Insert query execution result"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO queries (
                    run_id, query_id, session_id, query_text, domain, complexity,
                    ground_truth_docs_json, retrieved_docs_json, relevance_scores_json,
                    latency_ms, memory_mb, entities_covered_json, contradictions_json,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_data['run_id'],
                query_data['query_id'],
                query_data['session_id'],
                query_data['query_text'],
                query_data['domain'],
                query_data['complexity'],
                json.dumps(query_data['ground_truth_docs']),
                json.dumps(query_data['retrieved_docs']),
                json.dumps(query_data['relevance_scores']),
                query_data['latency_ms'],
                query_data['memory_mb'],
                json.dumps(query_data.get('entities_covered', [])),
                json.dumps(query_data.get('contradictions', [])),
                query_data['timestamp']
            ))
            
        return cursor.lastrowid
        
    def insert_metrics(self, metrics_data: List[Dict[str, Any]]):
        """Insert multiple metrics for a run"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO metrics (
                    run_id, metric_category, metric_name, metric_value,
                    metric_unit, confidence_interval_lower, confidence_interval_upper,
                    sample_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    metric['run_id'],
                    metric['metric_category'],
                    metric['metric_name'],
                    metric['metric_value'],
                    metric.get('metric_unit'),
                    metric.get('confidence_interval_lower'),
                    metric.get('confidence_interval_upper'),
                    metric.get('sample_size')
                ) for metric in metrics_data
            ])
            
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment summary"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Basic experiment info
            experiment = conn.execute("""
                SELECT * FROM experiments WHERE experiment_id = ?
            """, (experiment_id,)).fetchone()
            
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
                
            # Get run statistics
            stats = conn.execute("""
                SELECT
                    COUNT(*) as total_runs,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_runs,
                    COUNT(CASE WHEN status = 'failed' OR status = 'error' THEN 1 END) as failed_runs,
                    AVG(runtime_seconds) as avg_runtime,
                    AVG(peak_memory_mb) as avg_memory
                FROM runs WHERE experiment_id = ?
            """, (experiment_id,)).fetchone()
            
            # Get configuration counts
            config_stats = conn.execute("""
                SELECT
                    config_type,
                    COUNT(*) as count
                FROM configurations WHERE experiment_id = ?
                GROUP BY config_type
            """, (experiment_id,)).fetchall()
            
            return {
                'experiment': dict(experiment),
                'statistics': dict(stats) if stats else {},
                'configuration_counts': {row['config_type']: row['count'] for row in config_stats}
            }
            
    def get_leaderboard(self, experiment_id: str, metric_name: str = 'ndcg_at_10') -> List[Dict[str, Any]]:
        """Get configuration leaderboard for a specific metric"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            results = conn.execute("""
                SELECT
                    c.config_name,
                    c.config_type,
                    AVG(m.metric_value) as avg_score,
                    COUNT(m.metric_value) as sample_size,
                    MIN(m.metric_value) as min_score,
                    MAX(m.metric_value) as max_score,
                    AVG(m.confidence_interval_lower) as avg_ci_lower,
                    AVG(m.confidence_interval_upper) as avg_ci_upper
                FROM metrics m
                JOIN runs r ON m.run_id = r.run_id
                JOIN configurations c ON r.config_id = c.config_id
                WHERE r.experiment_id = ? AND m.metric_name = ?
                GROUP BY c.config_id, c.config_name, c.config_type
                ORDER BY avg_score DESC
            """, (experiment_id, metric_name)).fetchall()
            
            return [dict(row) for row in results]
            
    def close(self):
        """Close database connection"""
        pass  # Using context managers, no persistent connection
        
def create_experiment_database(db_path: str, config: Optional[DatabaseConfig] = None) -> ExperimentDatabase:
    """Factory function to create and initialize experiment database"""
    return ExperimentDatabase(db_path, config)

# Example usage and migration utilities
if __name__ == "__main__":
    # Example of creating a database
    db = create_experiment_database("example_experiments.db")
    
    # Example experiment data
    experiment_data = {
        'experiment_id': 'test_experiment_001',
        'name': 'Lethe Evaluation Test',
        'version': '1.0.0',
        'description': 'Test experiment for schema validation',
        'start_time': '2024-08-23T12:00:00Z',
        'status': 'running',
        'git_commit_sha': 'abc123def456'
    }
    
    db.insert_experiment(experiment_data)
    print(f"Created experiment database at: example_experiments.db")
    print("Schema ready for data ingestion!")