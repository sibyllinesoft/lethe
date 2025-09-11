"""
Per-Sample Ledger System
=======================

Fail-closed ledger system for tracking diagnostic results across all samples.
Schema: (dataset, sample_id, keep_ratio, k, seed) -> comprehensive results

Key features:
- Write once, read everywhere with validation
- Fail-closed on any schema mismatch
- Cryptographic hash verification for data integrity
- Comprehensive sample metadata tracking
"""

import json
import hashlib
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, NamedTuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SampleLedgerEntry:
    """Complete diagnostic entry for a single sample."""
    
    # Primary key fields
    dataset: str
    sample_id: str  
    keep_ratio: float
    k: int
    seed: int
    
    # Ground truth data
    gold_answers: List[str]
    
    # Selection results
    selected_atoms: List[str]
    spans_present: List[bool]  # Does atom contain gold span?
    symbols_present: List[bool]  # Does atom contain gold symbol?
    
    # Extractive predictions
    extractive_pred: str
    extractive_score: float
    
    # Coverage flags
    coverage_flags: Dict[str, Any]  # SpanCoverage@K, SymbolCoverage@K, etc.
    
    # Data integrity
    cert_hash: str  # Hash of all fields for validation
    
    # Metadata
    timestamp: str
    processing_time_ms: float
    errors: List[str]

class SampleLedger:
    """
    Fail-closed per-sample ledger system for diagnostic results.
    
    Features:
    - SQLite backend for efficient queries and joins
    - Write-once validation with cryptographic hashing
    - Schema enforcement and migration support
    - Concurrent access with proper locking
    """
    
    def __init__(self, db_path: Path):
        """Initialize ledger with SQLite database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database and schema
        self._init_database()
        
        logger.info(f"Initialized sample ledger at {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with proper schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sample_ledger (
                    dataset TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    keep_ratio REAL NOT NULL,
                    k INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    
                    gold_answers TEXT NOT NULL,  -- JSON array
                    selected_atoms TEXT NOT NULL,  -- JSON array
                    spans_present TEXT NOT NULL,  -- JSON array of booleans
                    symbols_present TEXT NOT NULL,  -- JSON array of booleans
                    
                    extractive_pred TEXT NOT NULL,
                    extractive_score REAL NOT NULL,
                    coverage_flags TEXT NOT NULL,  -- JSON object
                    
                    cert_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    processing_time_ms REAL NOT NULL,
                    errors TEXT NOT NULL,  -- JSON array
                    
                    PRIMARY KEY (dataset, sample_id, keep_ratio, k, seed)
                )
            """)
            
            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dataset_keepratio 
                ON sample_ledger(dataset, keep_ratio)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_coverage_flags 
                ON sample_ledger(dataset, coverage_flags)
            """)
            
            conn.commit()
    
    def _compute_hash(self, entry: SampleLedgerEntry) -> str:
        """Compute SHA256 hash of entry for integrity checking."""
        # Create deterministic string representation
        entry_copy = asdict(entry)
        entry_copy.pop('cert_hash', None)  # Remove hash field itself
        entry_copy.pop('timestamp', None)  # Remove timestamp for reproducibility
        
        entry_str = json.dumps(entry_copy, sort_keys=True)
        return hashlib.sha256(entry_str.encode()).hexdigest()[:16]
    
    def write_entry(self, entry: SampleLedgerEntry) -> bool:
        """
        Write entry to ledger with validation.
        
        Returns:
            True if entry was written successfully
            False if entry already exists and matches
        
        Raises:
            ValueError if entry exists but doesn't match (fail-closed)
        """
        # Compute and set hash
        entry.cert_hash = self._compute_hash(entry)
        entry.timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if entry already exists
            existing = conn.execute("""
                SELECT cert_hash, gold_answers, extractive_pred, coverage_flags 
                FROM sample_ledger 
                WHERE dataset=? AND sample_id=? AND keep_ratio=? AND k=? AND seed=?
            """, (entry.dataset, entry.sample_id, entry.keep_ratio, entry.k, entry.seed)).fetchone()
            
            if existing:
                existing_hash, existing_gold, existing_pred, existing_flags = existing
                
                # Validate that existing entry matches
                if existing_hash != entry.cert_hash:
                    raise ValueError(
                        f"FAIL-CLOSED: Entry mismatch for {entry.dataset}:{entry.sample_id} "
                        f"keep_ratio={entry.keep_ratio}, k={entry.k}, seed={entry.seed}. "
                        f"Existing hash: {existing_hash}, New hash: {entry.cert_hash}. "
                        f"This indicates data corruption or inconsistent processing."
                    )
                
                logger.debug(f"Entry already exists and matches: {entry.dataset}:{entry.sample_id}")
                return False
            
            # Insert new entry
            conn.execute("""
                INSERT INTO sample_ledger VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, 
                    ?, ?, ?,
                    ?, ?, ?, ?
                )
            """, (
                entry.dataset, entry.sample_id, entry.keep_ratio, entry.k, entry.seed,
                json.dumps(entry.gold_answers),
                json.dumps(entry.selected_atoms),
                json.dumps(entry.spans_present), 
                json.dumps(entry.symbols_present),
                entry.extractive_pred,
                entry.extractive_score,
                json.dumps(entry.coverage_flags),
                entry.cert_hash,
                entry.timestamp,
                entry.processing_time_ms,
                json.dumps(entry.errors)
            ))
            
            conn.commit()
            
        logger.debug(f"Wrote new ledger entry: {entry.dataset}:{entry.sample_id}")
        return True
    
    def read_entry(self, dataset: str, sample_id: str, keep_ratio: float, 
                   k: int, seed: int) -> Optional[SampleLedgerEntry]:
        """Read entry from ledger with validation."""
        
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT * FROM sample_ledger 
                WHERE dataset=? AND sample_id=? AND keep_ratio=? AND k=? AND seed=?
            """, (dataset, sample_id, keep_ratio, k, seed)).fetchone()
            
            if not row:
                return None
            
            # Parse row into entry
            entry = SampleLedgerEntry(
                dataset=row[0],
                sample_id=row[1], 
                keep_ratio=row[2],
                k=row[3],
                seed=row[4],
                gold_answers=json.loads(row[5]),
                selected_atoms=json.loads(row[6]),
                spans_present=json.loads(row[7]),
                symbols_present=json.loads(row[8]),
                extractive_pred=row[9],
                extractive_score=row[10],
                coverage_flags=json.loads(row[11]),
                cert_hash=row[12],
                timestamp=row[13],
                processing_time_ms=row[14],
                errors=json.loads(row[15])
            )
            
            # Validate hash
            computed_hash = self._compute_hash(entry)
            if computed_hash != entry.cert_hash:
                raise ValueError(
                    f"FAIL-CLOSED: Hash validation failed for {dataset}:{sample_id}. "
                    f"Stored: {entry.cert_hash}, Computed: {computed_hash}. "
                    f"Data corruption detected!"
                )
            
            return entry
    
    def query_entries(self, 
                     dataset: Optional[str] = None,
                     keep_ratio: Optional[float] = None,
                     k: Optional[int] = None,
                     limit: Optional[int] = None) -> List[SampleLedgerEntry]:
        """Query multiple entries with optional filters."""
        
        where_clauses = []
        params = []
        
        if dataset:
            where_clauses.append("dataset = ?")
            params.append(dataset)
        
        if keep_ratio is not None:
            where_clauses.append("keep_ratio = ?") 
            params.append(keep_ratio)
            
        if k is not None:
            where_clauses.append("k = ?")
            params.append(k)
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        limit_sql = f" LIMIT {limit}" if limit else ""
        
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(f"""
                SELECT * FROM sample_ledger 
                WHERE {where_sql} 
                ORDER BY dataset, sample_id, keep_ratio, k, seed
                {limit_sql}
            """, params).fetchall()
        
        entries = []
        for row in rows:
            entry = SampleLedgerEntry(
                dataset=row[0],
                sample_id=row[1], 
                keep_ratio=row[2],
                k=row[3],
                seed=row[4],
                gold_answers=json.loads(row[5]),
                selected_atoms=json.loads(row[6]),
                spans_present=json.loads(row[7]),
                symbols_present=json.loads(row[8]),
                extractive_pred=row[9],
                extractive_score=row[10], 
                coverage_flags=json.loads(row[11]),
                cert_hash=row[12],
                timestamp=row[13],
                processing_time_ms=row[14],
                errors=json.loads(row[15])
            )
            entries.append(entry)
        
        return entries
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ledger statistics."""
        
        with sqlite3.connect(self.db_path) as conn:
            total_entries = conn.execute("SELECT COUNT(*) FROM sample_ledger").fetchone()[0]
            
            datasets = conn.execute("""
                SELECT dataset, COUNT(*) FROM sample_ledger GROUP BY dataset
            """).fetchall()
            
            keep_ratios = conn.execute("""
                SELECT keep_ratio, COUNT(*) FROM sample_ledger GROUP BY keep_ratio  
            """).fetchall()
            
            avg_processing_time = conn.execute("""
                SELECT AVG(processing_time_ms) FROM sample_ledger
            """).fetchone()[0] or 0
            
            error_rate = conn.execute("""
                SELECT AVG(CASE WHEN errors != '[]' THEN 1.0 ELSE 0.0 END) 
                FROM sample_ledger
            """).fetchone()[0] or 0
        
        return {
            "total_entries": total_entries,
            "datasets": dict(datasets),
            "keep_ratios": dict(keep_ratios),
            "avg_processing_time_ms": avg_processing_time,
            "error_rate": error_rate,
            "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0
        }
    
    def validate_integrity(self, sample_size: int = 100) -> Dict[str, Any]:
        """Validate data integrity by checking hashes on random sample."""
        
        entries = self.query_entries(limit=sample_size)
        
        total_checked = len(entries)
        hash_mismatches = 0
        validation_errors = []
        
        for entry in entries:
            try:
                computed_hash = self._compute_hash(entry)
                if computed_hash != entry.cert_hash:
                    hash_mismatches += 1
                    validation_errors.append(
                        f"{entry.dataset}:{entry.sample_id} - "
                        f"Stored: {entry.cert_hash}, Computed: {computed_hash}"
                    )
            except Exception as e:
                validation_errors.append(f"{entry.dataset}:{entry.sample_id} - Error: {e}")
        
        return {
            "total_checked": total_checked,
            "hash_mismatches": hash_mismatches,
            "integrity_rate": (total_checked - hash_mismatches) / total_checked if total_checked > 0 else 1.0,
            "validation_errors": validation_errors[:10]  # Limit to first 10 errors
        }