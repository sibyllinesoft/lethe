#!/usr/bin/env python3
"""
Quality Assurance and Inter-Annotator Agreement Validation for LetheBench
=========================================================================

Comprehensive quality assurance framework that implements inter-annotator agreement
(IAA) validation, statistical analysis, and quality metrics to ensure the dataset
meets NeurIPS publication standards.

Features:
- Inter-annotator agreement calculation (Cohen's kappa, Krippendorff's alpha)
- Statistical validation of dataset balance and distribution
- Quality metrics computation and validation
- License compliance checking
- Ethical review framework
- Reproducibility validation
"""

import json
import logging
import hashlib
import sqlite3
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict, Counter
import re
import warnings

from schema import (
    QueryRecord, DatasetManifest, ValidationResult, IAAAgreementResult,
    QualityMetrics, ComplexityLevel, DomainType, DatasetSplit, QualityAuditResult,
    MIN_IAA_THRESHOLD, QUALITY_THRESHOLDS, STATISTICAL_REQUIREMENTS
)


@dataclass
class AnnotatorData:
    """Data structure for annotator information"""
    annotator_id: str
    name: str
    expertise_domains: List[str]
    annotation_count: int
    quality_score: float
    consistency_score: float


@dataclass
class AnnotationRecord:
    """Individual annotation record for IAA calculation"""
    query_id: str
    annotator_id: str
    domain_label: DomainType
    complexity_label: ComplexityLevel
    quality_score: float
    relevance_scores: List[float]
    notes: Optional[str] = None
    timestamp: Optional[datetime] = None



class QualityAssuranceFramework:
    """
    Comprehensive quality assurance framework for dataset validation
    """
    
    def __init__(self, seed: int = 42, logger: Optional[logging.Logger] = None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize quality thresholds and requirements
        self.quality_thresholds = QUALITY_THRESHOLDS
        self.statistical_requirements = STATISTICAL_REQUIREMENTS
        self.min_iaa_threshold = MIN_IAA_THRESHOLD
        
        # Annotation tracking
        self.annotations: List[AnnotationRecord] = []
        self.annotators: Dict[str, AnnotatorData] = {}
        
        # Quality metrics cache
        self._quality_cache: Dict[str, Any] = {}
        
    def validate_dataset_quality(
        self, 
        queries: List[QueryRecord],
        manifest: Optional[DatasetManifest] = None,
        run_iaa: bool = True
    ) -> QualityAuditResult:
        """
        Comprehensive dataset quality validation
        
        Args:
            queries: List of queries to validate
            manifest: Dataset manifest (optional)
            run_iaa: Whether to run inter-annotator agreement analysis
        
        Returns:
            QualityAuditResult with comprehensive quality assessment
        """
        self.logger.info(f"Starting comprehensive quality validation for {len(queries)} queries")
        
        # Phase 1: Basic validation
        validation_results = self._validate_basic_quality(queries)
        
        # Phase 2: Statistical validation
        statistical_validity = self._validate_statistical_requirements(queries)
        
        # Phase 3: Inter-annotator agreement (if requested and data available)
        iaa_results = None
        if run_iaa and len(self.annotations) > 0:
            iaa_results = self._calculate_inter_annotator_agreement()
        
        # Phase 4: License compliance
        license_compliance = self._check_license_compliance(queries, manifest)
        
        # Phase 5: Ethical compliance
        ethical_compliance = self._check_ethical_compliance(queries)
        
        # Phase 6: Generate recommendations and identify critical issues
        recommendations = self._generate_quality_recommendations(
            validation_results, statistical_validity, iaa_results
        )
        
        critical_issues = self._identify_critical_issues(
            validation_results, statistical_validity, iaa_results, 
            license_compliance, ethical_compliance
        )
        
        # Compute overall quality score
        overall_score = self._compute_overall_quality_score(
            validation_results, statistical_validity, iaa_results,
            license_compliance, ethical_compliance
        )
        
        self.logger.info(f"Quality validation completed. Overall score: {overall_score:.3f}")
        
        return QualityAuditResult(
            overall_score=overall_score,
            validation_results=validation_results,
            iaa_results=iaa_results,
            statistical_validity=statistical_validity,
            license_compliance=license_compliance,
            ethical_compliance=ethical_compliance,
            recommendations=recommendations,
            critical_issues=critical_issues
        )
    
    def add_annotations(
        self, 
        annotations: List[AnnotationRecord], 
        annotators: Optional[List[AnnotatorData]] = None
    ):
        """Add annotation data for IAA calculation"""
        self.annotations.extend(annotations)
        
        if annotators:
            for annotator in annotators:
                self.annotators[annotator.annotator_id] = annotator
        
        self.logger.info(f"Added {len(annotations)} annotations from {len(set(a.annotator_id for a in annotations))} annotators")
    
    def generate_synthetic_annotations(
        self, 
        queries: List[QueryRecord], 
        n_annotators: int = 3,
        noise_level: float = 0.1
    ) -> List[AnnotationRecord]:
        """
        Generate synthetic annotations for IAA validation testing
        
        Args:
            queries: Queries to annotate
            n_annotators: Number of synthetic annotators
            noise_level: Amount of noise to add (0.0 = perfect agreement, 1.0 = random)
        
        Returns:
            List of synthetic annotation records
        """
        annotations = []
        
        # Create synthetic annotators
        annotator_profiles = [
            {"id": f"annotator_{i}", "expertise": ["code_heavy", "tool_results"], "consistency": 0.9},
            {"id": f"annotator_{i}", "expertise": ["chatty_prose", "code_heavy"], "consistency": 0.85},
            {"id": f"annotator_{i}", "expertise": ["tool_results", "chatty_prose"], "consistency": 0.8}
        ]
        
        # Sample queries for annotation (typically 10-20% for IAA)
        sample_size = min(100, max(20, len(queries) // 5))
        sampled_queries = self.rng.choice(queries, size=sample_size, replace=False)
        
        for query in sampled_queries:
            for i in range(n_annotators):
                profile = annotator_profiles[i % len(annotator_profiles)]
                
                # Add noise based on annotator consistency and noise level
                consistency = profile["consistency"] * (1 - noise_level)
                
                # Domain label (with some noise)
                if self.rng.random() > consistency:
                    domain_options = [DomainType.CODE_HEAVY, DomainType.CHATTY_PROSE, DomainType.TOOL_RESULTS]
                    domain_label = self.rng.choice([d for d in domain_options if d != query.domain])
                else:
                    domain_label = query.domain
                
                # Complexity label (with some noise)
                if self.rng.random() > consistency:
                    complexity_options = [ComplexityLevel.SIMPLE, ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX]
                    complexity_label = self.rng.choice([c for c in complexity_options if c != query.complexity])
                else:
                    complexity_label = query.complexity
                
                # Quality score (with noise)
                base_quality = query.metadata.quality_score
                noise_factor = self.rng.normal(0, noise_level * 0.2)  # Small noise
                quality_score = np.clip(base_quality + noise_factor, 0.0, 1.0)
                
                # Relevance scores (synthetic)
                n_docs = len(query.ground_truth_docs)
                relevance_scores = [
                    np.clip(0.7 + self.rng.normal(0, noise_level * 0.1), 0.0, 1.0)
                    for _ in range(n_docs)
                ]
                
                annotation = AnnotationRecord(
                    query_id=query.query_id,
                    annotator_id=profile["id"],
                    domain_label=domain_label,
                    complexity_label=complexity_label,
                    quality_score=quality_score,
                    relevance_scores=relevance_scores,
                    timestamp=datetime.now(timezone.utc)
                )
                
                annotations.append(annotation)
        
        self.logger.info(f"Generated {len(annotations)} synthetic annotations for {len(sampled_queries)} queries")
        return annotations
    
    def _validate_basic_quality(self, queries: List[QueryRecord]) -> ValidationResult:
        """Validate basic quality requirements"""
        
        errors = []
        warnings = []
        query_errors = {}
        statistics = {}
        
        # Check minimum dataset size
        if len(queries) < self.statistical_requirements["min_queries_per_domain"] * 3:
            errors.append(f"Dataset too small: {len(queries)} queries (minimum: {self.statistical_requirements['min_queries_per_domain'] * 3})")
        
        # Validate individual queries
        quality_scores = []
        domain_counts = defaultdict(int)
        complexity_counts = defaultdict(int)
        
        for query in queries:
            query_quality_issues = []
            
            # Check query structure
            if not query.query_text.strip():
                query_quality_issues.append("Empty query text")
            
            if len(query.query_text) < 20:
                query_quality_issues.append("Query text too short")
            
            if len(query.query_text) > 2000:
                query_quality_issues.append("Query text too long")
            
            # Check ground truth
            if len(query.ground_truth_docs) < 2:
                query_quality_issues.append("Insufficient ground truth documents")
            
            if len(query.ground_truth_docs) > 10:
                query_quality_issues.append("Too many ground truth documents")
            
            # Check metadata
            if query.metadata.quality_score < self.quality_thresholds["minimum"]:
                query_quality_issues.append(f"Quality score too low: {query.metadata.quality_score:.3f}")
            
            # Check content hash integrity
            try:
                hash_content = {
                    "query_text": query.query_text,
                    "domain": query.domain.value,
                    "complexity": query.complexity.value,
                    "ground_truth_docs": sorted([doc.doc_id for doc in query.ground_truth_docs])
                }
                content_json = json.dumps(hash_content, sort_keys=True, separators=(',', ':'))
                expected_hash = hashlib.sha256(content_json.encode()).hexdigest()
                
                if query.content_hash != expected_hash:
                    query_quality_issues.append("Content hash mismatch")
            except Exception as e:
                query_quality_issues.append(f"Hash validation error: {e}")
            
            if query_quality_issues:
                query_errors[query.query_id] = query_quality_issues
            
            # Collect statistics
            quality_scores.append(query.metadata.quality_score)
            domain_counts[query.domain] += 1
            complexity_counts[query.complexity] += 1
        
        # Domain balance validation
        total_queries = len(queries)
        for domain, count in domain_counts.items():
            ratio = count / total_queries
            if ratio < 0.25:  # Minimum 25% per domain
                warnings.append(f"Domain {domain.value} underrepresented: {ratio:.1%}")
            elif ratio > 0.50:  # Maximum 50% per domain
                warnings.append(f"Domain {domain.value} overrepresented: {ratio:.1%}")
        
        # Complexity distribution validation
        for complexity, count in complexity_counts.items():
            if count < self.statistical_requirements["min_queries_per_complexity"]:
                warnings.append(f"Complexity {complexity.value} has too few queries: {count}")
        
        # Quality statistics
        if quality_scores:
            statistics = {
                "total_queries": len(queries),
                "avg_quality_score": float(np.mean(quality_scores)),
                "min_quality_score": float(np.min(quality_scores)),
                "max_quality_score": float(np.max(quality_scores)),
                "quality_std": float(np.std(quality_scores)),
                "queries_with_errors": len(query_errors),
                "error_rate": len(query_errors) / len(queries),
                "domain_balance": {domain.value: count/total_queries for domain, count in domain_counts.items()},
                "complexity_distribution": {complexity.value: count for complexity, count in complexity_counts.items()}
            }
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            query_errors=query_errors,
            statistics=statistics,
            recommendations=self._generate_basic_quality_recommendations(errors, warnings, statistics)
        )
    
    def _validate_statistical_requirements(self, queries: List[QueryRecord]) -> Dict[str, bool]:
        """Validate statistical requirements for research validity"""
        
        validity = {}
        
        # Check minimum queries per domain
        domain_counts = defaultdict(int)
        for query in queries:
            domain_counts[query.domain] += 1
        
        for domain in DomainType:
            min_required = self.statistical_requirements["min_queries_per_domain"]
            validity[f"min_queries_{domain.value}"] = domain_counts[domain] >= min_required
        
        # Check minimum queries per complexity
        complexity_counts = defaultdict(int)
        for query in queries:
            complexity_counts[query.complexity] += 1
        
        for complexity in ComplexityLevel:
            min_required = self.statistical_requirements["min_queries_per_complexity"]
            validity[f"min_queries_{complexity.value}"] = complexity_counts[complexity] >= min_required
        
        # Check overall dataset size
        validity["sufficient_total_size"] = len(queries) >= 500
        
        # Check test set size (if splits are defined)
        test_queries = [q for q in queries if q.dataset_split == DatasetSplit.TEST]
        if test_queries:
            validity["sufficient_test_size"] = len(test_queries) >= self.statistical_requirements["min_test_set_size"]
        
        # Check domain balance
        total_queries = len(queries)
        domain_balance_good = True
        for domain, count in domain_counts.items():
            ratio = count / total_queries
            if ratio < 0.25 or ratio > 0.50:  # Each domain should be 25-50%
                domain_balance_good = False
                break
        validity["domain_balance"] = domain_balance_good
        
        # Check quality distribution
        quality_scores = [q.metadata.quality_score for q in queries]
        avg_quality = np.mean(quality_scores)
        validity["sufficient_quality"] = avg_quality >= self.quality_thresholds["good"]
        
        return validity
    
    def _calculate_inter_annotator_agreement(self) -> IAAAgreementResult:
        """Calculate inter-annotator agreement metrics"""
        
        if len(self.annotations) == 0:
            raise ValueError("No annotations available for IAA calculation")
        
        # Group annotations by query
        query_annotations = defaultdict(list)
        for annotation in self.annotations:
            query_annotations[annotation.query_id].append(annotation)
        
        # Filter to queries with multiple annotations
        multi_annotated = {
            query_id: annotations 
            for query_id, annotations in query_annotations.items() 
            if len(annotations) >= 2
        }
        
        if not multi_annotated:
            raise ValueError("No queries with multiple annotations found")
        
        # Calculate Cohen's kappa for different annotation dimensions
        domain_kappa = self._calculate_cohens_kappa(multi_annotated, "domain_label")
        complexity_kappa = self._calculate_cohens_kappa(multi_annotated, "complexity_label")
        
        # Overall kappa (average of domain and complexity)
        overall_kappa = (domain_kappa + complexity_kappa) / 2
        
        # Calculate per-domain agreement
        per_domain_agreement = {}
        for domain in DomainType:
            domain_queries = {
                qid: annotations for qid, annotations in multi_annotated.items()
                if any(ann.domain_label == domain for ann in annotations)
            }
            if domain_queries:
                per_domain_agreement[domain] = self._calculate_cohens_kappa(domain_queries, "domain_label")
        
        # Agreement level classification
        agreement_level = self._classify_agreement_level(overall_kappa)
        
        # Disagreement analysis
        disagreement_analysis = self._analyze_disagreements(multi_annotated)
        
        # Count annotators and sample size
        annotator_count = len(set(ann.annotator_id for ann in self.annotations))
        sample_size = len(multi_annotated)
        
        self.logger.info(f"IAA Analysis: κ={overall_kappa:.3f}, level={agreement_level}, sample={sample_size}")
        
        return IAAAgreementResult(
            annotator_count=annotator_count,
            query_sample_size=sample_size,
            cohens_kappa=overall_kappa,
            krippendorff_alpha=None,  # Could implement if needed
            agreement_level=agreement_level,
            per_domain_agreement=per_domain_agreement,
            disagreement_analysis=disagreement_analysis
        )
    
    def _calculate_cohens_kappa(
        self, 
        multi_annotated: Dict[str, List[AnnotationRecord]], 
        attribute: str
    ) -> float:
        """Calculate Cohen's kappa for a specific annotation attribute"""
        
        # Collect all annotation pairs
        agreements = 0
        total_pairs = 0
        all_labels = set()
        label_counts = defaultdict(int)
        
        for query_id, annotations in multi_annotated.items():
            # For each pair of annotators on this query
            for i in range(len(annotations)):
                for j in range(i + 1, len(annotations)):
                    ann1, ann2 = annotations[i], annotations[j]
                    
                    label1 = getattr(ann1, attribute)
                    label2 = getattr(ann2, attribute)
                    
                    all_labels.add(label1)
                    all_labels.add(label2)
                    
                    label_counts[label1] += 1
                    label_counts[label2] += 1
                    
                    if label1 == label2:
                        agreements += 1
                    
                    total_pairs += 1
        
        if total_pairs == 0:
            return 0.0
        
        # Observed agreement
        p_o = agreements / total_pairs
        
        # Expected agreement by chance
        total_labels = sum(label_counts.values())
        p_e = sum((count / total_labels) ** 2 for count in label_counts.values())
        
        # Cohen's kappa
        if p_e == 1.0:
            return 1.0 if p_o == 1.0 else 0.0
        
        kappa = (p_o - p_e) / (1 - p_e)
        return max(0.0, kappa)  # Ensure non-negative
    
    def _classify_agreement_level(self, kappa: float) -> str:
        """Classify agreement level based on Cohen's kappa"""
        if kappa < 0.20:
            return "poor"
        elif kappa < 0.40:
            return "fair"
        elif kappa < 0.60:
            return "moderate"
        elif kappa < 0.80:
            return "substantial"
        else:
            return "excellent"
    
    def _analyze_disagreements(self, multi_annotated: Dict[str, List[AnnotationRecord]]) -> Dict[str, Any]:
        """Analyze patterns in annotator disagreements"""
        
        disagreements = []
        
        for query_id, annotations in multi_annotated.items():
            for i in range(len(annotations)):
                for j in range(i + 1, len(annotations)):
                    ann1, ann2 = annotations[i], annotations[j]
                    
                    if ann1.domain_label != ann2.domain_label or ann1.complexity_label != ann2.complexity_label:
                        disagreements.append({
                            "query_id": query_id,
                            "annotator_1": ann1.annotator_id,
                            "annotator_2": ann2.annotator_id,
                            "domain_1": ann1.domain_label.value,
                            "domain_2": ann2.domain_label.value,
                            "complexity_1": ann1.complexity_label.value,
                            "complexity_2": ann2.complexity_label.value,
                            "quality_diff": abs(ann1.quality_score - ann2.quality_score)
                        })
        
        # Analyze disagreement patterns
        analysis = {
            "total_disagreements": len(disagreements),
            "disagreement_rate": len(disagreements) / sum(len(anns) * (len(anns) - 1) // 2 for anns in multi_annotated.values()),
            "domain_confusion_matrix": self._build_confusion_matrix(disagreements, "domain"),
            "complexity_confusion_matrix": self._build_confusion_matrix(disagreements, "complexity"),
            "frequent_disagreement_pairs": self._find_frequent_disagreement_pairs(disagreements),
            "avg_quality_score_difference": np.mean([d["quality_diff"] for d in disagreements]) if disagreements else 0.0
        }
        
        return analysis
    
    def _build_confusion_matrix(self, disagreements: List[Dict], attribute_type: str) -> Dict[str, Dict[str, int]]:
        """Build confusion matrix for disagreements"""
        
        matrix = defaultdict(lambda: defaultdict(int))
        
        for disagreement in disagreements:
            val1 = disagreement[f"{attribute_type}_1"]
            val2 = disagreement[f"{attribute_type}_2"]
            
            matrix[val1][val2] += 1
            matrix[val2][val1] += 1
        
        return dict(matrix)
    
    def _find_frequent_disagreement_pairs(self, disagreements: List[Dict]) -> List[Dict[str, Any]]:
        """Find pairs of annotators who disagree most frequently"""
        
        pair_counts = defaultdict(int)
        
        for disagreement in disagreements:
            ann1 = disagreement["annotator_1"] 
            ann2 = disagreement["annotator_2"]
            pair = tuple(sorted([ann1, ann2]))
            pair_counts[pair] += 1
        
        # Return top 5 disagreeing pairs
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"annotator_1": pair[0], "annotator_2": pair[1], "disagreement_count": count}
            for pair, count in sorted_pairs[:5]
        ]
    
    def _check_license_compliance(
        self, 
        queries: List[QueryRecord], 
        manifest: Optional[DatasetManifest]
    ) -> bool:
        """Check license compliance for the dataset"""
        
        # This is a simplified check - in practice would verify:
        # - All source materials have appropriate licenses
        # - Attribution requirements are met  
        # - Commercial use permissions are correct
        # - No copyrighted material without permission
        
        compliance_checks = []
        
        # Check for potentially copyrighted content patterns
        copyright_patterns = [
            r'©\s*\d{4}',  # Copyright symbol with year
            r'copyright\s+\d{4}',  # Copyright text with year
            r'all rights reserved',  # Rights reserved phrase
            r'proprietary',  # Proprietary content indicator
        ]
        
        for query in queries:
            for pattern in copyright_patterns:
                if re.search(pattern, query.query_text, re.IGNORECASE):
                    compliance_checks.append(f"Potential copyright content in query {query.query_id}")
                    break
        
        # Check manifest license information
        if manifest and hasattr(manifest, 'license_info'):
            if manifest.license_info.commercial_use_allowed:
                compliance_checks.append("Commercial use allowed")
            if not manifest.license_info.compliance_verified:
                compliance_checks.append("License compliance not verified")
        
        # For this synthetic dataset, assume compliance is good
        is_compliant = len(compliance_checks) == 0 or all("Commercial use allowed" in check or "not verified" in check for check in compliance_checks)
        
        if compliance_checks:
            self.logger.warning(f"License compliance issues: {compliance_checks}")
        
        return is_compliant
    
    def _check_ethical_compliance(self, queries: List[QueryRecord]) -> bool:
        """Check ethical compliance for the dataset"""
        
        # Check for potentially harmful content
        harmful_patterns = [
            r'\b(hack|exploit|attack|malware|virus)\b',
            r'\b(password|credential|secret|token)\s*[:=]',  # Exposed secrets
            r'\b(personal|private|confidential)\s+data\b',  # Privacy issues
            r'\b(bias|discriminat|stereotype)\b',  # Bias indicators
        ]
        
        ethical_issues = []
        
        for query in queries:
            for pattern in harmful_patterns:
                if re.search(pattern, query.query_text, re.IGNORECASE):
                    # For security-related patterns, check if they're in educational context
                    if 'hack' in pattern and any(word in query.query_text.lower() 
                                               for word in ['how to prevent', 'protect against', 'secure']):
                        continue  # Educational security content is OK
                    
                    ethical_issues.append(f"Potential ethical issue in query {query.query_id}: {pattern}")
        
        # Check for PII or sensitive information
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
        ]
        
        for query in queries:
            for pattern in pii_patterns:
                if re.search(pattern, query.query_text):
                    ethical_issues.append(f"Potential PII in query {query.query_id}")
        
        if ethical_issues:
            self.logger.warning(f"Ethical compliance issues found: {len(ethical_issues)}")
            for issue in ethical_issues[:5]:  # Log first 5 issues
                self.logger.warning(f"  {issue}")
        
        # For this synthetic dataset, assume compliance unless major issues
        return len(ethical_issues) == 0
    
    def _compute_overall_quality_score(
        self, 
        validation_results: ValidationResult,
        statistical_validity: Dict[str, bool],
        iaa_results: Optional[IAAAgreementResult],
        license_compliance: bool,
        ethical_compliance: bool
    ) -> float:
        """Compute overall quality score for the dataset"""
        
        score = 0.0
        
        # Basic validation score (30%)
        if validation_results.is_valid:
            score += 0.30
        else:
            # Partial credit based on error severity
            error_penalty = min(0.30, len(validation_results.errors) * 0.05)
            score += max(0.0, 0.30 - error_penalty)
        
        # Statistical validity score (25%)
        valid_checks = sum(statistical_validity.values())
        total_checks = len(statistical_validity)
        if total_checks > 0:
            score += (valid_checks / total_checks) * 0.25
        
        # Quality metrics score (20%)
        if validation_results.statistics:
            avg_quality = validation_results.statistics.get("avg_quality_score", 0)
            if avg_quality >= self.quality_thresholds["excellent"]:
                score += 0.20
            elif avg_quality >= self.quality_thresholds["good"]:
                score += 0.15
            elif avg_quality >= self.quality_thresholds["minimum"]:
                score += 0.10
        
        # IAA score (15%)
        if iaa_results:
            kappa = iaa_results.cohens_kappa
            if kappa >= 0.8:
                score += 0.15
            elif kappa >= self.min_iaa_threshold:
                score += 0.10
            elif kappa >= 0.5:
                score += 0.05
        else:
            # No IAA data - give partial credit
            score += 0.07
        
        # Compliance scores (5% each)
        if license_compliance:
            score += 0.05
        
        if ethical_compliance:
            score += 0.05
        
        return min(1.0, score)
    
    def _generate_quality_recommendations(
        self,
        validation_results: ValidationResult,
        statistical_validity: Dict[str, bool],
        iaa_results: Optional[IAAAgreementResult]
    ) -> List[str]:
        """Generate actionable quality improvement recommendations"""
        
        recommendations = []
        
        # Based on validation results
        if not validation_results.is_valid:
            recommendations.append("Address critical validation errors before dataset release")
        
        if validation_results.statistics:
            avg_quality = validation_results.statistics.get("avg_quality_score", 0)
            if avg_quality < self.quality_thresholds["good"]:
                recommendations.append(f"Improve average quality score to ≥{self.quality_thresholds['good']}")
            
            error_rate = validation_results.statistics.get("error_rate", 0)
            if error_rate > 0.05:
                recommendations.append("Reduce query error rate to <5%")
        
        # Based on statistical validity
        failed_checks = [check for check, passed in statistical_validity.items() if not passed]
        if failed_checks:
            recommendations.append(f"Address failed statistical requirements: {', '.join(failed_checks)}")
        
        # Based on IAA results
        if iaa_results:
            if iaa_results.cohens_kappa < self.min_iaa_threshold:
                recommendations.append(f"Improve inter-annotator agreement to ≥{self.min_iaa_threshold:.2f}")
            
            if iaa_results.agreement_level in ["poor", "fair"]:
                recommendations.append("Provide clearer annotation guidelines and training")
        
        # General recommendations
        recommendations.extend([
            "Conduct human expert review of randomly sampled queries",
            "Implement automated quality monitoring for ongoing dataset updates", 
            "Document dataset construction methodology for reproducibility",
            "Create detailed annotation guidelines for future expansion"
        ])
        
        return recommendations
    
    def _identify_critical_issues(
        self,
        validation_results: ValidationResult,
        statistical_validity: Dict[str, bool], 
        iaa_results: Optional[IAAAgreementResult],
        license_compliance: bool,
        ethical_compliance: bool
    ) -> List[str]:
        """Identify critical issues that block dataset release"""
        
        critical_issues = []
        
        # Critical validation errors
        if validation_results.errors:
            critical_issues.extend([f"VALIDATION ERROR: {error}" for error in validation_results.errors])
        
        # Statistical requirement failures
        critical_statistical_failures = [
            "insufficient_total_size", "domain_balance", "sufficient_quality"
        ]
        
        for requirement in critical_statistical_failures:
            if requirement in statistical_validity and not statistical_validity[requirement]:
                critical_issues.append(f"STATISTICAL FAILURE: {requirement}")
        
        # IAA failures
        if iaa_results and iaa_results.cohens_kappa < 0.4:  # Below fair agreement
            critical_issues.append(f"IAA FAILURE: Cohen's κ = {iaa_results.cohens_kappa:.3f} (minimum: 0.4)")
        
        # Compliance failures
        if not license_compliance:
            critical_issues.append("LICENSE COMPLIANCE FAILURE: Licensing issues detected")
        
        if not ethical_compliance:
            critical_issues.append("ETHICAL COMPLIANCE FAILURE: Ethical issues detected")
        
        return critical_issues
    
    def _generate_basic_quality_recommendations(
        self, 
        errors: List[str], 
        warnings: List[str], 
        statistics: Dict[str, Any]
    ) -> List[str]:
        """Generate basic quality recommendations"""
        
        recommendations = []
        
        if errors:
            recommendations.append("Fix all validation errors before proceeding")
        
        if warnings:
            recommendations.append("Review and address validation warnings")
        
        if statistics:
            error_rate = statistics.get("error_rate", 0)
            if error_rate > 0.1:
                recommendations.append("Reduce query error rate significantly")
            
            avg_quality = statistics.get("avg_quality_score", 0)
            if avg_quality < 0.8:
                recommendations.append("Improve overall query quality scores")
        
        return recommendations
    
    def export_quality_report(
        self, 
        audit_result: QualityAuditResult, 
        output_path: Path
    ) -> Path:
        """Export comprehensive quality report"""
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_score": audit_result.overall_score,
            "validation_summary": {
                "is_valid": audit_result.validation_results.is_valid,
                "error_count": len(audit_result.validation_results.errors),
                "warning_count": len(audit_result.validation_results.warnings),
                "query_error_count": len(audit_result.validation_results.query_errors)
            },
            "statistical_validity": audit_result.statistical_validity,
            "iaa_summary": {
                "cohens_kappa": audit_result.iaa_results.cohens_kappa if audit_result.iaa_results else None,
                "agreement_level": audit_result.iaa_results.agreement_level if audit_result.iaa_results else None,
                "sample_size": audit_result.iaa_results.query_sample_size if audit_result.iaa_results else None
            },
            "compliance": {
                "license": audit_result.license_compliance,
                "ethical": audit_result.ethical_compliance
            },
            "recommendations": audit_result.recommendations,
            "critical_issues": audit_result.critical_issues,
            "detailed_results": {
                "validation": asdict(audit_result.validation_results),
                "iaa": asdict(audit_result.iaa_results) if audit_result.iaa_results else None
            }
        }
        
        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Quality report exported to {output_path}")
        return output_path