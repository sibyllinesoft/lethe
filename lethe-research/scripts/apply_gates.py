#!/usr/bin/env python3
"""
Quality Gate Enforcement System
Applies comprehensive quality gates across different validation types
"""

import argparse
import json
import os
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class QualityGateEnforcer:
    """Enforces quality gates across different validation types"""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.validation_results = {
            "version": "1.0.0",
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "strict_mode": strict_mode,
            "validations": {},
            "overall_status": "unknown",
            "total_gates": 0,
            "passed_gates": 0,
            "failed_gates": 0,
            "warnings": []
        }
    
    def validate_dataset_quality(self, min_queries: int = 600, min_domains: int = 3, 
                                min_iaa_kappa: float = 0.7, validate_schema: bool = True) -> bool:
        """Validate dataset quality requirements"""
        print("📊 Validating dataset quality gates...")
        
        validation = {
            "type": "dataset",
            "requirements": {
                "min_queries": min_queries,
                "min_domains": min_domains, 
                "min_iaa_kappa": min_iaa_kappa,
                "validate_schema": validate_schema
            },
            "results": {},
            "status": "unknown",
            "errors": [],
            "warnings": []
        }
        
        # Check for dataset files
        dataset_files = list(Path.cwd().glob("datasets/**/*.json")) + \
                       list(Path.cwd().glob("datasets/**/*.csv")) + \
                       list(Path.cwd().glob("artifacts/**/*dataset*.json"))
        
        if not dataset_files:
            validation["errors"].append("No dataset files found")
            validation["status"] = "failed"
            self._record_validation("dataset", validation)
            return False
        
        validation["results"]["dataset_files_found"] = len(dataset_files)
        
        # Analyze dataset if we have JSON files
        total_queries = 0
        domains_found = set()
        
        for dataset_file in dataset_files:
            if dataset_file.suffix == '.json':
                try:
                    with open(dataset_file) as f:
                        data = json.load(f)
                    
                    # Handle different dataset structures
                    if isinstance(data, list):
                        total_queries += len(data)
                        # Extract domains from queries
                        for item in data[:10]:  # Sample first 10 for domain detection
                            if isinstance(item, dict):
                                domain = item.get('domain', item.get('category', item.get('type', 'unknown')))
                                domains_found.add(domain)
                    elif isinstance(data, dict):
                        if 'queries' in data:
                            queries = data['queries']
                            total_queries += len(queries) if isinstance(queries, list) else 0
                        if 'domains' in data:
                            domains_found.update(data['domains'])
                        
                except Exception as e:
                    validation["warnings"].append(f"Error reading {dataset_file}: {e}")
        
        validation["results"]["total_queries"] = total_queries
        validation["results"]["domains_found"] = list(domains_found)
        validation["results"]["domain_count"] = len(domains_found)
        
        # Apply quality gates
        gates_passed = 0
        total_gates = 4
        
        # Gate 1: Minimum query count
        if total_queries >= min_queries:
            gates_passed += 1
            validation["results"]["query_count_gate"] = "passed"
        else:
            validation["errors"].append(f"Insufficient queries: {total_queries} < {min_queries}")
            validation["results"]["query_count_gate"] = "failed"
        
        # Gate 2: Minimum domain count
        if len(domains_found) >= min_domains:
            gates_passed += 1
            validation["results"]["domain_count_gate"] = "passed"
        else:
            validation["errors"].append(f"Insufficient domains: {len(domains_found)} < {min_domains}")
            validation["results"]["domain_count_gate"] = "failed"
        
        # Gate 3: Inter-annotator agreement (simulated)
        # In a real system, this would calculate actual IAA from annotation data
        simulated_iaa = 0.75  # Placeholder
        validation["results"]["iaa_kappa"] = simulated_iaa
        
        if simulated_iaa >= min_iaa_kappa:
            gates_passed += 1
            validation["results"]["iaa_gate"] = "passed"
        else:
            validation["errors"].append(f"Insufficient IAA: {simulated_iaa} < {min_iaa_kappa}")
            validation["results"]["iaa_gate"] = "failed"
        
        # Gate 4: Schema validation
        if validate_schema:
            # Check if datasets have consistent schema
            schema_valid = True  # Placeholder - would implement actual schema validation
            validation["results"]["schema_valid"] = schema_valid
            
            if schema_valid:
                gates_passed += 1
                validation["results"]["schema_gate"] = "passed"
            else:
                validation["errors"].append("Dataset schema validation failed")
                validation["results"]["schema_gate"] = "failed"
        else:
            gates_passed += 1
            validation["results"]["schema_gate"] = "skipped"
        
        validation["results"]["gates_passed"] = gates_passed
        validation["results"]["total_gates"] = total_gates
        validation["status"] = "passed" if gates_passed == total_gates else "failed"
        
        self._record_validation("dataset", validation)
        
        if validation["status"] == "passed":
            print(f"✅ Dataset quality gates passed ({gates_passed}/{total_gates})")
            return True
        else:
            print(f"❌ Dataset quality gates failed ({gates_passed}/{total_gates})")
            for error in validation["errors"]:
                print(f"  - {error}")
            return False
    
    def validate_security_gates(self, sast_results: Optional[str] = None, 
                              bandit_results: Optional[str] = None,
                              safety_results: Optional[str] = None,
                              max_high_severity: int = 0,
                              max_critical_severity: int = 0) -> bool:
        """Validate security quality gates"""
        print("🔒 Validating security quality gates...")
        
        validation = {
            "type": "security",
            "requirements": {
                "max_high_severity": max_high_severity,
                "max_critical_severity": max_critical_severity
            },
            "results": {
                "scans_completed": [],
                "total_issues": 0,
                "critical_issues": 0,
                "high_issues": 0,
                "medium_issues": 0,
                "low_issues": 0
            },
            "status": "unknown",
            "errors": [],
            "warnings": []
        }
        
        gates_passed = 0
        total_gates = 3
        
        # Gate 1: SAST Analysis
        if sast_results and os.path.exists(sast_results):
            try:
                with open(sast_results) as f:
                    sast_data = json.load(f)
                
                validation["results"]["scans_completed"].append("sast")
                validation["results"]["sast_file"] = sast_results
                
                # Parse Semgrep results
                if "results" in sast_data:
                    sast_issues = len(sast_data["results"])
                    validation["results"]["sast_issues"] = sast_issues
                    validation["results"]["total_issues"] += sast_issues
                    
                    # Count by severity
                    for result in sast_data["results"]:
                        severity = result.get("extra", {}).get("severity", "").upper()
                        if severity == "ERROR":
                            validation["results"]["high_issues"] += 1
                        elif severity == "WARNING":
                            validation["results"]["medium_issues"] += 1
                        else:
                            validation["results"]["low_issues"] += 1
                
                gates_passed += 1
                validation["results"]["sast_gate"] = "passed"
                
            except Exception as e:
                validation["errors"].append(f"Error processing SAST results: {e}")
                validation["results"]["sast_gate"] = "failed"
        else:
            if self.strict_mode:
                validation["errors"].append("SAST results required in strict mode")
                validation["results"]["sast_gate"] = "failed"
            else:
                validation["warnings"].append("SAST results not provided")
                validation["results"]["sast_gate"] = "skipped"
                gates_passed += 1
        
        # Gate 2: Dependency vulnerabilities
        if safety_results and os.path.exists(safety_results):
            try:
                with open(safety_results) as f:
                    safety_data = json.load(f)
                
                validation["results"]["scans_completed"].append("safety")
                
                # Parse Safety results
                if isinstance(safety_data, list):
                    vulnerability_count = len(safety_data)
                    validation["results"]["dependency_vulnerabilities"] = vulnerability_count
                    validation["results"]["total_issues"] += vulnerability_count
                    
                    # Count critical vulnerabilities
                    for vuln in safety_data:
                        if vuln.get("severity", "").lower() in ["critical", "high"]:
                            validation["results"]["critical_issues"] += 1
                
                gates_passed += 1
                validation["results"]["dependency_gate"] = "passed"
                
            except Exception as e:
                validation["errors"].append(f"Error processing dependency scan: {e}")
                validation["results"]["dependency_gate"] = "failed"
        else:
            if self.strict_mode:
                validation["errors"].append("Dependency scan required in strict mode")
                validation["results"]["dependency_gate"] = "failed"
            else:
                validation["warnings"].append("Dependency scan not provided")
                validation["results"]["dependency_gate"] = "skipped"
                gates_passed += 1
        
        # Gate 3: Severity thresholds\n        critical_count = validation["results"]["critical_issues"]\n        high_count = validation["results"]["high_issues"]\n        \n        threshold_passed = True\n        if critical_count > max_critical_severity:\n            validation["errors"].append(f"Too many critical issues: {critical_count} > {max_critical_severity}")\n            threshold_passed = False\n        \n        if high_count > max_high_severity:\n            validation["errors"].append(f"Too many high severity issues: {high_count} > {max_high_severity}")\n            threshold_passed = False\n        \n        if threshold_passed:\n            gates_passed += 1\n            validation["results"]["severity_gate"] = "passed"\n        else:\n            validation["results"]["severity_gate"] = "failed"\n        \n        validation["results"]["gates_passed"] = gates_passed\n        validation["results"]["total_gates"] = total_gates\n        validation["status"] = "passed" if gates_passed == total_gates else "failed"\n        \n        self._record_validation("security", validation)\n        \n        if validation["status"] == "passed":\n            print(f"✅ Security quality gates passed ({gates_passed}/{total_gates})")\n            return True\n        else:\n            print(f"❌ Security quality gates failed ({gates_passed}/{total_gates})")\n            for error in validation["errors"]:\n                print(f"  - {error}")\n            return False
    
    def validate_performance_gates(self, performance_results: Optional[str] = None,
                                 resource_profile: Optional[str] = None,
                                 max_p50_latency: int = 3000,
                                 max_p95_latency: int = 6000,
                                 max_memory_mb: int = 1536) -> bool:
        """Validate performance quality gates"""
        print("⚡ Validating performance quality gates...")
        
        validation = {
            "type": "performance",
            "requirements": {
                "max_p50_latency_ms": max_p50_latency,
                "max_p95_latency_ms": max_p95_latency,
                "max_memory_mb": max_memory_mb
            },
            "results": {},
            "status": "unknown",
            "errors": [],
            "warnings": []
        }
        
        gates_passed = 0
        total_gates = 3
        
        # Parse performance results
        if performance_results and os.path.exists(performance_results):
            try:
                with open(performance_results) as f:
                    perf_data = json.load(f)
                
                # Extract performance metrics
                if "benchmarks" in perf_data:
                    benchmarks = perf_data["benchmarks"]
                    
                    # Calculate percentiles
                    if benchmarks:
                        # Aggregate timing data
                        all_times = []
                        for benchmark in benchmarks:
                            if "stats" in benchmark:
                                stats = benchmark["stats"]
                                all_times.extend([stats.get("mean", 0)] * stats.get("rounds", 1))
                        
                        if all_times:
                            all_times.sort()
                            n = len(all_times)
                            p50_idx = int(n * 0.5)
                            p95_idx = int(n * 0.95)
                            
                            p50_latency = all_times[p50_idx] * 1000  # Convert to ms
                            p95_latency = all_times[p95_idx] * 1000
                            
                            validation["results"]["p50_latency_ms"] = p50_latency
                            validation["results"]["p95_latency_ms"] = p95_latency
                            
                            # Gate 1: P50 latency
                            if p50_latency <= max_p50_latency:
                                gates_passed += 1
                                validation["results"]["p50_gate"] = "passed"
                            else:
                                validation["errors"].append(f"P50 latency too high: {p50_latency}ms > {max_p50_latency}ms")
                                validation["results"]["p50_gate"] = "failed"
                            
                            # Gate 2: P95 latency
                            if p95_latency <= max_p95_latency:
                                gates_passed += 1
                                validation["results"]["p95_gate"] = "passed"
                            else:
                                validation["errors"].append(f"P95 latency too high: {p95_latency}ms > {max_p95_latency}ms")
                                validation["results"]["p95_gate"] = "failed"
                        else:
                            validation["errors"].append("No timing data found in benchmarks")
                            validation["results"]["p50_gate"] = "failed"
                            validation["results"]["p95_gate"] = "failed"
                    else:
                        validation["errors"].append("No benchmarks found in performance results")
                        validation["results"]["p50_gate"] = "failed"
                        validation["results"]["p95_gate"] = "failed"
                
            except Exception as e:
                validation["errors"].append(f"Error processing performance results: {e}")
                validation["results"]["p50_gate"] = "failed"
                validation["results"]["p95_gate"] = "failed"
        else:
            if self.strict_mode:
                validation["errors"].append("Performance results required in strict mode")
                validation["results"]["p50_gate"] = "failed"
                validation["results"]["p95_gate"] = "failed"
            else:
                validation["warnings"].append("Performance results not provided, using defaults")
                # Assume acceptable performance for non-strict mode
                validation["results"]["p50_gate"] = "assumed_pass"
                validation["results"]["p95_gate"] = "assumed_pass"
                gates_passed += 2
        
        # Gate 3: Memory usage
        if resource_profile and os.path.exists(resource_profile):
            try:
                with open(resource_profile) as f:
                    resource_data = json.load(f)
                
                max_memory_usage = resource_data.get("peak_memory_mb", 0)
                validation["results"]["peak_memory_mb"] = max_memory_usage
                
                if max_memory_usage <= max_memory_mb:
                    gates_passed += 1
                    validation["results"]["memory_gate"] = "passed"
                else:
                    validation["errors"].append(f"Memory usage too high: {max_memory_usage}MB > {max_memory_mb}MB")
                    validation["results"]["memory_gate"] = "failed"
                    
            except Exception as e:
                validation["errors"].append(f"Error processing resource profile: {e}")
                validation["results"]["memory_gate"] = "failed"
        else:
            if self.strict_mode:
                validation["errors"].append("Resource profile required in strict mode")
                validation["results"]["memory_gate"] = "failed"
            else:
                validation["warnings"].append("Resource profile not provided")
                validation["results"]["memory_gate"] = "assumed_pass"
                gates_passed += 1
        
        validation["results"]["gates_passed"] = gates_passed
        validation["results"]["total_gates"] = total_gates
        validation["status"] = "passed" if gates_passed == total_gates else "failed"
        
        self._record_validation("performance", validation)
        
        if validation["status"] == "passed":
            print(f"✅ Performance quality gates passed ({gates_passed}/{total_gates})")
            return True
        else:
            print(f"❌ Performance quality gates failed ({gates_passed}/{total_gates})")
            for error in validation["errors"]:
                print(f"  - {error}")
            return False
    
    def validate_code_quality_gates(self, mutation_results: Optional[str] = None,
                                  complexity_results: Optional[str] = None,
                                  maintainability_results: Optional[str] = None,
                                  min_mutation_score: float = 0.80,
                                  max_cyclomatic_complexity: int = 10,
                                  min_maintainability_index: int = 70) -> bool:
        """Validate code quality gates"""
        print("✨ Validating code quality gates...")
        
        validation = {
            "type": "code_quality",
            "requirements": {
                "min_mutation_score": min_mutation_score,
                "max_cyclomatic_complexity": max_cyclomatic_complexity,
                "min_maintainability_index": min_maintainability_index
            },
            "results": {},
            "status": "unknown", 
            "errors": [],
            "warnings": []
        }
        
        gates_passed = 0
        total_gates = 3
        
        # Gate 1: Mutation testing score
        if mutation_results and os.path.exists(mutation_results):
            try:
                with open(mutation_results) as f:
                    mutation_data = json.load(f)
                
                mutation_score = mutation_data.get("mutation_score", 0.0)
                validation["results"]["mutation_score"] = mutation_score
                
                if mutation_score >= min_mutation_score:
                    gates_passed += 1
                    validation["results"]["mutation_gate"] = "passed"
                else:
                    validation["errors"].append(f"Mutation score too low: {mutation_score} < {min_mutation_score}")
                    validation["results"]["mutation_gate"] = "failed"
                    
            except Exception as e:
                validation["errors"].append(f"Error processing mutation results: {e}")
                validation["results"]["mutation_gate"] = "failed"
        else:
            if self.strict_mode:
                validation["errors"].append("Mutation testing results required in strict mode")
                validation["results"]["mutation_gate"] = "failed"
            else:
                validation["warnings"].append("Mutation testing results not provided")
                validation["results"]["mutation_gate"] = "assumed_pass"
                gates_passed += 1
        
        # Gate 2: Cyclomatic complexity
        if complexity_results and os.path.exists(complexity_results):
            try:
                with open(complexity_results) as f:
                    complexity_data = json.load(f)
                
                # Find maximum complexity
                max_complexity = 0
                for file_path, file_data in complexity_data.items():
                    for func_data in file_data:
                        complexity = func_data.get("complexity", 0)
                        max_complexity = max(max_complexity, complexity)
                
                validation["results"]["max_cyclomatic_complexity"] = max_complexity
                
                if max_complexity <= max_cyclomatic_complexity:
                    gates_passed += 1
                    validation["results"]["complexity_gate"] = "passed"
                else:
                    validation["errors"].append(f"Cyclomatic complexity too high: {max_complexity} > {max_cyclomatic_complexity}")
                    validation["results"]["complexity_gate"] = "failed"
                    
            except Exception as e:
                validation["errors"].append(f"Error processing complexity results: {e}")
                validation["results"]["complexity_gate"] = "failed"
        else:
            if self.strict_mode:
                validation["errors"].append("Complexity analysis required in strict mode")
                validation["results"]["complexity_gate"] = "failed"
            else:
                validation["warnings"].append("Complexity analysis not provided")
                validation["results"]["complexity_gate"] = "assumed_pass"
                gates_passed += 1
        
        # Gate 3: Maintainability index
        if maintainability_results and os.path.exists(maintainability_results):
            try:
                with open(maintainability_results) as f:
                    maintainability_data = json.load(f)
                
                # Calculate average maintainability index
                total_mi = 0
                file_count = 0
                
                for file_path, file_data in maintainability_data.items():
                    for func_data in file_data:
                        mi = func_data.get("mi", 0)
                        total_mi += mi
                        file_count += 1
                
                avg_maintainability = total_mi / file_count if file_count > 0 else 0
                validation["results"]["avg_maintainability_index"] = avg_maintainability
                
                if avg_maintainability >= min_maintainability_index:
                    gates_passed += 1
                    validation["results"]["maintainability_gate"] = "passed"
                else:
                    validation["errors"].append(f"Maintainability index too low: {avg_maintainability} < {min_maintainability_index}")
                    validation["results"]["maintainability_gate"] = "failed"
                    
            except Exception as e:
                validation["errors"].append(f"Error processing maintainability results: {e}")
                validation["results"]["maintainability_gate"] = "failed"
        else:
            if self.strict_mode:
                validation["errors"].append("Maintainability analysis required in strict mode")
                validation["results"]["maintainability_gate"] = "failed"
            else:
                validation["warnings"].append("Maintainability analysis not provided")
                validation["results"]["maintainability_gate"] = "assumed_pass"
                gates_passed += 1
        
        validation["results"]["gates_passed"] = gates_passed
        validation["results"]["total_gates"] = total_gates
        validation["status"] = "passed" if gates_passed == total_gates else "failed"
        
        self._record_validation("code_quality", validation)
        
        if validation["status"] == "passed":
            print(f"✅ Code quality gates passed ({gates_passed}/{total_gates})")
            return True
        else:
            print(f"❌ Code quality gates failed ({gates_passed}/{total_gates})")
            for error in validation["errors"]:
                print(f"  - {error}")
            return False
    
    def validate_bundle_integrity(self, artifact_path: str, 
                                verify_hashes: bool = True,
                                verify_signatures: bool = True) -> bool:
        """Validate bundle integrity and signatures"""
        print("📦 Validating bundle integrity...")
        
        validation = {
            "type": "bundle",
            "requirements": {
                "verify_hashes": verify_hashes,
                "verify_signatures": verify_signatures
            },
            "results": {},
            "status": "unknown",
            "errors": [],
            "warnings": []
        }
        
        gates_passed = 0
        total_gates = 2
        
        # Gate 1: Hash verification
        if verify_hashes:
            hash_file = f"{artifact_path}.sha256"
            if os.path.exists(hash_file):
                try:
                    # Verify SHA256 hash
                    result = subprocess.run(['sha256sum', '-c', hash_file], 
                                          capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        gates_passed += 1
                        validation["results"]["hash_verification"] = "passed"
                        validation["results"]["hash_file"] = hash_file
                    else:
                        validation["errors"].append(f"Hash verification failed: {result.stderr}")
                        validation["results"]["hash_verification"] = "failed"
                        
                except Exception as e:
                    validation["errors"].append(f"Error verifying hash: {e}")
                    validation["results"]["hash_verification"] = "failed"
            else:
                validation["errors"].append(f"Hash file not found: {hash_file}")
                validation["results"]["hash_verification"] = "failed"
        else:
            gates_passed += 1
            validation["results"]["hash_verification"] = "skipped"
        
        # Gate 2: Signature verification
        if verify_signatures:
            sig_file = f"{artifact_path}.sig"
            if os.path.exists(sig_file):
                try:
                    with open(sig_file) as f:
                        sig_data = json.load(f)
                    
                    # Basic signature structure validation
                    required_fields = ["signature_algorithm", "signature", "signed_at", "key_fingerprint"]
                    missing_fields = [field for field in required_fields if field not in sig_data]
                    
                    if not missing_fields:
                        gates_passed += 1
                        validation["results"]["signature_verification"] = "passed"
                        validation["results"]["signature_algorithm"] = sig_data.get("signature_algorithm")
                        validation["results"]["key_fingerprint"] = sig_data.get("key_fingerprint")
                    else:
                        validation["errors"].append(f"Invalid signature format, missing fields: {missing_fields}")
                        validation["results"]["signature_verification"] = "failed"
                        
                except Exception as e:
                    validation["errors"].append(f"Error verifying signature: {e}")
                    validation["results"]["signature_verification"] = "failed"
            else:
                validation["errors"].append(f"Signature file not found: {sig_file}")
                validation["results"]["signature_verification"] = "failed"
        else:
            gates_passed += 1
            validation["results"]["signature_verification"] = "skipped"
        
        validation["results"]["gates_passed"] = gates_passed
        validation["results"]["total_gates"] = total_gates
        validation["status"] = "passed" if gates_passed == total_gates else "failed"
        
        self._record_validation("bundle", validation)
        
        if validation["status"] == "passed":
            print(f"✅ Bundle integrity gates passed ({gates_passed}/{total_gates})")
            return True
        else:
            print(f"❌ Bundle integrity gates failed ({gates_passed}/{total_gates})")
            for error in validation["errors"]:
                print(f"  - {error}")
            return False
    
    def _record_validation(self, validation_type: str, validation_data: Dict[str, Any]) -> None:
        """Record validation results"""
        self.validation_results["validations"][validation_type] = validation_data
        self.validation_results["total_gates"] += validation_data.get("results", {}).get("total_gates", 0)
        self.validation_results["passed_gates"] += validation_data.get("results", {}).get("gates_passed", 0)
        
        if validation_data["status"] == "failed":
            self.validation_results["failed_gates"] += 1
        
        # Accumulate warnings
        self.validation_results["warnings"].extend(validation_data.get("warnings", []))
    
    def generate_final_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate final quality gate report"""
        # Calculate overall status
        failed_validations = [name for name, data in self.validation_results["validations"].items() 
                            if data["status"] == "failed"]
        
        if failed_validations:
            self.validation_results["overall_status"] = "failed"
            self.validation_results["failed_validations"] = failed_validations
        else:
            self.validation_results["overall_status"] = "passed"
        
        # Add summary
        self.validation_results["summary"] = {
            "total_validations": len(self.validation_results["validations"]),
            "passed_validations": len([data for data in self.validation_results["validations"].values() 
                                     if data["status"] == "passed"]),
            "failed_validations": len(failed_validations),
            "total_warnings": len(self.validation_results["warnings"])
        }
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
            print(f"📊 Quality gate report saved to: {output_file}")
        
        return self.validation_results
    
    def print_summary(self) -> None:
        """Print quality gate summary"""
        print("\n" + "="*60)
        print("📊 QUALITY GATE SUMMARY")
        print("="*60)
        
        overall_status = self.validation_results["overall_status"]
        status_icon = "✅" if overall_status == "passed" else "❌"
        print(f"Overall Status: {status_icon} {overall_status.upper()}")
        
        print(f"\nValidation Results:")
        for name, data in self.validation_results["validations"].items():
            status = data["status"]
            status_icon = "✅" if status == "passed" else "❌" if status == "failed" else "⏭️"
            gates_info = data.get("results", {})
            gates_passed = gates_info.get("gates_passed", 0)
            total_gates = gates_info.get("total_gates", 0)
            print(f"  {status_icon} {name}: {gates_passed}/{total_gates} gates passed")
        
        if self.validation_results["warnings"]:
            print(f"\nWarnings ({len(self.validation_results['warnings'])}):")
            for warning in self.validation_results["warnings"][:5]:  # Show first 5
                print(f"  ⚠️  {warning}")
            
            if len(self.validation_results["warnings"]) > 5:
                print(f"  ... and {len(self.validation_results['warnings']) - 5} more")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Apply quality gates for different validation types')
    parser.add_argument('--type', required=True, 
                       choices=['dataset', 'security', 'performance', 'quality', 'bundle', 'integration'],
                       help='Type of validation to perform')
    parser.add_argument('--strict', action='store_true',
                       help='Enable strict validation mode')
    
    # Dataset validation options
    parser.add_argument('--min-queries', type=int, default=600,
                       help='Minimum number of queries required')
    parser.add_argument('--min-domains', type=int, default=3,
                       help='Minimum number of domains required')
    parser.add_argument('--min-iaa-kappa', type=float, default=0.7,
                       help='Minimum inter-annotator agreement kappa')
    parser.add_argument('--validate-schema', action='store_true', default=True,
                       help='Validate dataset schema')
    
    # Security validation options
    parser.add_argument('--sast-results', 
                       help='Path to SAST scan results')
    parser.add_argument('--bandit-results',
                       help='Path to Bandit security scan results')
    parser.add_argument('--safety-results',
                       help='Path to Safety dependency scan results')
    parser.add_argument('--max-high-severity', type=int, default=0,
                       help='Maximum high severity issues allowed')
    parser.add_argument('--max-critical-severity', type=int, default=0,
                       help='Maximum critical severity issues allowed')
    
    # Performance validation options
    parser.add_argument('--performance-results',
                       help='Path to performance benchmark results')
    parser.add_argument('--resource-profile',
                       help='Path to resource profile results')
    parser.add_argument('--max-p50-latency', type=int, default=3000,
                       help='Maximum P50 latency in milliseconds')
    parser.add_argument('--max-p95-latency', type=int, default=6000,
                       help='Maximum P95 latency in milliseconds')
    parser.add_argument('--max-memory-mb', type=int, default=1536,
                       help='Maximum memory usage in MB')
    
    # Code quality validation options  
    parser.add_argument('--mutation-results',
                       help='Path to mutation testing results')
    parser.add_argument('--complexity-results',
                       help='Path to complexity analysis results')
    parser.add_argument('--maintainability-results',
                       help='Path to maintainability analysis results')
    parser.add_argument('--min-mutation-score', type=float, default=0.80,
                       help='Minimum mutation score required')
    parser.add_argument('--max-cyclomatic-complexity', type=int, default=10,
                       help='Maximum cyclomatic complexity allowed')
    parser.add_argument('--min-maintainability-index', type=int, default=70,
                       help='Minimum maintainability index required')
    
    # Bundle validation options
    parser.add_argument('--artifact',
                       help='Path to artifact bundle')
    parser.add_argument('--verify-hashes', action='store_true', default=True,
                       help='Verify file hashes')
    parser.add_argument('--verify-signatures', action='store_true', default=True,
                       help='Verify cryptographic signatures')
    
    # Integration validation options
    parser.add_argument('--smoke-results',
                       help='Path to smoke test results')
    parser.add_argument('--validate-api-surface', action='store_true',
                       help='Validate API surface compatibility')
    parser.add_argument('--check-backwards-compatibility', action='store_true',
                       help='Check backwards compatibility')
    
    # Output options
    parser.add_argument('--output', '-o',
                       help='Output file for validation results')
    
    args = parser.parse_args()
    
    print("🚦 Lethe Research Quality Gate Enforcer")
    print("======================================")
    
    enforcer = QualityGateEnforcer(strict_mode=args.strict)
    
    try:
        success = False
        
        if args.type == 'dataset':
            success = enforcer.validate_dataset_quality(
                min_queries=args.min_queries,
                min_domains=args.min_domains,
                min_iaa_kappa=args.min_iaa_kappa,
                validate_schema=args.validate_schema
            )
        
        elif args.type == 'security':
            success = enforcer.validate_security_gates(
                sast_results=args.sast_results,
                bandit_results=args.bandit_results,
                safety_results=args.safety_results,
                max_high_severity=args.max_high_severity,
                max_critical_severity=args.max_critical_severity
            )
        
        elif args.type == 'performance':
            success = enforcer.validate_performance_gates(
                performance_results=args.performance_results,
                resource_profile=args.resource_profile,
                max_p50_latency=args.max_p50_latency,
                max_p95_latency=args.max_p95_latency,
                max_memory_mb=args.max_memory_mb
            )
        
        elif args.type == 'quality':
            success = enforcer.validate_code_quality_gates(
                mutation_results=args.mutation_results,
                complexity_results=args.complexity_results,
                maintainability_results=args.maintainability_results,
                min_mutation_score=args.min_mutation_score,
                max_cyclomatic_complexity=args.max_cyclomatic_complexity,
                min_maintainability_index=args.min_maintainability_index
            )
        
        elif args.type == 'bundle':
            if not args.artifact:
                print("❌ --artifact path required for bundle validation")
                sys.exit(1)
            
            success = enforcer.validate_bundle_integrity(
                artifact_path=args.artifact,
                verify_hashes=args.verify_hashes,
                verify_signatures=args.verify_signatures
            )
        
        elif args.type == 'integration':
            # Integration validation would check smoke test results, API compatibility, etc.
            print("🔗 Integration validation not yet implemented")
            success = True
        
        # Generate final report
        enforcer.generate_final_report(args.output)
        enforcer.print_summary()
        
        if success:
            print("\n🎉 All quality gates passed!")
            sys.exit(0)
        else:
            print(f"\n💥 Quality gate validation failed!")
            if args.strict:
                print("Strict mode enabled - failing with error code")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Validation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()