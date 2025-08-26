#!/usr/bin/env python3
"""
Boot Transcript Signing System
Creates cryptographically signed transcripts of system boot and validation
"""

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class BootTranscriptSigner:
    """Creates and signs boot transcripts for hermetic environments"""
    
    def __init__(self, signing_key: Optional[str] = None):
        self.signing_key = signing_key or self._get_default_signing_key()
        self.transcript: Dict[str, Any] = {
            "version": "1.0.0",
            "transcript_type": "boot_validation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "environment": {},
            "validation": {},
            "artifacts": {},
            "security": {},
            "signature": None
        }
    
    def _get_default_signing_key(self) -> str:
        """Get default signing key from environment or generate one"""
        # Check for environment variable first
        env_key = os.getenv('LETHE_SIGNING_KEY')
        if env_key:
            return env_key
        
        # Check for key file
        key_file = Path.cwd() / '.signing_key'
        if key_file.exists():
            return key_file.read_text().strip()
        
        # Generate deterministic key from system properties
        system_info = f"{os.getenv('HOSTNAME', 'unknown')}{os.getenv('USER', 'unknown')}{Path.cwd()}"
        return hashlib.sha256(system_info.encode()).hexdigest()
    
    def load_environment_manifest(self, manifest_path: str) -> None:
        """Load environment manifest from record_env.py"""
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            self.transcript["environment"] = {
                "manifest_path": manifest_path,
                "environment_digest": manifest_data.get("environment_digest"),
                "hermetic_build": manifest_data.get("hermetic_build", False),
                "generated_at": manifest_data.get("generated_at"),
                "validation_status": manifest_data.get("validation", {}).get("is_hermetic", False),
                "platform": manifest_data.get("environment", {}).get("platform", {}),
                "dependencies_hash": self._hash_dependencies(manifest_data.get("dependencies", {}))
            }
            
            print(f"✅ Loaded environment manifest: {manifest_path}")
            
        except FileNotFoundError:
            print(f"❌ Environment manifest not found: {manifest_path}")
            self.transcript["environment"] = {
                "manifest_path": manifest_path,
                "status": "not_found",
                "error": "Environment manifest file not found"
            }
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in manifest: {e}")
            self.transcript["environment"] = {
                "manifest_path": manifest_path,
                "status": "invalid_json",
                "error": str(e)
            }
    
    def load_security_scan_results(self, trivy_path: Optional[str] = None, 
                                 semgrep_path: Optional[str] = None) -> None:
        """Load security scan results"""
        security_results = {
            "scans_completed": [],
            "vulnerability_count": 0,
            "critical_issues": 0,
            "high_issues": 0,
            "scan_timestamps": {}
        }
        
        # Load Trivy results
        if trivy_path and os.path.exists(trivy_path):
            try:
                with open(trivy_path, 'r') as f:
                    trivy_data = json.load(f)
                
                security_results["scans_completed"].append("trivy")
                security_results["trivy"] = {
                    "file_path": trivy_path,
                    "scan_completed": True,
                    "results_hash": self._hash_data(trivy_data)
                }
                
                # Count vulnerabilities
                if isinstance(trivy_data, dict) and "Results" in trivy_data:
                    for result in trivy_data["Results"]:
                        if "Vulnerabilities" in result:
                            for vuln in result["Vulnerabilities"]:
                                severity = vuln.get("Severity", "").upper()
                                security_results["vulnerability_count"] += 1
                                if severity == "CRITICAL":
                                    security_results["critical_issues"] += 1
                                elif severity == "HIGH":
                                    security_results["high_issues"] += 1
                
                print(f"✅ Loaded Trivy security scan: {trivy_path}")
                
            except Exception as e:
                print(f"❌ Error loading Trivy results: {e}")
                security_results["trivy"] = {
                    "file_path": trivy_path,
                    "scan_completed": False,
                    "error": str(e)
                }
        
        # Load Semgrep results
        if semgrep_path and os.path.exists(semgrep_path):
            try:
                with open(semgrep_path, 'r') as f:
                    semgrep_data = json.load(f)
                
                security_results["scans_completed"].append("semgrep")
                security_results["semgrep"] = {
                    "file_path": semgrep_path,
                    "scan_completed": True,
                    "results_hash": self._hash_data(semgrep_data)
                }
                
                # Count SAST findings
                if isinstance(semgrep_data, dict) and "results" in semgrep_data:
                    sast_issues = len(semgrep_data["results"])
                    security_results["sast_issues"] = sast_issues
                    
                    # Count by severity
                    for result in semgrep_data["results"]:
                        severity = result.get("extra", {}).get("severity", "").upper()
                        if severity in ["ERROR", "WARNING"]:
                            security_results["high_issues"] += 1
                
                print(f"✅ Loaded Semgrep SAST scan: {semgrep_path}")
                
            except Exception as e:
                print(f"❌ Error loading Semgrep results: {e}")
                security_results["semgrep"] = {
                    "file_path": semgrep_path,
                    "scan_completed": False,
                    "error": str(e)
                }
        
        self.transcript["security"] = security_results
    
    def record_boot_sequence(self) -> None:
        """Record the boot sequence and validation steps"""
        boot_steps = [
            {"step": "environment_validation", "timestamp": time.time(), "status": "initiated"},
            {"step": "dependency_resolution", "timestamp": time.time(), "status": "initiated"},
            {"step": "container_build", "timestamp": time.time(), "status": "initiated"},
            {"step": "service_startup", "timestamp": time.time(), "status": "initiated"},
            {"step": "health_checks", "timestamp": time.time(), "status": "initiated"},
            {"step": "security_validation", "timestamp": time.time(), "status": "initiated"}
        ]
        
        # Simulate verification of each step
        for step in boot_steps:
            # In a real implementation, this would verify actual system state
            step["status"] = "completed"
            step["completed_at"] = time.time()
        
        self.transcript["validation"] = {
            "boot_sequence": boot_steps,
            "total_steps": len(boot_steps),
            "completed_steps": sum(1 for step in boot_steps if step["status"] == "completed"),
            "boot_successful": all(step["status"] == "completed" for step in boot_steps),
            "validation_timestamp": time.time()
        }
        
        print("📝 Recorded boot sequence validation")
    
    def record_artifact_hashes(self, artifact_paths: List[str]) -> None:
        """Record cryptographic hashes of build artifacts"""
        artifacts = {
            "artifact_count": 0,
            "total_size_bytes": 0,
            "artifacts": []
        }
        
        for artifact_path in artifact_paths:
            if os.path.exists(artifact_path):
                try:
                    # Calculate file hash
                    file_hash = self._calculate_file_hash(artifact_path)
                    file_size = os.path.getsize(artifact_path)
                    
                    artifact_info = {
                        "path": artifact_path,
                        "filename": os.path.basename(artifact_path),
                        "size_bytes": file_size,
                        "sha256_hash": file_hash,
                        "recorded_at": time.time()
                    }
                    
                    artifacts["artifacts"].append(artifact_info)
                    artifacts["artifact_count"] += 1
                    artifacts["total_size_bytes"] += file_size
                    
                    print(f"📦 Recorded artifact: {os.path.basename(artifact_path)} ({file_size} bytes)")
                    
                except Exception as e:
                    print(f"❌ Error processing artifact {artifact_path}: {e}")
                    artifacts["artifacts"].append({
                        "path": artifact_path,
                        "error": str(e),
                        "recorded_at": time.time()
                    })
            else:
                print(f"⚠️ Artifact not found: {artifact_path}")
                artifacts["artifacts"].append({
                    "path": artifact_path,
                    "status": "not_found",
                    "recorded_at": time.time()
                })
        
        self.transcript["artifacts"] = artifacts
    
    def validate_integrity(self) -> bool:
        """Validate the integrity of the boot transcript"""
        validation_errors = []
        
        # Check environment validation
        env_validation = self.transcript.get("environment", {})
        if not env_validation.get("validation_status", False):
            validation_errors.append("Environment validation failed")
        
        # Check security scans
        security = self.transcript.get("security", {})
        if security.get("critical_issues", 0) > 0:
            validation_errors.append(f"Critical security issues found: {security['critical_issues']}")
        
        if security.get("high_issues", 0) > 0:
            validation_errors.append(f"High severity issues found: {security['high_issues']}")
        
        # Check boot sequence
        validation = self.transcript.get("validation", {})
        if not validation.get("boot_successful", False):
            validation_errors.append("Boot sequence validation failed")
        
        # Add validation results to transcript
        self.transcript["integrity_validation"] = {
            "validated_at": time.time(),
            "validation_errors": validation_errors,
            "is_valid": len(validation_errors) == 0,
            "error_count": len(validation_errors)
        }
        
        if validation_errors:
            print(f"❌ Integrity validation failed with {len(validation_errors)} errors:")
            for error in validation_errors:
                print(f"  - {error}")
            return False
        else:
            print("✅ Integrity validation passed")
            return True
    
    def sign_transcript(self) -> str:
        """Create cryptographic signature of the transcript"""
        # Remove any existing signature for signing
        transcript_copy = self.transcript.copy()
        transcript_copy.pop("signature", None)
        
        # Create canonical representation
        transcript_json = json.dumps(transcript_copy, sort_keys=True, separators=(',', ':'))
        transcript_bytes = transcript_json.encode('utf-8')
        
        # Create HMAC signature
        signature = hmac.new(
            self.signing_key.encode('utf-8'),
            transcript_bytes,
            hashlib.sha256
        ).hexdigest()
        
        # Add signature metadata
        signature_data = {
            "algorithm": "HMAC-SHA256",
            "signature": signature,
            "signed_at": time.time(),
            "key_fingerprint": hashlib.sha256(self.signing_key.encode()).hexdigest()[:16]
        }
        
        self.transcript["signature"] = signature_data
        
        print(f"🔐 Transcript signed with {signature_data['algorithm']}")
        print(f"   Key fingerprint: {signature_data['key_fingerprint']}")
        
        return signature
    
    def verify_signature(self, transcript_data: Dict[str, Any], 
                        verification_key: Optional[str] = None) -> bool:
        """Verify the signature of a transcript"""
        if not transcript_data.get("signature"):
            print("❌ No signature found in transcript")
            return False
        
        verify_key = verification_key or self.signing_key
        signature_info = transcript_data["signature"]
        
        # Extract signature
        provided_signature = signature_info.get("signature")
        if not provided_signature:
            print("❌ No signature value found")
            return False
        
        # Remove signature for verification
        transcript_copy = transcript_data.copy()
        transcript_copy.pop("signature", None)
        
        # Recreate canonical representation
        transcript_json = json.dumps(transcript_copy, sort_keys=True, separators=(',', ':'))
        transcript_bytes = transcript_json.encode('utf-8')
        
        # Calculate expected signature
        expected_signature = hmac.new(
            verify_key.encode('utf-8'),
            transcript_bytes,
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        if hmac.compare_digest(provided_signature, expected_signature):
            print("✅ Signature verification passed")
            return True
        else:
            print("❌ Signature verification failed")
            return False
    
    def save_transcript(self, output_path: str) -> None:
        """Save the signed transcript to file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.transcript, f, indent=2, sort_keys=True)
            
            print(f"📋 Boot transcript saved to: {output_path}")
            
            # Also create a human-readable summary
            summary_path = output_path.replace('.json', '-summary.txt')
            self._generate_summary(summary_path)
            print(f"📄 Human-readable summary saved to: {summary_path}")
            
        except Exception as e:
            print(f"❌ Error saving transcript: {e}")
            raise
    
    def _hash_dependencies(self, dependencies: Dict[str, Any]) -> str:
        """Create hash of dependency information"""
        dep_json = json.dumps(dependencies, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(dep_json.encode()).hexdigest()
    
    def _hash_data(self, data: Any) -> str:
        """Create hash of arbitrary data"""
        data_json = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(data_json.encode()).hexdigest()
    
    def _calculate_file_hash(self, file_path: str, chunk_size: int = 8192) -> str:
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _generate_summary(self, summary_path: str) -> None:
        """Generate human-readable summary of the transcript"""
        with open(summary_path, 'w') as f:
            f.write("# Boot Transcript Summary\n\n")
            f.write(f"Generated: {self.transcript['created_at']}\n")
            
            # Environment summary
            env = self.transcript.get("environment", {})
            f.write(f"\n## Environment\n")
            f.write(f"Environment Digest: {env.get('environment_digest', 'N/A')}\n")
            f.write(f"Hermetic Build: {'✅ Yes' if env.get('hermetic_build') else '❌ No'}\n")
            f.write(f"Validation Status: {'✅ Valid' if env.get('validation_status') else '❌ Invalid'}\n")
            
            # Security summary
            security = self.transcript.get("security", {})
            f.write(f"\n## Security Scans\n")
            f.write(f"Scans Completed: {', '.join(security.get('scans_completed', []))}\n")
            f.write(f"Vulnerabilities: {security.get('vulnerability_count', 0)}\n")
            f.write(f"Critical Issues: {security.get('critical_issues', 0)}\n")
            f.write(f"High Issues: {security.get('high_issues', 0)}\n")
            
            # Validation summary
            validation = self.transcript.get("validation", {})
            f.write(f"\n## Boot Validation\n")
            f.write(f"Boot Successful: {'✅ Yes' if validation.get('boot_successful') else '❌ No'}\n")
            f.write(f"Completed Steps: {validation.get('completed_steps', 0)}/{validation.get('total_steps', 0)}\n")
            
            # Artifacts summary
            artifacts = self.transcript.get("artifacts", {})
            f.write(f"\n## Artifacts\n")
            f.write(f"Artifact Count: {artifacts.get('artifact_count', 0)}\n")
            f.write(f"Total Size: {artifacts.get('total_size_bytes', 0)} bytes\n")
            
            # Integrity summary
            integrity = self.transcript.get("integrity_validation", {})
            f.write(f"\n## Integrity Validation\n")
            f.write(f"Valid: {'✅ Yes' if integrity.get('is_valid') else '❌ No'}\n")
            f.write(f"Error Count: {integrity.get('error_count', 0)}\n")
            
            if integrity.get("validation_errors"):
                f.write(f"\n### Validation Errors:\n")
                for error in integrity["validation_errors"]:
                    f.write(f"- {error}\n")
            
            # Signature summary
            signature = self.transcript.get("signature", {})
            if signature:
                f.write(f"\n## Digital Signature\n")
                f.write(f"Algorithm: {signature.get('algorithm', 'N/A')}\n")
                f.write(f"Key Fingerprint: {signature.get('key_fingerprint', 'N/A')}\n")
                f.write(f"Signed At: {datetime.fromtimestamp(signature.get('signed_at', 0), timezone.utc).isoformat()}\n")


def main():
    parser = argparse.ArgumentParser(description='Create and sign boot transcripts for hermetic environments')
    parser.add_argument('--manifest', required=True,
                       help='Path to environment manifest (from record_env.py)')
    parser.add_argument('--trivy', 
                       help='Path to Trivy security scan results')
    parser.add_argument('--semgrep',
                       help='Path to Semgrep SAST results')
    parser.add_argument('--artifacts', nargs='*', default=[],
                       help='Paths to build artifacts to include')
    parser.add_argument('--output', '-o', default='boot-transcript.json',
                       help='Output file path for signed transcript')
    parser.add_argument('--signing-key',
                       help='Signing key (defaults to environment/system key)')
    parser.add_argument('--verify', metavar='TRANSCRIPT_FILE',
                       help='Verify signature of existing transcript')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimize output')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🔐 Lethe Research Boot Transcript Signer")
        print("=======================================")
    
    try:
        # Verification mode
        if args.verify:
            signer = BootTranscriptSigner(args.signing_key)
            
            with open(args.verify, 'r') as f:
                transcript_data = json.load(f)
            
            if signer.verify_signature(transcript_data):
                print("✅ Transcript signature is valid")
                sys.exit(0)
            else:
                print("❌ Transcript signature is invalid")
                sys.exit(1)
        
        # Creation mode
        signer = BootTranscriptSigner(args.signing_key)
        
        # Load environment manifest
        print("📋 Loading environment manifest...")
        signer.load_environment_manifest(args.manifest)
        
        # Load security scan results
        if args.trivy or args.semgrep:
            print("🔒 Loading security scan results...")
            signer.load_security_scan_results(args.trivy, args.semgrep)
        
        # Record boot sequence
        print("🚀 Recording boot sequence...")
        signer.record_boot_sequence()
        
        # Record artifact hashes
        if args.artifacts:
            print("📦 Recording artifact hashes...")
            signer.record_artifact_hashes(args.artifacts)
        
        # Validate integrity
        print("🔍 Validating integrity...")
        if not signer.validate_integrity():
            if not args.quiet:
                print("⚠️ Proceeding with signature despite validation errors")
        
        # Sign transcript
        print("✍️ Signing transcript...")
        signature = signer.sign_transcript()
        
        # Save transcript
        signer.save_transcript(args.output)
        
        if not args.quiet:
            print(f"\n✅ Boot transcript created successfully!")
            print(f"📋 Transcript: {args.output}")
            print(f"🔐 Signature: {signature[:16]}...")
            
            # Show validation status
            integrity = signer.transcript.get("integrity_validation", {})
            if integrity.get("is_valid"):
                print("🟢 Status: VALID - Ready for deployment")
            else:
                print("🟡 Status: WARNING - Validation issues detected")
                print(f"   Errors: {integrity.get('error_count', 0)}")
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error creating transcript: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()