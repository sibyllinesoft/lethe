#!/usr/bin/env python3
"""
NeurIPS 2025 Submission Validation Script
Validates that all submission requirements are met before upload.
"""

import os
import subprocess
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

class NeurIPSSubmissionValidator:
    def __init__(self, submission_dir: str):
        self.submission_dir = Path(submission_dir)
        self.results = []
        
    def validate_submission(self) -> bool:
        """Run complete submission validation"""
        print("🔍 NeurIPS 2025 Submission Validation")
        print("=" * 50)
        
        validation_checks = [
            ("File Structure", self.validate_file_structure),
            ("LaTeX Compilation", self.validate_latex_compilation),
            ("Page Limits", self.validate_page_limits),
            ("Anonymous Submission", self.validate_anonymity),
            ("Figure Quality", self.validate_figures),
            ("Bibliography", self.validate_bibliography),
            ("Supplementary Materials", self.validate_supplementary),
            ("Reproducibility", self.validate_reproducibility),
            ("Ethics Statement", self.validate_ethics),
            ("File Sizes", self.validate_file_sizes)
        ]
        
        all_passed = True
        
        for check_name, check_function in validation_checks:
            print(f"\n📋 {check_name}")
            try:
                passed, details = check_function()
                status = "✅ PASSED" if passed else "❌ FAILED"
                print(f"   {status}")
                
                if details:
                    for detail in details:
                        print(f"   • {detail}")
                
                self.results.append({
                    "check": check_name,
                    "passed": passed,
                    "details": details
                })
                
                if not passed:
                    all_passed = False
                    
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                all_passed = False
                self.results.append({
                    "check": check_name,
                    "passed": False,
                    "details": [f"Validation error: {e}"]
                })
        
        # Generate summary
        self.generate_validation_report(all_passed)
        
        print(f"\n🏆 OVERALL VALIDATION: {'✅ READY FOR SUBMISSION' if all_passed else '❌ REQUIRES FIXES'}")
        return all_passed
    
    def validate_file_structure(self) -> Tuple[bool, List[str]]:
        """Validate required files are present"""
        required_files = [
            "neurips_2025_lethe_submission.tex",
            "neurips_2025.sty", 
            "references.bib",
            "SUPPLEMENTARY_MATERIALS.md",
            "NEURIPS_2025_SUBMISSION_README.md",
            "NEURIPS_2025_REPRODUCIBILITY_CHECKLIST.md",
            "NEURIPS_2025_ETHICS_STATEMENT.md"
        ]
        
        missing_files = []
        details = []
        
        for file in required_files:
            filepath = self.submission_dir / file
            if not filepath.exists():
                missing_files.append(file)
                details.append(f"Missing: {file}")
            else:
                details.append(f"Found: {file}")
        
        passed = len(missing_files) == 0
        return passed, details
    
    def validate_latex_compilation(self) -> Tuple[bool, List[str]]:
        """Validate LaTeX compilation"""
        details = []
        main_tex = self.submission_dir / "neurips_2025_lethe_submission.tex"
        
        if not main_tex.exists():
            return False, ["Main LaTeX file not found"]
        
        try:
            # Test compilation
            result = subprocess.run([
                "pdflatex", "-interaction=nonstopmode",
                str(main_tex)
            ], cwd=self.submission_dir, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                details.append("LaTeX compilation successful")
                
                # Check for warnings
                if "Warning" in result.stdout:
                    warning_count = result.stdout.count("Warning")
                    details.append(f"{warning_count} warnings found")
                
                return True, details
            else:
                details.append("LaTeX compilation failed")
                details.append(f"Error: {result.stderr[:200]}...")
                return False, details
                
        except subprocess.TimeoutExpired:
            return False, ["LaTeX compilation timeout"]
        except Exception as e:
            return False, [f"Compilation error: {e}"]
    
    def validate_page_limits(self) -> Tuple[bool, List[str]]:
        """Validate page count limits"""
        pdf_file = self.submission_dir / "neurips_2025_lethe_submission.pdf"
        details = []
        
        if not pdf_file.exists():
            return False, ["PDF file not found - run LaTeX compilation first"]
        
        try:
            # Get page count using pdfinfo
            result = subprocess.run([
                "pdfinfo", str(pdf_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Extract page count
                match = re.search(r'Pages:\s+(\d+)', result.stdout)
                if match:
                    page_count = int(match.group(1))
                    details.append(f"Total pages: {page_count}")
                    
                    # NeurIPS allows 8 pages main content + unlimited references/appendix
                    if page_count <= 10:  # Reasonable upper limit
                        details.append("Page count within limits")
                        return True, details
                    else:
                        details.append("Page count exceeds reasonable limits")
                        return False, details
                else:
                    return False, ["Could not determine page count"]
            else:
                return False, ["pdfinfo command failed"]
                
        except Exception as e:
            return False, [f"Page count validation error: {e}"]
    
    def validate_anonymity(self) -> Tuple[bool, List[str]]:
        """Validate anonymous submission requirements"""
        main_tex = self.submission_dir / "neurips_2025_lethe_submission.tex"
        details = []
        
        if not main_tex.exists():
            return False, ["Main LaTeX file not found"]
        
        with open(main_tex, 'r') as f:
            content = f.read()
        
        # Check for author information
        anonymous_indicators = [
            "Anonymous",
            "anonymous"
        ]
        
        non_anonymous_patterns = [
            r'\\author\{[^}]*[A-Z][a-z]+ [A-Z][a-z]+[^}]*\}',  # Real names
            r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses
            r'\\affiliation\{[^}]*university[^}]*\}',  # Affiliations
        ]
        
        has_anonymous = any(indicator in content for indicator in anonymous_indicators)
        has_real_info = any(re.search(pattern, content, re.IGNORECASE) 
                          for pattern in non_anonymous_patterns)
        
        if has_anonymous and not has_real_info:
            details.append("Proper anonymous submission format")
            return True, details
        elif has_real_info:
            details.append("Real author information detected - must be anonymized")
            return False, details
        else:
            details.append("No clear anonymization found - please verify")
            return False, details
    
    def validate_figures(self) -> Tuple[bool, List[str]]:
        """Validate figure quality and references"""
        details = []
        figures_dir = self.submission_dir / "figures"
        
        if not figures_dir.exists():
            details.append("No figures directory found")
            return True, details  # Figures are optional
        
        figure_files = list(figures_dir.glob("*.pdf")) + list(figures_dir.glob("*.png"))
        
        if not figure_files:
            details.append("No figure files found")
            return True, details
        
        details.append(f"Found {len(figure_files)} figure files")
        
        # Check file sizes (should be reasonable for academic submission)
        large_files = []
        for fig_file in figure_files:
            size_mb = fig_file.stat().st_size / (1024 * 1024)
            if size_mb > 5:  # More than 5MB is unusual for academic figures
                large_files.append(f"{fig_file.name}: {size_mb:.1f}MB")
        
        if large_files:
            details.extend([f"Large figure: {f}" for f in large_files])
            details.append("Consider compressing large figures")
        
        return True, details
    
    def validate_bibliography(self) -> Tuple[bool, List[str]]:
        """Validate bibliography completeness"""
        bib_file = self.submission_dir / "references.bib"
        main_tex = self.submission_dir / "neurips_2025_lethe_submission.tex"
        details = []
        
        if not bib_file.exists():
            return False, ["Bibliography file not found"]
        
        if not main_tex.exists():
            return False, ["Main LaTeX file not found"]
        
        # Count bibliography entries
        with open(bib_file, 'r') as f:
            bib_content = f.read()
        
        entry_count = len(re.findall(r'@\w+\{', bib_content))
        details.append(f"Bibliography entries: {entry_count}")
        
        # Check for citations in main text
        with open(main_tex, 'r') as f:
            tex_content = f.read()
        
        citation_count = len(re.findall(r'\\cite\{[^}]+\}', tex_content))
        details.append(f"Citations in text: {citation_count}")
        
        if entry_count >= 10 and citation_count >= 5:
            details.append("Adequate bibliography and citations")
            return True, details
        else:
            details.append("Consider adding more references and citations")
            return True, details  # Warning, not failure
    
    def validate_supplementary(self) -> Tuple[bool, List[str]]:
        """Validate supplementary materials"""
        supp_file = self.submission_dir / "SUPPLEMENTARY_MATERIALS.md"
        details = []
        
        if not supp_file.exists():
            return False, ["Supplementary materials file not found"]
        
        with open(supp_file, 'r') as f:
            content = f.read()
        
        required_sections = [
            "Extended Experimental Results",
            "Statistical Analysis Details", 
            "Implementation Details",
            "Reproducibility Materials"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
            else:
                details.append(f"Found: {section}")
        
        if missing_sections:
            details.extend([f"Missing: {s}" for s in missing_sections])
            return False, details
        
        details.append(f"Content length: {len(content):,} characters")
        return True, details
    
    def validate_reproducibility(self) -> Tuple[bool, List[str]]:
        """Validate reproducibility requirements"""
        repro_file = self.submission_dir / "NEURIPS_2025_REPRODUCIBILITY_CHECKLIST.md"
        details = []
        
        if not repro_file.exists():
            return False, ["Reproducibility checklist not found"]
        
        with open(repro_file, 'r') as f:
            content = f.read()
        
        # Count completed items
        completed_count = content.count("- [x]")
        total_count = content.count("- [")
        
        details.append(f"Completed items: {completed_count}/{total_count}")
        
        if completed_count >= 40:  # Should have most items completed
            details.append("Comprehensive reproducibility documentation")
            return True, details
        else:
            details.append("More reproducibility items need completion")
            return False, details
    
    def validate_ethics(self) -> Tuple[bool, List[str]]:
        """Validate ethics statement"""
        ethics_file = self.submission_dir / "NEURIPS_2025_ETHICS_STATEMENT.md"
        details = []
        
        if not ethics_file.exists():
            return False, ["Ethics statement not found"]
        
        with open(ethics_file, 'r') as f:
            content = f.read()
        
        required_topics = [
            "Potential Benefits",
            "Potential Risks", 
            "Mitigation Strategies",
            "Broader Impact"
        ]
        
        covered_topics = []
        for topic in required_topics:
            if topic.lower() in content.lower():
                covered_topics.append(topic)
                details.append(f"Addresses: {topic}")
        
        if len(covered_topics) >= 3:
            details.append("Comprehensive ethics analysis")
            return True, details
        else:
            details.append("Ethics statement needs more comprehensive coverage")
            return False, details
    
    def validate_file_sizes(self) -> Tuple[bool, List[str]]:
        """Validate file sizes are reasonable"""
        details = []
        size_warnings = []
        
        # Check main files
        main_files = [
            "neurips_2025_lethe_submission.pdf",
            "SUPPLEMENTARY_MATERIALS.md"
        ]
        
        for filename in main_files:
            filepath = self.submission_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                details.append(f"{filename}: {size_mb:.1f}MB")
                
                if size_mb > 10:  # More than 10MB is large for academic submission
                    size_warnings.append(f"{filename} is large ({size_mb:.1f}MB)")
        
        # Calculate total package size
        total_size = sum(f.stat().st_size for f in self.submission_dir.rglob('*') if f.is_file())
        total_mb = total_size / (1024 * 1024)
        details.append(f"Total package size: {total_mb:.1f}MB")
        
        if size_warnings:
            details.extend(size_warnings)
            details.append("Consider compressing large files")
        
        return True, details  # Size warnings, not failures
    
    def generate_validation_report(self, all_passed: bool):
        """Generate validation report"""
        report = {
            "validation_timestamp": "2025-08-25T12:00:00Z",
            "overall_status": "PASSED" if all_passed else "FAILED",
            "checks": self.results,
            "summary": {
                "total_checks": len(self.results),
                "passed_checks": sum(1 for r in self.results if r["passed"]),
                "failed_checks": sum(1 for r in self.results if not r["passed"])
            }
        }
        
        report_file = self.submission_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Validation report saved: {report_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate NeurIPS 2025 submission")
    parser.add_argument("--submission-dir", default=".", 
                       help="Submission directory path")
    args = parser.parse_args()
    
    validator = NeurIPSSubmissionValidator(args.submission_dir)
    success = validator.validate_submission()
    
    exit(0 if success else 1)