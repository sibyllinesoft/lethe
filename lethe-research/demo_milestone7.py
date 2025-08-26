#!/usr/bin/env python3
"""
Milestone 7 Demo Script
Quick demonstration of publication-ready analysis pipeline.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and show results"""
    print(f"\n🚀 {description}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print("❌ FAILED")
            print("STDERR:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT (5 minutes exceeded)")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

def main():
    """Run Milestone 7 demonstration"""
    
    print("🎯 Milestone 7: Publication-Ready Analysis Pipeline Demo")
    print("=" * 80)
    
    # Check if validation script exists
    validation_script = Path("validate_milestone7_implementation.py")
    if not validation_script.exists():
        print("❌ Validation script not found!")
        sys.exit(1)
    
    # 1. Run validation
    run_command([
        "python", "validate_milestone7_implementation.py"
    ], "Implementation Validation")
    
    # 2. Show make targets
    print("\n📋 Available Make Targets:")
    print("=" * 60)
    
    available_targets = [
        ("make figures", "Generate all publication outputs"),
        ("make milestone7-quick", "Quick test with synthetic data"),
        ("make tables", "Generate only LaTeX + CSV tables"),
        ("make plots", "Generate only publication plots"), 
        ("make sanity-checks", "Run experimental validation"),
        ("make clean-analysis", "Clean all analysis outputs"),
        ("make analysis-summary", "Show generated files")
    ]
    
    for target, description in available_targets:
        print(f"  {target:<20} - {description}")
    
    # 3. Quick test
    print("\n🧪 Running Quick Test")
    print("=" * 60)
    run_command([
        "python", "run_milestone7_analysis.py", "--quick-test"
    ], "Quick Analysis with Synthetic Data")
    
    # 4. Show output structure
    print("\n📁 Expected Output Structure:")
    print("=" * 60)
    print("""
./analysis/hardware_profiles/SYSTEM_NAME/
├── hardware_profile.json
├── tables/
│   ├── quality_metrics.csv + .tex
│   ├── agent_metrics.csv + .tex  
│   └── efficiency_metrics.csv + .tex
├── figures/
│   ├── scalability_latency_vs_corpus_size.png
│   ├── throughput_qps_vs_concurrency.png
│   ├── quality_vs_latency_tradeoffs.png
│   ├── quality_vs_memory_tradeoffs.png
│   └── agent_scenario_breakdown.png
├── sanity_checks/
│   └── sanity_check_report.json
└── milestone7_completion_report.json
    """)
    
    # 5. Usage examples
    print("\n💡 Usage Examples:")
    print("=" * 60)
    print("""
# Complete publication pipeline
make figures

# With existing evaluation data  
python run_milestone7_analysis.py \\
  --metrics-file analysis/final_statistical_gatekeeper_results.json \\
  --train-data datasets/lethebench \\
  --test-data datasets/lethebench

# Custom hardware profile
python run_milestone7_analysis.py \\
  --quick-test \\
  --hardware-profile "Custom_System_Name"
    """)
    
    print("\n✅ Milestone 7 Demo Complete!")
    print("\nNext steps:")
    print("1. Review validation results above")
    print("2. Run 'make milestone7-quick' for full synthetic test")
    print("3. Run 'make figures' with real data for publication outputs")

if __name__ == "__main__":
    main()