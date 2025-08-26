#!/usr/bin/env python3
"""
Run Milestone 6: Comprehensive Metrics & Evaluation Protocol

Single command to produce metrics.json and plots under ./results/HW_PROFILE/

Usage:
    python run_milestone6_evaluation.py --dataset ./datasets/lethebench_agents.json
    python run_milestone6_evaluation.py --dataset ./datasets/lethebench_agents.json --quick-test
    python run_milestone6_evaluation.py --dataset ./datasets/lethebench_agents.json --hardware-profile "M2_MacBook_Air"
"""

import sys
import logging
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the evaluation framework
from eval.milestone6_evaluation import main

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('milestone6_evaluation.log')
        ]
    )
    
    # Run the evaluation
    main()