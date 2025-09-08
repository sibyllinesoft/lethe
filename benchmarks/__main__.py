#!/usr/bin/env python3
"""
Benchmarks Package Main Entry Point
===================================

Provides command-line interface for comprehensive benchmarking.
"""

import sys
import os
from pathlib import Path

# Add the benchmarks package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.orchestrator import main

if __name__ == "__main__":
    main()