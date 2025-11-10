#!/bin/bash

# Optimization Strategy Comparison Runner
# This script runs the optimization comparison analysis

echo "Starting NSA Optimization Strategy Comparison..."
echo "This may take 10-15 minutes to complete the full grid search..."

# Change to the analysis directory
cd "$(dirname "$0")"

# Run the optimization comparison script
python optimization_comparison.py

echo "Analysis complete! Check the results directory for outputs."