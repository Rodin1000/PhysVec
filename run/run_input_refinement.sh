#!/usr/bin/env bash
set -euo pipefail

# --- set parameters ---
log_folder="logs/dft_qc/_ja5b10819_Fig._8_panel_a_test2"  # specify the log folder path (contains code_LLM.inp and log_code_*.jsonl)
user_model="deepseek/deepseek-v3.1-terminus"   # LLM model for refinement
max_iterations=10       # maximum iterations for refinement
program_timeout=60     # timeout for ORCA execution (seconds)
refine_query_num=60      # number of queries for RAG-based refinement
output_dir=""           # output directory (empty = log_folder/refined)

# --- run input refinement workflow ---
python ../src/input_refinement.py \
  --log_folder "$log_folder" \
  --user_model "$user_model" \
  --max_iterations "$max_iterations" \
  --program_timeout "$program_timeout" \
  --refine_query_num "$refine_query_num" \
  ${output_dir:+--output_dir "$output_dir"}
