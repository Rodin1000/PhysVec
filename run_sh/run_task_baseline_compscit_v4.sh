#!/bin/bash
# task_baseline_compscit_v4: Merge + Modify_code_plotsave + Execute_and_refine in baseline tag subdir.
set -e
export CUDA_VISIBLE_DEVICES=6

# --- set tags ---
tag1="BATCH5_15000token_18_1_18"
tag_name="freereact"

# --- set parameters ---
topic="dmrg"
result_dir="../results/Sciresults_${tag1}"
ref_result_dir="../results/results_${tag1}"
isall="false"
task_list="task_dmrg_1_3"

# Model parameters
model_author="gemini-2.5-flash-nothinking"
model_step_repair_role_author="$model_author"

# Iteration parameters
max_iter_refine=5
timeout=30000

# --- run baseline compscit workflow ---
python ../src/task_baseline_compscit_v4.py \
    --topic "$topic" \
    --result_dir "$result_dir" \
    --ref_result_dir "$ref_result_dir" \
    --tag_name "$tag_name" \
    --isall "$isall" \
    --task_list "$task_list" \
    --model_step_repair_role_author "$model_step_repair_role_author" \
    --max_iter_refine "$max_iter_refine" \
    --timeout "$timeout"
