#!/bin/bash
# Continue task_baseline_ReAct_v2 from an existing loop into an isolated result run.
set -e
export CUDA_VISIBLE_DEVICES=6

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${project_root}/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "Project virtual-environment Python not found: $python_bin" >&2
    exit 1
fi

# --- source and destination run tags ---
source_tag="BATCH5_15000token_18_1_18"
continue_tag="BATCH5_15000token_18_1_18_continue_2"

# --- original task_baseline_ReAct_v2 parameters ---
topic="nnwf"
result_dir="${project_root}/results/results_${continue_tag}"
isall="false"
task_list="task_nnwf_2_2"
tag_name="freereactrag"

# Phase 1 parameters are retained for CLI/settings compatibility; continuation skips Phase 1.
model_author="gemini-2.5-flash-nothinking"
model_programtest="$model_author"
user_model_planner="$model_author"
user_model_coder="$model_author"
user_model_suggest="$model_programtest"
iscodefile="false"
isauthorrag="false"

# Repair-loop settings retain the normal ReAct behavior.
isrepairrag="true"
concurr_num_retrieve=5
# In continuation mode this is the number of additional code versions to execute.
max_iter_ReAct=4

# --- continuation-only parameters ---
resume_result_dir="${project_root}/results/results_${source_tag}"
resume_tag_name="freereactrag"
resume_from_loop=8
only_failed_at_resume_loop="true"
include_missing_execute_report="false"
allow_existing_result_dir="true"
dry_run="false"  # Inspect selection first; change to false only after confirming it.

"$python_bin" "${project_root}/src/task_baseline_ReAct_continue.py" \
    --topic "$topic" \
    --result_dir "$result_dir" \
    --isall "$isall" \
    --task_list "$task_list" \
    --tag_name "$tag_name" \
    --user_model_planner "$user_model_planner" \
    --user_model_coder "$user_model_coder" \
    --user_model_suggest "$user_model_suggest" \
    --max_iter_ReAct "$max_iter_ReAct" \
    --iscodefile "$iscodefile" \
    --isauthorrag "$isauthorrag" \
    --isrepairrag "$isrepairrag" \
    --concurr_num_retrieve "$concurr_num_retrieve" \
    --resume_result_dir "$resume_result_dir" \
    --resume_tag_name "$resume_tag_name" \
    --resume_from_loop "$resume_from_loop" \
    --only_failed_at_resume_loop "$only_failed_at_resume_loop" \
    --include_missing_execute_report "$include_missing_execute_report" \
    --allow_existing_result_dir "$allow_existing_result_dir" \
    --dry_run "$dry_run"
