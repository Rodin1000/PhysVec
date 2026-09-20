#!/bin/bash
# Continue task_workflow_compprog_v2 from an existing loop into a fully isolated run.
set -e
export CUDA_VISIBLE_DEVICES=6

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

# --- source and destination run tags ---
source_tag="BATCH5_15000token_18_1_18"
continue_tag="BATCH5_15000token_18_1_18_TESTAPI"

# --- original task_workflow_compprog_v2 parameters ---
topic="nnwf"
result_dir="${project_root}/results/results_${continue_tag}"
output_repo_dir="${project_root}/results/results_${continue_tag}_repo"
isall="false"
task_list="task_nnwf_2_2"

# Author parameters are retained for CLI/config compatibility; continuation skips author.
model_author="qwen/qwen3-max"
model_author_judge="qwen/qwen3-max" # or try "google/gemini-3-flash-preview" for better performance
model_step_author_role_planner="$model_author"
model_step_author_role_coder="$model_author"
model_step_author_role_repogenerator="$model_author"
model_step_author_role_coderefiner="$model_author"
model_step_author_role_codejudge="$model_author_judge"

# Programming-test models.
model_programtest="$model_author"
model_step_unittest_role_codeverifier="$model_programtest"
model_step_unittest_role_judge="$model_programtest"
model_step_integtest_role_codeintegrator="$model_programtest"
model_step_integtest_role_judge="$model_programtest"

# Repair models.
model_step_repair_role_suggest="$model_programtest"
model_step_repair_role_author="$model_author"
model_step_repair_role_judge="$model_author_judge"

# Original iteration parameters. Here max_workflow_iter means additional loops.
max_iter_refinecode_rules=5
max_iter_retry=4
max_workflow_iter=4
concurr_num_retrieve=5

# New run sandbox roots; never point these at the source run's sandbox directories.
output_sandbox_dir_unittest="${project_root}/CodeVerifier_sandbox/${continue_tag}"
output_sandbox_dir_integtest="${project_root}/CodeIntegrator_sandbox/${continue_tag}"

# --- continuation-only parameters ---
resume_result_dir="${project_root}/results/results_${source_tag}"
resume_repo_dir="${project_root}/results/results_${source_tag}_repo"
resume_from_loop=8
only_failed_fullcode="true"
include_missing_loop_report="false"
allow_existing_result_dir="true"
dry_run="false"  # First run should stay true; change to false only after checking the selection and paths.

python "${project_root}/src/task_workflow_compprog_continue.py" \
    --topic "$topic" \
    --result_dir "$result_dir" \
    --output_repo_dir "$output_repo_dir" \
    --isall "$isall" \
    --task_list "$task_list" \
    --model_step_author_role_planner "$model_step_author_role_planner" \
    --model_step_author_role_coder "$model_step_author_role_coder" \
    --model_step_author_role_repogenerator "$model_step_author_role_repogenerator" \
    --model_step_author_role_coderefiner "$model_step_author_role_coderefiner" \
    --model_step_author_role_codejudge "$model_step_author_role_codejudge" \
    --model_step_unittest_role_codeverifier "$model_step_unittest_role_codeverifier" \
    --model_step_unittest_role_judge "$model_step_unittest_role_judge" \
    --model_step_integtest_role_codeintegrator "$model_step_integtest_role_codeintegrator" \
    --model_step_integtest_role_judge "$model_step_integtest_role_judge" \
    --model_step_repair_role_suggest "$model_step_repair_role_suggest" \
    --model_step_repair_role_author "$model_step_repair_role_author" \
    --model_step_repair_role_judge "$model_step_repair_role_judge" \
    --max_iter_refinecode_rules "$max_iter_refinecode_rules" \
    --max_iter_retry "$max_iter_retry" \
    --max_workflow_iter "$max_workflow_iter" \
    --concurr_num_retrieve "$concurr_num_retrieve" \
    --output_sandbox_dir_unittest "$output_sandbox_dir_unittest" \
    --output_sandbox_dir_integtest "$output_sandbox_dir_integtest" \
    --resume_result_dir "$resume_result_dir" \
    --resume_repo_dir "$resume_repo_dir" \
    --resume_from_loop "$resume_from_loop" \
    --only_failed_fullcode "$only_failed_fullcode" \
    --include_missing_loop_report "$include_missing_loop_report" \
    --allow_existing_result_dir "$allow_existing_result_dir" \
    --dry_run "$dry_run"
