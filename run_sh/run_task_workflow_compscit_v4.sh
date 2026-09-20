#!/bin/bash
# task_workflow_compscit_v4: merge smallscale+fullparas -> code_LLM_loop1, then loop rubrics_v9 -> suggest -> retrieve -> refine -> repair (no repo).
# Supports ref_result_dir to copy from ref xxx_check before workflow.
set -e
export CUDA_VISIBLE_DEVICES=6

# --- set tags ---
tag1="BATCH5_15000token_18_1_18"

# --- set parameters ---
topic="dmrg"
result_dir="../results/Sciresults_TESTAPI"
ref_result_dir="../results/results_${tag1}"
task_list="task_dmrg_2_2"

# Model parameters (rubrics + repair)
model_author="qwen/qwen3-max"
user_model_rubricsgrader="$model_author"
model_step_repair_role_suggest="$model_author"
model_step_repair_role_author="$model_author"

# Iteration parameters
max_workflow_iter=5
concurr_num_retrieve=5
num_rubrics_grade=3
concurr_num_limiting=3

# Convtest parameters (Step 2)
max_convtest_loop=6
concurr_num_convtest=3
max_convtest_refine=5
convtest_timeout=30000
exec_refine_timeout=30000

# --- run compscit workflow ---
python ../src/task_workflow_compscit_v4.py \
    --result_dir "$result_dir" \
    --ref_result_dir "$ref_result_dir" \
    --topic "$topic" \
    --task_list "$task_list" \
    --max_workflow_iter "$max_workflow_iter" \
    --user_model_rubricsgrader "$user_model_rubricsgrader" \
    --model_step_repair_role_suggest "$model_step_repair_role_suggest" \
    --model_step_repair_role_author "$model_step_repair_role_author" \
    --concurr_num_retrieve "$concurr_num_retrieve" \
    --num_rubrics_grade "$num_rubrics_grade" \
    --concurr_num_limiting "$concurr_num_limiting" \
    --max_convtest_loop "$max_convtest_loop" \
    --concurr_num_convtest "$concurr_num_convtest" \
    --max_convtest_refine "$max_convtest_refine" \
    --convtest_timeout "$convtest_timeout" \
    --exec_refine_timeout "$exec_refine_timeout"
