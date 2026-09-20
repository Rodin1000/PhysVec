#!/bin/bash
# task_baseline_ReAct_v2: batch ReAct baseline
# Run this programming baseline before run_task_baseline_compscit_v4.sh.
# A usable result contains code_ReAct_fullparas and at least one code_ReAct_loopN.
set -e
export CUDA_VISIBLE_DEVICES=6

# --- set tags ---
tag1="BATCH5_15000token_18_1_18_TESTAPI"
# Recommended tags: freereact (no RAG) or freereactrag (ReAct-RAG).
# The matching baseline compscit script must use the same tag_name.
tag_name="freereactrag"
# iscodefile: true => skip Phase 1, use code from *_check/code_LLM_loop1
iscodefile="false"
# isauthorrag: true => Retrieve (RAG) between Plan and Generate
isauthorrag="false"
# isrepairrag: true => Retrieve (RAG) between Suggest and Repair
isrepairrag="true"
concurr_num_retrieve=5

# --- set parameters ---
topic="dmrg"
result_dir="../results/results_${tag1}"
isall="false"
task_list="task_dmrg_7_2"

# Model parameters (same style as compprog_v2)
model_author="qwen/qwen3-max"
model_programtest="$model_author"  #"gpt-5.2"
user_model_planner="$model_author"
user_model_coder="$model_author"
user_model_suggest="$model_programtest"

# Iteration parameters
max_iter_ReAct=8


# --- run ReAct baseline ---
python ../src/task_baseline_ReAct_v2.py \
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
    --concurr_num_retrieve "$concurr_num_retrieve"
