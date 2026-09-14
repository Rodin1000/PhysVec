#!/bin/bash
# task_baseline_ReAct_v2: batch ReAct baseline
set -e
export CUDA_VISIBLE_DEVICES=7

# --- set tags ---
tag1="BATCH5_15000token_18_1_18"
model_author="gemini-2.5-flash-nothinking"


# tag1="BATCH5_15000token_13_1_13"
# model_author="gpt-5.1-2025-11-13"


# tag1="BATCH5_15000token_22_1_22"
# model_author="deepseek-v3"


tag_name="freereactrag"
# iscodefile: true => skip Phase 1, use code from *_check/code_LLM_loop1
iscodefile="false"
# isauthorrag: true => Retrieve (RAG) between Plan and Generate
isauthorrag="false"
# isrepairrag: true => Retrieve (RAG) between Suggest and Repair
isrepairrag="true"
concurr_num_retrieve=5

# --- set parameters ---
topic="dft_qc"
result_dir="../results/results_${tag1}"
isall="false"
task_list="task_dft_qc_1_2.json,task_dft_qc_2_2.json,task_dft_qc_3_2.json,task_dft_qc_4_2.json,task_dft_qc_5_2.json"

# Model parameters (same style as compprog_v2)
model_programtest="$model_author"  #"gpt-5.2"
user_model_planner="$model_author"
user_model_coder="$model_author"
user_model_suggest="$model_programtest"

# Iteration parameters
max_iter_ReAct_dft=8


# --- run ReAct baseline ---
python ../src/task_baseline_ReAct_dft_v2.py \
    --topic "$topic" \
    --result_dir "$result_dir" \
    --isall "$isall" \
    --task_list "$task_list" \
    --tag_name "$tag_name" \
    --user_model_planner "$user_model_planner" \
    --user_model_coder "$user_model_coder" \
    --user_model_suggest "$user_model_suggest" \
    --max_iter_ReAct_dft "$max_iter_ReAct_dft" \
    --iscodefile "$iscodefile" \
    --isauthorrag "$isauthorrag" \
    --isrepairrag "$isrepairrag" \
    --concurr_num_retrieve "$concurr_num_retrieve"
