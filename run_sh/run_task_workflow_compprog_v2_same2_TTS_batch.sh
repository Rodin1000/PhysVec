#!/bin/bash
# task_workflow_compprog_v2: author + (unittest + integtest + postprocess + coderepair) loop
set -e
export CUDA_VISIBLE_DEVICES=7

# --- set mode and indices (edit here) ---
# mode: "list" or "range"
mode="range"
# For list mode: indices in [1,2,3,4,5] format
indices_list=""
# For range mode: start and end
range_start=7
range_end=15

# --- build indices array from config ---
indices=()
case "$mode" in
    list)
        raw="$indices_list"
        if [[ ! "$raw" =~ ^\[.*\]$ ]]; then
            echo "Error: list mode requires [1,2,3,4,5] format (with square brackets)"
            exit 1
        fi
        inner="${raw:1:${#raw}-2}"
        IFS=',' read -ra indices <<< "$inner"
        ;;
    range)
        for i in $(seq "$range_start" "$range_end"); do
            indices+=("$i")
        done
        ;;
    *)
        echo "Error: mode must be 'list' or 'range'"
        exit 1
        ;;
esac

# --- set parameters (tag1 set per iteration below) ---
topic="nnwf"
isall="false"
task_list="task_nnwf_2_2,task_nnwf_4_2"

for idx in "${indices[@]}"; do
    tag1="TTS${idx}_15000token_18_1_18"
    result_dir="../results/results_${tag1}"
    output_repo_dir="../results/results_${tag1}_repo"
    echo "=== Running index $idx (tag1=$tag1) ==="

# Model parameters (author step)
model_author="qwen/qwen3-max"
model_author_judge="qwen/qwen3-max" # or try "google/gemini-3-flash-preview" for better performance
model_step_author_role_planner="$model_author"
model_step_author_role_coder="$model_author"
model_step_author_role_repogenerator="$model_author"
model_step_author_role_coderefiner="$model_author"
model_step_author_role_codejudge="$model_author_judge"

# Model parameters (unittest step)
model_programtest="$model_author"   #"gpt-5.2"
model_step_unittest_role_codeverifier="$model_programtest"
model_step_unittest_role_judge="$model_programtest"

# Model parameters (integtest step)
model_step_integtest_role_codeintegrator="$model_programtest"
model_step_integtest_role_judge="$model_programtest"

# Model parameters (repair step)
model_step_repair_role_suggest="$model_programtest"
model_step_repair_role_author="$model_author"
model_step_repair_role_judge="$model_author_judge"

# Iteration parameters
max_iter_refinecode_rules=5
max_iter_retry=4
max_workflow_iter=8
concurr_num_retrieve=5

# Sandbox base directories (program appends tag1/topic/task_stem, e.g. CodeVerifier_sandbox/tag1/dmrg/task_dmrg_2_4/)
output_sandbox_dir_unittest="../CodeVerifier_sandbox"
output_sandbox_dir_integtest="../CodeIntegrator_sandbox"


# --- run compprog workflow ---
python ../src/task_workflow_compprog_v2.py \
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
    --output_sandbox_dir_integtest "$output_sandbox_dir_integtest"

done
