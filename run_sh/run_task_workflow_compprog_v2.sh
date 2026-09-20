#!/bin/bash
# task_workflow_compprog_v2: author + (unittest + integtest + postprocess + coderepair) loop
set -e
export CUDA_VISIBLE_DEVICES=6

# --- set tags ---
tag1="TESTAPI"  # author_format_judge

# --- set parameters ---
topic="dmrg"
result_dir="../results/results_${tag1}"
output_repo_dir="../results/results_${tag1}_repo"
isall="false"
task_list="task_dmrg_3_2"

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
