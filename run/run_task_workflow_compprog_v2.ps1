# PowerShell entry point for the DFT programming workflow.
# Uses python from the active environment; activate .venv before running.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# task_workflow_compprog_v2: author + (unittest + integtest + postprocess + coderepair) loop

$env:CUDA_VISIBLE_DEVICES=7

# --- set tags ---

$model_author="qwen/qwen3-max"
$tag1="TESTAPI"  # author_format_judge gemini-2.5-flash-nothinking

# model_author="gpt-5.1-2025-11-13"
# tag1="BATCH5_15000token_13_1_13"  # author_format_judge deepseek-v3


# model_author="deepseek-v3"
# tag1="BATCH5_15000token_22_1_22"  # author_format_judge deepseek-v3


# --- set parameters ---
$topic="dft_qc"
$result_dir="../results/results_${tag1}"
$output_repo_dir="../results/results_${tag1}_repo"
$isall="false"
$task_list="task_dft_qc_1_1.json,task_dft_qc_2_1.json,task_dft_qc_3_1.json,task_dft_qc_4_1.json,task_dft_qc_4_2.json,"


# Model parameters (author step)
# Use non-thinking model for MCP/tool-using roles to avoid thought_signature API error
# (Thinking models require thought_signature in functionCall parts; SDK does not support it)
# model_author="qwen3-next-80b-a3b-instruct"
# model_author="deepseek-r1"
# model_author="claude-sonnet-4-5-20250929"



$model_author_judge="qwen/qwen3-max" # or try "google/gemini-3-flash-preview" for better performance

$model_step_author_role_planner="$model_author"
$model_step_author_role_coder="$model_author"
$model_step_author_role_repogenerator="$model_author"
$model_step_author_role_coderefiner="$model_author"
$model_step_author_role_codejudge="$model_author_judge"

# Model parameters (unittest step)
$model_programtest="$model_author"   #"gpt-5.2"
$model_step_unittest_role_codeverifier="$model_programtest"
$model_step_unittest_role_judge="$model_programtest"

# Model parameters (integtest step)
$model_step_integtest_role_codeintegrator="$model_programtest"
$model_step_integtest_role_judge="$model_programtest"

# Model parameters (repair step)
$model_step_repair_role_suggest="$model_programtest"
$model_step_repair_role_author="$model_author"
$model_step_repair_role_judge="$model_author_judge"

# Iteration parameters
$max_iter_refinecode_rules=5
$max_iter_retry=4
$max_workflow_iter=8
$concurr_num_retrieve=5

# Sandbox base directories (program appends tag1/topic/task_stem, e.g. CodeVerifier_sandbox/tag1/dmrg/task_dmrg_2_4/)
$output_sandbox_dir_unittest="../CodeVerifier_sandbox"
$output_sandbox_dir_integtest="../CodeIntegrator_sandbox"



# Run from run/ so the existing ../ paths resolve as in run_sh/.
$workflowArgs = @(
    "../src/task_workflow_compprog_dft_v2.py"
    "--topic", $topic
    "--result_dir", $result_dir
    "--output_repo_dir", $output_repo_dir
    "--isall", $isall
    "--task_list", $task_list
    "--model_step_author_role_planner", $model_step_author_role_planner
    "--model_step_author_role_coder", $model_step_author_role_coder
    "--model_step_author_role_repogenerator", $model_step_author_role_repogenerator
    "--model_step_author_role_coderefiner", $model_step_author_role_coderefiner
    "--model_step_author_role_codejudge", $model_step_author_role_codejudge
    "--model_step_unittest_role_codeverifier", $model_step_unittest_role_codeverifier
    "--model_step_unittest_role_judge", $model_step_unittest_role_judge
    "--model_step_integtest_role_codeintegrator", $model_step_integtest_role_codeintegrator
    "--model_step_integtest_role_judge", $model_step_integtest_role_judge
    "--model_step_repair_role_suggest", $model_step_repair_role_suggest
    "--model_step_repair_role_author", $model_step_repair_role_author
    "--model_step_repair_role_judge", $model_step_repair_role_judge
    "--max_iter_refinecode_rules", $max_iter_refinecode_rules
    "--max_iter_retry", $max_iter_retry
    "--max_workflow_iter", $max_workflow_iter
    "--concurr_num_retrieve", $concurr_num_retrieve
    "--output_sandbox_dir_unittest", $output_sandbox_dir_unittest
    "--output_sandbox_dir_integtest", $output_sandbox_dir_integtest
)

Push-Location -LiteralPath $PSScriptRoot
try {
    & python @workflowArgs
    $workflowExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $workflowExitCode
