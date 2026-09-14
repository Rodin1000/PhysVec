# corresponding to 'set -e'
$ErrorActionPreference = "Stop"

# --- set parameters ---
$topic = "dmrg"
$task_name = "task_1_2.json"
$code_file = "../logs/dmrg/_PhysRevB.58.R14741_Fig._2_devtest/code_LLM.jl"  # Specify the code file that you want to evaluate
$idname_proj = "devtest"
$idname_rubrics = "devtest"
$user_model_rubricsgrader = "deepseek/deepseek-v3.1-terminus"


# --- generate code and repository ---
python ../src/rubrics_v2_5.py `
    --topic $topic `
    --task_name $task_name `
    --code_file $code_file `
    --idname_proj $idname_proj `
    --idname_rubrics $idname_rubrics `
    --user_model_rubricsgrader $user_model_rubricsgrader
    