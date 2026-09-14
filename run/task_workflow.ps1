# corresponding to 'set -e'
$ErrorActionPreference = "Stop"

# --- set parameters ---
# author
$task_name = "task_1_1.json"
$idname_proj = "res1107-2"
$idname_code = $idname_proj
$idname_repo = $idname_proj
$user_model_planner = "deepseek/deepseek-v3.1-terminus"
$user_model_coder = "openai/gpt-5"
$user_model_repo = "deepseek/deepseek-v3.1-terminus"

# unittest
$CodeVerifier_library_dir = "../CodeVerifier_library_compact"
$idname_unittest = $idname_code
$upper_num_gencode_unittest = 0
$upper_num_exec_unittest = 0
$user_model_codeverifier = "deepseek/deepseek-v3.1-terminus"
$concurr_num_unittest = 10

# integtest
$idname_integtest = $idname_code
$upper_num_integtest = 0
$level_num_integtest = 0
$upper_num_execode_integtest = 0
$user_model_codeintegrator = "deepseek/deepseek-v3.1-terminus"
$concurr_num_integtest = 10

# rubrics
$idname_rubrics = $idname_code
$user_model_rubrics = "deepseek/deepseek-v3.1-terminus"


# --- generate code and repository ---
python ../src/task_workflow.py `
    --task_name $task_name `
    --idname_proj $idname_proj `
    --idname_code $idname_code `
    --idname_repo $idname_repo `
    --user_model_planner $user_model_planner `
    --user_model_coder $user_model_coder `
    --user_model_repo $user_model_repo `
    --CodeVerifier_library_dir $CodeVerifier_library_dir `
    --idname_unittest $idname_unittest `
    --upper_num_gencode_unittest $upper_num_gencode_unittest `
    --upper_num_exec_unittest $upper_num_exec_unittest `
    --user_model_codeverifier $user_model_codeverifier `
    --concurr_num_unittest $concurr_num_unittest `
    --idname_integtest $idname_integtest `
    --upper_num_integtest $upper_num_integtest `
    --level_num_integtest $level_num_integtest `
    --upper_num_execode_integtest $upper_num_execode_integtest `
    --user_model_codeintegrator $user_model_codeintegrator `
    --concurr_num_integtest $concurr_num_integtest `
    --idname_rubrics $idname_rubrics `
    --user_model_rubrics $user_model_rubrics `

