# corresponding to 'set -e'
$ErrorActionPreference = "Stop"

# --- set parameters ---
$topic = "dmrg"
$task_name = "task_1_2.json"
$repo_dir = "../Output_repo/dmrg/repo_PhysRevB.58.R14741_Fig._2_devtest"
$idname_proj = "devtest"
$idname_integtest = "devtest"
$upper_num_integcode = 0    # How many run functions to tested in this run; 0 means all
$level_num = 2              # How many re-construction levels in this run; 0 means all
$upper_num_execode = $level_num  # How many integration codes are executed in this run; set to level_num
$isintegcode = $true       # Whether to generate integration codes
$isexecode = $true          # Whether to execute integration codes
$user_model_codeintegrator = "deepseek/deepseek-v3.1-terminus"
$sandbox_dir = "../CodeIntegrator_sandbox/dmrg/integtest_PhysRevB.58.R14741_Fig._1_devtest"  # Specify the sandbox containing the integration codes if you set is integcode to false and isexecode to true
$concurr_num_integtest = 10  # Number of concurrency



# --- generate code and repository ---
python ../src/integtest_v2_5.py `
    --topic $topic `
    --task_name $task_name `
    --repo_dir $repo_dir `
    --idname_proj $idname_proj `
    --idname_integtest $idname_integtest `
    --upper_num_integcode $upper_num_integcode `
    --level_num $level_num `
    --upper_num_execode $upper_num_execode `
    --isintegcode $isintegcode `
    --isexecode $isexecode `
    --user_model_codeintegrator $user_model_codeintegrator `
    --sandbox_dir $sandbox_dir `
    --concurr_num_integtest $concurr_num_integtest
    
