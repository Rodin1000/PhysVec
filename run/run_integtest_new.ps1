# corresponding to 'set -e'
$ErrorActionPreference = "Stop"

# --- set parameters ---
$topic = "dmrg"
$task_name = "task_2_1.json"
$repo_dir = "../Output_repo/dmrg/repo_PhysRevLett.95.240404_Fig._1_a_test2"
$idname_proj = "test2"
$idname_integtest = "test2"
$upper_num_integcode = 0    # How many run functions to tested in this run; 0 means all
$level_num = 0              # How many re-construction levels in this run; 0 means all
$upper_num_execode = $level_num  # How many integration codes are executed in this run; set to level_num
$isintegcode = $true       # Whether to generate integration codes
$isexecode = $true          # Whether to execute integration codes
$user_model_codeintegrator = "openai/gpt-5"
$judge_model_codeintegrator = "openai/gpt-5"    # the model used for judging integration code execution results
$sandbox_dir = "../CodeIntegrator_sandbox/qcmb/integtest_npj_Quantum_Information_5_106_Fig._2_a_test5"  # Specify the sandbox containing the integration codes if you set is integcode to false and isexecode to true
$concurr_num_integtest = 10  # Number of concurrency



# --- generate code and repository ---
python ../src/integtest_v4.py `
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
    --judge_model_codeintegrator $judge_model_codeintegrator `
    --sandbox_dir $sandbox_dir `
    --concurr_num_integtest $concurr_num_integtest
    
