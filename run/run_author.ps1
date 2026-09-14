# corresponding to 'set -e'
$ErrorActionPreference = "Stop"

# --- set parameters ---
$topic = "nnwf"  # specify the topic of the paper
$task_name = "task_1_2.json"  # specify the task json file
$idname_proj = "devtest"      # this name will be used to create a project folder under ../logs/
$idname_code = "devtest"  
$idname_repo = "devtest"
$user_model_planner = "deepseek/deepseek-v3.1-terminus"   # the model used for project planning 
$user_model_coder = "openai/gpt-4o"         # the model used for code generation 
$user_model_repo = "deepseek/deepseek-v3.1-terminus"      # the model used for repository generation
$iscode = $true   # whether to generate code in this run
$isrepo = $true   # whether to generate repository in this run 
$code_file = "../logs/dmrg/_PhysRevB.58.R14741_Fig._1_devtest/code_LLM.jl" # if iscode is false, this input specifies the code file to be splitted into repo or refined
$user_model_coderefiner = "deepseek/deepseek-v3.1-terminus"  # the model used for code refinement
$isrefinecode_rules = $false   # whether to refine the generated code based on rule-based checks
$max_iter_refinecode_rules = 1


# --- generate code and repository ---
python ../src/author_v2_5.py `
    --topic $topic `
    --task_name $task_name `
    --idname_proj $idname_proj `
    --idname_code $idname_code `
    --idname_repo $idname_repo `
    --user_model_planner $user_model_planner `
    --user_model_coder $user_model_coder `
    --user_model_repo $user_model_repo `
    --iscode $iscode `
    --isrepo $isrepo `
    --code_file $code_file `
    --user_model_coderefiner $user_model_coderefiner `
    --isrefinecode_rules $isrefinecode_rules `
    --max_iter_refinecode_rules $max_iter_refinecode_rules 
