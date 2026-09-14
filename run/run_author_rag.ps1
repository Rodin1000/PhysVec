# corresponding to 'set -e'
$ErrorActionPreference = "Stop"

# --- set parameters ---
$topic = "dmrg"  # specify the topic of the paper
$task_name = "task_2_1.json"  # specify the task json file
$idname_proj = "test2"      # this name will be used to create a project folder under ../logs/
$idname_code = "test2"  
$idname_repo = "test2"

$user_model = "openai/gpt-5"#"google/gemini-2.5-flash"
$user_model_planner = $user_model   # the model used for project planning 
$user_model_coder = $user_model         # the model used for code generation 
$user_model_repo = $user_model      # the model used for repository generation
$iscode = $false   # whether to generate code in this run
$isrepo = $true   # whether to generate repository in this run 
$code_file = "../logs/dmrg/_PhysRevLett.95.240404_Fig._1_a_test2/code_LLM.jl" # if iscode is false, this input specifies the code file to be splitted into repo or refined
$user_model_coderefiner = $user_model  # the model used for code refinement
$user_model_codejudge = "openai/gpt-5"  # the model used for code judgment
$isrefinecode_rules = $false   # whether to refine the generated code based on rule-based checks
$max_iter_refinecode_rules = 5


# --- generate code and repository ---
python ../src/author_v4.py `
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
    --user_model_codejudge $user_model_codejudge `
    --isrefinecode_rules $isrefinecode_rules `
    --max_iter_refinecode_rules $max_iter_refinecode_rules 
