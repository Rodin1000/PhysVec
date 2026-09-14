#!/usr/bin/env bash
# corresponding to 'set -e'
set -e

# --- set parameters ---
topic="dft_qc"
task_name="task_1_1.json"
code_file="../logs/dft_qc/_ja5b10819_Fig._8_panel_a_test2/code_LLM.inp"  # specify the code file to use for automatic library selection
repo_dir="../Output_repo/dft_qc/repo_ja5b10819_Fig._8_panel_a_test2/"    # specify the repository directory needed to be tested
idname_proj="test2"      # this name will be used to create a project folder under ../logs/, or you can choose an already existing one (as used in run_author.ps1)
idname_unittest="test2"
upper_num_gencode=0      # 0 means all
upper_num_exec=0         # 0 means all
isgencode=true           # enable code generation to create verify code files
isexec=true              # enable execution to test ORCA

user_model_codeverifier="gpt-5.1-chat"  # Juge model must use gpt5.2
judge_model_codeverifier="gpt-5.1-chat" # Juge model must use gpt5.2

sandbox_dir="../CodeVerifier_sandbox/dft_qc/unittest_ja5b10819_Fig._8_panel_a_test2"  # needed if isgencode=false while isexec=true (will be auto-generated if isgencode=true)
concurr_num_unittest=10

# --- generate code and repository ---
python ../src/unittest_dft_v4.py \
  --topic "$topic" \
  --task_name "$task_name" \
  --code_file "$code_file" \
  --repo_dir "$repo_dir" \
  --idname_proj "$idname_proj" \
  --idname_unittest "$idname_unittest" \
  --upper_num_gencode "$upper_num_gencode" \
  --upper_num_exec "$upper_num_exec" \
  --isgencode "$isgencode" \
  --isexec "$isexec" \
  --user_model_codeverifier "$user_model_codeverifier" \
  --judge_model_codeverifier "$judge_model_codeverifier" \
  --sandbox_dir "$sandbox_dir" \
  --concurr_num_unittest "$concurr_num_unittest"
