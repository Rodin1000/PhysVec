#!/usr/bin/env bash
# corresponding to 'set -e'
set -e

# --- set parameters ---
topic="dmrg"
task_name="task_2_1.json"
code_file="../logs/dmrg/_PhysRevLett.95.240404_Fig._1_a_test2/code_LLM.jl"  # specify the code file to use for automatic library selection
repo_dir="../Output_repo/dmrg/repo_PhysRevLett.95.240404_Fig._1_a_test2"    # specify the repository directory needed to be tested
idname_proj="test2"      # this name will be used to create a project folder under ../logs/, or you can choose an already existing one (as used in run_author.ps1)
idname_unittest="test2"
upper_num_gencode=0      # 0 means all
upper_num_exec=0         # 0 means all
isgencode=true
ismcp=false              # set false always in this version
isexec=true
user_model_codeverifier="openai/gpt-5"
judge_model_codeverifier="openai/gpt-5"
sandbox_dir="../CodeVerifier_sandbox/dmrg/unittest_PhysRevLett.73.882_Fig._1_test2"  # needed if isgencode=false while isexec=true
concurr_num_unittest=10

# --- generate code and repository ---
python ../src/unittest_v4.py \
  --topic "$topic" \
  --task_name "$task_name" \
  --code_file "$code_file" \
  --repo_dir "$repo_dir" \
  --idname_proj "$idname_proj" \
  --idname_unittest "$idname_unittest" \
  --upper_num_gencode "$upper_num_gencode" \
  --upper_num_exec "$upper_num_exec" \
  --isgencode "$isgencode" \
  --ismcp "$ismcp" \
  --isexec "$isexec" \
  --user_model_codeverifier "$user_model_codeverifier" \
  --judge_model_codeverifier "$judge_model_codeverifier" \
  --sandbox_dir "$sandbox_dir" \
  --concurr_num_unittest "$concurr_num_unittest"
