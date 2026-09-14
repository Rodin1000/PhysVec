#!/usr/bin/env bash
# corresponding to 'set -e'
set -e

# --- set parameters ---
topic="dmrg"
task_name="task_1_1.json"
repo_dir="../Output_repo/dmrg/repo_PhysRevB.58.R14741_Fig._1_devtest"  # specify the repository directory needed to be tested
CodeVerifier_library_dir="../CodeVerifier_library_compact/${topic}"    # No need to change this
idname_proj="devtest"      # this name will be used to create a project folder under ../logs/, or you can choose an already existing one (as used in run_author.ps1)
idname_unittest="devtest"
upper_num_gencode=0        # How many element functions to be tested; 0 means all
upper_num_exec=0           # How many run functions to be executed; 0 means all
isgencode=true             # whether to generate verify code for this unittest
ismcp=false                # in this version, set false always
isexec=true                # whether to execute the generated verify code in this run
user_model_codeverifier="deepseek/deepseek-v3.1-terminus"  # the model used for code verification (LLM judge)
sandbox_dir="../CodeVerifier_sandbox/dmrg/unittest_PhysRevB.58.R14741_Fig._1_devtest"  # You must specify a sandbox containing all the verify codes if isgencode=false while isexec=true
concurr_num_unittest=10    # number of concurrency

# --- generate code and repository ---
python ../src/unittest_v2_5.py \
  --topic "$topic" \
  --task_name "$task_name" \
  --repo_dir "$repo_dir" \
  --CodeVerifier_library_dir "$CodeVerifier_library_dir" \
  --idname_proj "$idname_proj" \
  --idname_unittest "$idname_unittest" \
  --upper_num_gencode "$upper_num_gencode" \
  --upper_num_exec "$upper_num_exec" \
  --isgencode "$isgencode" \
  --ismcp "$ismcp" \
  --isexec "$isexec" \
  --user_model_codeverifier "$user_model_codeverifier" \
  --sandbox_dir "$sandbox_dir" \
  --concurr_num_unittest "$concurr_num_unittest"
