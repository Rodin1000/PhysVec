#!/usr/bin/env bash
set -euo pipefail

# --- set parameters ---
topic="dft_qc"                 # specify the topic (e.g., nnwf, dmrg)
type_name="XAS_ligand_Rel"         # specify the type name
idname_proj="$topic"         # project folder name under ../logs/
idname_codeverifier="$type_name"  # identifier for codeverifier logs

user_model="gemini-3-flash-preview-thinking"   # LLM model for all steps
concurr_num=10          # number of concurrent tasks for Prepare and Generate steps
query_num=3            # number of queries for Prepare step
num_check=3             # number of check iterations for Generate step
max_iterations=10       # maximum iterations for Refine step
program_timeout=15     # timeout for ORCA input validation (60s is enough to check if input is valid)
refine_query_num=10      # number of queries for Refine2 step (RAG-based refinement)
min_refinement_iterations=1   # minimum iterations before allowing success (set to 2+ to force extra passes)
do_cold_init=true       # whether to execute Cold_init step
do_prepare=true         # whether to execute Prepare step
do_generate=true        # whether to execute Generate step
do_combine=true         # whether to execute Combine step
do_refine=true          # whether to execute Refine step

# --- run codeverifier workflow ---
python ../src/codeverifier_v1_dft.py \
  --topic "$topic" \
  --type_name "$type_name" \
  --idname_proj "$idname_proj" \
  --idname_codeverifier "$idname_codeverifier" \
  --user_model "$user_model" \
  --concurr_num "$concurr_num" \
  --query_num "$query_num" \
  --num_check "$num_check" \
  --max_iterations "$max_iterations" \
  --program_timeout "$program_timeout" \
  --refine_query_num "$refine_query_num" \
  --min_refinement_iterations "$min_refinement_iterations" \
  --do_cold_init "$do_cold_init" \
  --do_prepare "$do_prepare" \
  --do_generate "$do_generate" \
  --do_combine "$do_combine" \
  --do_refine "$do_refine"
