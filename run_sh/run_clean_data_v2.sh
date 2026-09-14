#!/bin/bash
# clean_data_v2: under result_dir/topic, for tasks with no _check or fullcode mismatch,
# remove top-level files and _check/retry* subdirs; remove task dir only if empty.
# Set result_dir and topic_list below; use dry_run="true" to preview only.
set -e

# --- set parameters ---
# tag1="BATCH5_15000token_24_1_24"
# tag1="BATCH5_15000token_18_1_18"
# tag1="BATCH5_10000token_20_1_20"
# tag1="BATCH5_10000token_13_1_13"
# tag1="BATCH5_10000token_22_1_22"
# tag1="BATCH5_15000token_26_1_26"
# tag1="BATCH5_15000token_28_1_28"
tag1="BATCH5_15000token_29_1_29"


result_dir="../results/results_${tag1}"
topic_list="qcmb"
# Set to "true" to only print what would be deleted
dry_run="false"
# Set to "true" to also clean tasks where code_LLM_loop* and report_fullcode_loop* mismatch in _check
isexecode="true"

# --- run clean_data_v2 ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

cmd="python clean_data_v2.py --result_dir \"$result_dir\" --topic_list \"$topic_list\""
[ "$dry_run" = "true" ] && cmd="$cmd --dry_run"
[ "$isexecode" = "true" ] && cmd="$cmd --isexecode"
eval "$cmd"
