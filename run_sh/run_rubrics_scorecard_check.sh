#!/bin/bash
set -e

# --- parameters ---
# mode: json = scan Rubrics/topic/*.json; task = scan topic/tasks/*.json and check each task's rubrics_name scorecard
mode="task"                                     # json | task
topic="all"                                     # dmrg | nnwf | qcmb | all
rubrics_dir="../Paper_dataset/Rubrics"         # path to Rubrics root (relative to run_sh)
paper_dataset="../Paper_dataset"               # path to Paper_dataset root (used in task mode)

# --- run rubrics scorecard NormCheck ---
python ../src/rubrics_scorecard_check.py \
    --mode "$mode" \
    --topic "$topic" \
    --rubrics_dir "$rubrics_dir" \
    --paper_dataset "$paper_dataset"
