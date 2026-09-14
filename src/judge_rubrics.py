# judge_rubrics: grade code by rubrics
import os
import sys
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Import from this project
from src import QMBagents



async def judge_rubrics(pdf_path: str, pdf_name: str, subplot_name: str, rubrics_name: str, code_path: str, code_name: str, idname: str, user_model_rubricsgrader: str, log_dir: str):

    # Create log and rubric directories-------------------------------------------------
    current_log_file, current_log_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir=log_dir, issubfile=True, iscreatlog=True, headname="judge", isidname=True, idname=idname)
    current_rubrics_log_file, current_rubrics_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../RubricsGrader", issubfile=True, iscreatlog=True, headname="rubrics", isidname=True, idname=idname)

    # Define agents-----------------------------------------------------------------------
    agent_RubricsGrader = QMBagents.RubricsGrader(current_log_file=current_log_file)

    # Process: Grade the code content by Human-curated rubrics----------------------------------------------
    scorecard = agent_RubricsGrader.Generate_rubrics_scorecard(
        rubrics_dir=pdf_path,
        rubrics_name=rubrics_name,
    )
    print(f'Rubrics scorecard generated with NormCheck: {scorecard["NormCheck"]}.')

    result = agent_RubricsGrader.Grade_rubrics_scorecard(
        user_model=user_model_rubricsgrader,
        paper_path=pdf_path,
        paper_name=pdf_name,
        code_path=code_path,
        code_name=code_name,
        scorecard=scorecard,
    )
    print(f'Rubrics grading completed. Results stored in {agent_RubricsGrader.current_log_file}. ')




