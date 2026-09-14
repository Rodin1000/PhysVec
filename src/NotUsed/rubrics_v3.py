# judge_rubrics: grade code by rubrics
import os
import sys
import asyncio
import json
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Import from this project
from src import QMBagents, OtherTools


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = v.lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean like true/false/1/0/yes/no")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run author program to conduct unittest for an implementation."
    )
    parser.add_argument(
        "--task_name",
        default="task_1_1.json",
        help="Name of the task to process (a figure or table in a specific paper).",
    )
    parser.add_argument(
        "--code_file",
        help="Path to the code file to be graded.",
    )
    parser.add_argument(
        "--idname_proj",
        default="untagged",
        help="Identifier for the project.",
    )
    parser.add_argument(
        "--idname_rubrics",
        default="untagged",
        help="Identifier for the rubrics.",
    )
    parser.add_argument(
        "--user_model_rubricsgrader",
        default="deepseek/deepseek-v3.1-terminus",
        help="User model for the rubrics grader.",
    )
    parser.set_defaults(include_unit=True)
    return parser.parse_args()


async def judge_rubrics(pdf_path: str, pdf_name: str, subplot_name: str, rubrics_name: str, code_file: str, idname_proj: str, idname_rubrics: str, user_model_rubricsgrader: str):

    # Create log and rubric directories-------------------------------------------------
    current_rubrics_file, _ = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../logs", issubfile=True, iscreatlog=True, headname_dir="", headname_log="log_rubrics", idname_dir=idname_proj, idname_log=idname_rubrics)
    # Define agents-----------------------------------------------------------------------
    agent_RubricsGrader = QMBagents.RubricsGrader(current_log_file=current_rubrics_file)

    # Process: Grade the code content by Human-curated rubrics----------------------------------------------
    scorecard = agent_RubricsGrader.Generate_rubrics_scorecard(
        rubrics_dir="../Paper_dataset/Rubrics",
        rubrics_name=rubrics_name,
    )
    print(f'Rubrics scorecard generated with NormCheck: {scorecard["NormCheck"]}.')

    result = agent_RubricsGrader.Grade_rubrics_scorecard(
        user_model=user_model_rubricsgrader,
        paper_path=pdf_path,
        paper_name=pdf_name,
        code_file=code_file,
        scorecard=scorecard,
    )
    print(f'Rubrics grading completed. Results stored in {agent_RubricsGrader.current_log_file}. ')



async def main():
    args = parse_args()
    task_data = OtherTools.load_task_parameters(args.task_name)

    await judge_rubrics(
        pdf_path=task_data.get("pdf_path"),
        pdf_name=task_data.get("pdf_name"),
        subplot_name=task_data.get("subplot_name"),
        rubrics_name=task_data.get("rubrics_name"),
        code_file=args.code_file,
        idname_proj=args.idname_proj,
        idname_rubrics=args.idname_rubrics,
        user_model_rubricsgrader=args.user_model_rubricsgrader,
    )



async def task_rubrics(args, code_file: str):
    task_data = OtherTools.load_task_parameters(args.task_name)

    await judge_rubrics(
        pdf_path=task_data.get("pdf_path"),
        pdf_name=task_data.get("pdf_name"),
        subplot_name=task_data.get("subplot_name"),
        rubrics_name=task_data.get("rubrics_name"),
        code_file=code_file, # from external input
        idname_proj=args.idname_proj,
        idname_rubrics=args.idname_rubrics,
        user_model_rubricsgrader=args.user_model_rubricsgrader,
    )



if __name__ == "__main__":
    asyncio.run(main())