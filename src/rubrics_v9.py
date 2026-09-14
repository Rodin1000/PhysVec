# judge_rubrics: grade code by rubrics
import os
import sys
import asyncio
import json
import argparse
import statistics
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
        "--topic",
        default="dmrg",
        help="Name of the topic to process (e.g., dmrg, nnwf, etc.).",
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
    parser.add_argument(
        "--output_dir",
        default="../logs",
        help="Path to the output directory for logs.",
    )
    parser.add_argument(
        "--num_rubrics_grade",
        type=int,
        default=1,
        help="Number of times to run Grade_rubrics_scorecard; mean and std are computed and written to report.",
    )
    parser.add_argument(
        "--concurr_num_limiting",
        type=int,
        default=3,
        help="Concurrency limit for Test_LimitingCases.",
    )
    parser.set_defaults(include_unit=True)
    return parser.parse_args()


async def judge_rubrics(topic: str, pdf_path: str, pdf_name: str, subplot_name: str, rubrics_name: str, code_file: str, idname_proj: str, idname_rubrics: str, user_model_rubricsgrader: str, output_dir: str, LimitingCases=None, num_rubrics_grade: int = 1, concurr_num_limiting: int = 3, max_iter_limiting: int = 3):

    # Create log and rubric directories-------------------------------------------------
    current_rubrics_file, _ = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir=output_dir, topic=None, issubfile=False, iscreatlog=True, headname_dir="", headname_log="log_rubrics", idname_dir=idname_proj, idname_log=idname_rubrics)
    # Define agents-----------------------------------------------------------------------
    agent_RubricsGrader = QMBagents.RubricsGrader(current_log_file=current_rubrics_file, topic=topic)

    # Process: Grade the code content by Human-curated rubrics----------------------------------------------
    scorecard = agent_RubricsGrader.Generate_rubrics_scorecard(
        rubrics_dir=f"../Paper_dataset/Rubrics/{topic}",
        rubrics_name=rubrics_name,
    )
    print(f'Rubrics scorecard generated with NormCheck: {scorecard["NormCheck"]}.')

    if scorecard["NormCheck"] == True:
        grades = []
        run_scorecards = []
        # Temperature increases by 0.15 per run (hardcoded): run 0 -> 0.0, run 1 -> 0.15, run 2 -> 0.3, ...
        for run_idx in range(num_rubrics_grade):
            current_temp = 0.0 + run_idx * 0.15
            result = agent_RubricsGrader.Grade_rubrics_scorecard(
                user_model=user_model_rubricsgrader,
                paper_path=pdf_path,
                paper_name=pdf_name,
                code_file=code_file,
                scorecard=scorecard,
                iscaltoken=True,
                temperature=current_temp,
            )
            result = agent_RubricsGrader.CheckJudgeSupportConsistency(
                user_model=user_model_rubricsgrader,
                scorecard_graded=result,
                scorecard_original=scorecard,
                iscaltoken=True,
            )
            run_scorecards.append(result)
            g = result.get("rubrics_grade")
            if g is not None:
                grades.append(float(g))
        mean = statistics.mean(grades) if grades else 0.0
        std = statistics.stdev(grades) if len(grades) >= 2 else 0.0
        print(f'Rubrics grading completed. Results stored in {agent_RubricsGrader.current_log_file}. ')
        limiting_results = None
        limiting_token_stats = None
        if LimitingCases is not None:
            # limiting_output = await agent_RubricsGrader.Test_LimitingCases(
            #     code_file=code_file,
            #     author_model=user_model_rubricsgrader,
            #     judge_model=user_model_rubricsgrader,
            #     idname_rubrics=idname_rubrics,
            #     LimitingCases=LimitingCases,
            #     max_iter_LC=max_iter_limiting,
            #     concurr_num=concurr_num_limiting,
            #     iscaltoken=True,
            #     scorecard=scorecard,
            # )
            limiting_output = await agent_RubricsGrader.Test_LimitingCases_v2(
                code_file=code_file,
                author_model=user_model_rubricsgrader,
                judge_model=user_model_rubricsgrader,
                idname_rubrics=idname_rubrics,
                LimitingCases=LimitingCases,
                max_iter_LC=max_iter_limiting,
                concurr_num=concurr_num_limiting,
                iscaltoken=True,
                scorecard=scorecard,
            )
            limiting_results = limiting_output.get("results") if isinstance(limiting_output, dict) else limiting_output
            limiting_token_stats = limiting_output.get("token_stats") if isinstance(limiting_output, dict) else None
            print(f'Limiting cases test completed. Results stored in {agent_RubricsGrader.current_log_file}.')
        # Aggregate scorecard token stats across all Grade_rubrics_scorecard runs
        scorecard_token_stats = None
        scorecard_tokens = [s.get("token_stats") for s in run_scorecards if s.get("token_stats")]
        if scorecard_tokens:
            ti = sum(t.get("input_tokens", 0) for t in scorecard_tokens)
            to = sum(t.get("output_tokens", 0) for t in scorecard_tokens)
            if ti + to > 0:
                scorecard_token_stats = {"input_tokens": ti, "output_tokens": to, "total": ti + to}
        report_info = agent_RubricsGrader.Scientific_test_generate_report(
            rubrics_grade_mean=mean,
            rubrics_grade_std=std,
            user_model=user_model_rubricsgrader,
            limiting_results=limiting_results,
            scientific_results=None,
            rubric_run_results=run_scorecards,
            scorecard_token_stats=scorecard_token_stats,
            limiting_token_stats=limiting_token_stats,
            scientific_token_stats=None,
        )
        print(f'Rubrics report generated at {report_info.get("report_file")}. ')



async def main():
    args = parse_args()
    task_data = OtherTools.load_task_parameters(args.topic, args.task_name)

    await judge_rubrics(
        topic=args.topic,
        pdf_path=task_data.get("pdf_path"),
        pdf_name=task_data.get("pdf_name"),
        subplot_name=task_data.get("subplot_name"),
        rubrics_name=task_data.get("rubrics_name"),
        code_file=args.code_file,
        idname_proj=args.idname_proj,
        idname_rubrics=args.idname_rubrics,
        user_model_rubricsgrader=args.user_model_rubricsgrader,
        output_dir=args.output_dir,
        LimitingCases=task_data.get("LimitingCases"),
        num_rubrics_grade=getattr(args, 'num_rubrics_grade', 1),
        concurr_num_limiting=getattr(args, 'concurr_num_limiting', 3),
        max_iter_limiting=getattr(args, 'max_iter_limiting', 3),
    )



async def task_rubrics(args, code_file: str):
    task_data = OtherTools.load_task_parameters(args.topic, args.task_name)

    await judge_rubrics(
        topic=args.topic,
        pdf_path=task_data.get("pdf_path"),
        pdf_name=task_data.get("pdf_name"),
        subplot_name=task_data.get("subplot_name"),
        rubrics_name=task_data.get("rubrics_name"),
        code_file=code_file,
        idname_proj=args.idname_proj,
        idname_rubrics=args.idname_rubrics,
        user_model_rubricsgrader=args.user_model_rubricsgrader,
        output_dir=args.output_dir,
        LimitingCases=task_data.get("LimitingCases"),
        num_rubrics_grade=getattr(args, 'num_rubrics_grade', 1),
        concurr_num_limiting=getattr(args, 'concurr_num_limiting', 3),
        max_iter_limiting=getattr(args, 'max_iter_limiting', 3),
    )



if __name__ == "__main__":
    asyncio.run(main())