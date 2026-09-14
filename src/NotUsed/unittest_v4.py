# judge_code: verify and grade code
# copy from unittest_v2.5, only cover the 'unittest' step.
# In this version, we utilize the automatic-generated verifier code library
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
        default="",
        help="Path to the code file to use for repository generation.",
    )
    parser.add_argument(
        "--repo_dir",
        default="../Output_repo",
        help="Path to the targetted repository directory.",
    )
    parser.add_argument(
        "--idname_proj",
        default="untagged",
        help="Identifier for the project.",
    )
    parser.add_argument(
        "--idname_unittest",
        default="untagged",
        help="Identifier for the unittest.",
    )
    parser.add_argument(
        "--upper_num_gencode",
        type=int,
        default=2,
        help="Number of verify code generation in this unittest.",
    )
    parser.add_argument(
        "--upper_num_exec",
        type=int,
        default=2,
        help="Number of executions in this unittest.",
    )
    parser.add_argument(
        "--isgencode",
        type=str2bool,
        default=True,
        help="Whether to generate verify codes.",
    )
    parser.add_argument(
        "--ismcp",
        type=str2bool,
        default=False,
        help="Whether to use MCP method in verify code generation.",
    )
    parser.add_argument(
        "--isexec",
        type=str2bool,
        default=False,
        help="Whether to execute verify codes.",
    )
    parser.add_argument(
        "--user_model_codeverifier",
        default="deepseek/deepseek-v3.1-terminus",
        help="LLM for this unittest workflow.",
    )
    parser.add_argument(
        "--judge_model_codeverifier",
        default="deepseek/deepseek-v3.1-terminus",
        help="LLM for judging code execution results.",
    )
    parser.add_argument(
        "--sandbox_dir",
        default="../CodeVerifier_sandbox",
        help="Path to the sandbox directory.",
    )
    parser.add_argument(
        "--concurr_num_unittest",
        type=int,
        default=5,
        help="Number of concurrent code generation in this unittest.",
    )
    parser.add_argument(
        "--isreport",
        type=str2bool,
        default=False,
        help="Whether to generate unittest report.",
    )
    parser.set_defaults(include_unit=True)
    return parser.parse_args()


async def judge_unittest(topic: str, pdf_name: str, subplot_name: str, code_file: str, repo_dir: str, idname_proj: str, idname_unittest: str, upper_num_gencode: int, upper_num_exec: int, isgencode: bool, ismcp: bool, isexec: bool, user_model_codeverifier: str, judge_model_codeverifier: str, sandbox_dir: str, concurr_num_unittest: int, isreport: bool):

    if not isgencode and not isexec and not isreport:
        raise ValueError("At least one of isgencode or isexec or isreport must be True.")

    # Create log and sandbox directories-------------------------------------------------
    current_unittest_file, _ = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../logs", topic=topic, issubfile=True, iscreatlog=True, headname_dir="", headname_log="log_unittest", idname_dir=idname_proj, idname_log=idname_unittest)
    # Define agents-----------------------------------------------------------------------
    agent_CodeVerifier = QMBagents.CodeVerifier(topic=topic, current_log_file=current_unittest_file)


    # Process: Generate verify code----------------------------------------------------------------
    if isgencode:
        current_unittest_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../CodeVerifier_sandbox", topic=topic, issubfile=True, iscreatlog=False, headname_dir="unittest", idname_dir=idname_unittest)

        Verify_single_repocode_info = await agent_CodeVerifier.Verify_code_single_repocode_generation_autolibrary_split_concurr(
            user_model=user_model_codeverifier,
            current_sandbox_dir=current_unittest_dir,
            code_file=code_file,
            repo_dir=repo_dir,
            upper_num=upper_num_gencode,
            concurr_num=concurr_num_unittest,
        )
        print(f'single repocode verify code generated at {current_unittest_dir}. ')

    # Process: Execute verify code----------------------------------------------------------------
    if isexec and isgencode:
        Single_repocode_judge_info = await agent_CodeVerifier.Verify_code_single_repocode_execution_concurr(
            user_model=judge_model_codeverifier,
            current_sandbox_dir=current_unittest_dir,
            upper_num=upper_num_exec,
            concurr_num=concurr_num_unittest,
        )
        Single_repocode_refine_judge_info= await agent_CodeVerifier.Verify_code_single_repocode_split_refine_concurr(
            user_model=judge_model_codeverifier,
            current_sandbox_dir=current_unittest_dir,
            upper_num=0,
            concurr_num=concurr_num_unittest,
        )
        print(f'single repocode verify code executed, results are stored in {agent_CodeVerifier.current_log_file}. ')

    elif isexec and not isgencode:
        Single_repocode_judge_info = await agent_CodeVerifier.Verify_code_single_repocode_execution_concurr(
            user_model=judge_model_codeverifier,
            current_sandbox_dir=sandbox_dir,
            upper_num=upper_num_exec,
            concurr_num=concurr_num_unittest,
        )
        Single_repocode_refine_judge_info = await agent_CodeVerifier.Verify_code_single_repocode_split_refine_concurr(
            user_model=judge_model_codeverifier,
            current_sandbox_dir=sandbox_dir,
            upper_num=0,
            concurr_num=concurr_num_unittest,
        )
        print(f'single repocode verify code executed, results are stored in {agent_CodeVerifier.current_log_file}. ')

    # Process: Generate report----------------------------------------------------------------
    if isreport:
        report_info = agent_CodeVerifier.Verify_code_generate_report(
            user_model=judge_model_codeverifier
        )
        print(f'unittest report generated at {report_info.get("report_file")}. ')


async def main():
    args = parse_args()
    task_data = OtherTools.load_task_parameters(args.topic, args.task_name)

    await judge_unittest(
        topic=args.topic,
        pdf_name=task_data.get("pdf_name"),
        subplot_name=task_data.get("subplot_name"),
        code_file=args.code_file,
        repo_dir=args.repo_dir,
        idname_proj=args.idname_proj,
        idname_unittest=args.idname_unittest,
        upper_num_gencode=args.upper_num_gencode,
        upper_num_exec=args.upper_num_exec,
        isgencode=args.isgencode,
        ismcp=args.ismcp,
        isexec=args.isexec,
        user_model_codeverifier=args.user_model_codeverifier,
        judge_model_codeverifier=args.judge_model_codeverifier,
        sandbox_dir=args.sandbox_dir,
        concurr_num_unittest=args.concurr_num_unittest,
        isreport=args.isreport,
    )

if __name__ == "__main__":
    asyncio.run(main())

