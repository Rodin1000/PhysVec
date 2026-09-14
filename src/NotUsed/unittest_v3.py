# judge_code: verify and grade code
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
        "--repo_dir",
        default="../Output_repo",
        help="Path to the targetted repository directory.",
    )
    parser.add_argument(
        "--CodeVerifier_library_dir",
        default="../CodeVerifier_library_compact",
        help="Path to the CodeVerifier library directory.",
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
        "--upper_num_gencode_unittest",
        type=int,
        default=2,
        help="Number of verify code generation in this unittest.",
    )
    parser.add_argument(
        "--upper_num_exec_unittest",
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
        "--sandbox_dir",
        default="../CodeVerifier_sandbox",
        help="Path to the sandbox directory.",
    )
    parser.add_argument(
        "--concurr_num_unittest",
        type=int,
        default=10,
        help="Number of concurrent unittests.",
    )
    parser.set_defaults(include_unit=True)
    return parser.parse_args()


async def judge_unittest(pdf_name: str, subplot_name: str, repo_dir: str, CodeVerifier_library_dir: str, idname_proj: str, idname_unittest: str, upper_num_gencode_unittest: int, upper_num_exec_unittest: int, isgencode: bool, ismcp: bool, isexec: bool, user_model_codeverifier: str, sandbox_dir: str, concurr_num_unittest: int):

    if not isgencode and not isexec:
        raise ValueError("At least one of isgencode or isexec must be True.")

    # Create log and sandbox directories-------------------------------------------------
    current_unittest_file, _ = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../logs", issubfile=True, iscreatlog=True, headname_dir="", headname_log="log_unittest", idname_dir=idname_proj, idname_log=idname_unittest)
    # Define agents-----------------------------------------------------------------------
    agent_CodeVerifier = QMBagents.CodeVerifier(current_log_file=current_unittest_file)


    # Process: Generate verify code----------------------------------------------------------------
    if isgencode:
        current_unittest_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../CodeVerifier_sandbox", issubfile=True, iscreatlog=False, headname_dir="unittest", idname_dir=idname_unittest)

        # Verify_single_repocode_info = await agent_CodeVerifier.Verify_code_single_repocode_generation(
        #     user_model=user_model_codeverifier,
        #     CodeVerifier_library_dir=CodeVerifier_library_dir,
        #     current_sandbox_dir=current_unittest_dir,
        #     repo_dir=repo_dir,
        #     upper_num=upper_num_gencode_unittest,
        #     ismcp=ismcp,
        # )
        Verify_single_repocode_info = await agent_CodeVerifier.Verify_code_single_repocode_generation_concurr(
            user_model=user_model_codeverifier,
            CodeVerifier_library_dir=CodeVerifier_library_dir,
            current_sandbox_dir=current_unittest_dir,
            repo_dir=repo_dir,
            upper_num=upper_num_gencode_unittest,
            concurr_num=concurr_num_unittest,
        )
        print(f'single repocode verify code generated at {current_unittest_dir}. ')

    # Process: Execute verify code----------------------------------------------------------------
    if isexec and isgencode:
        # Single_repocode_judge_info = agent_CodeVerifier.Verify_code_single_repocode_execution(
        #     user_model=user_model_codeverifier,
        #     current_sandbox_dir=current_unittest_dir,
        #     upper_num=upper_num_exec_unittest,
        # )
        Single_repocode_judge_info = await agent_CodeVerifier.Verify_code_single_repocode_execution_concurr(
            user_model=user_model_codeverifier,
            current_sandbox_dir=current_unittest_dir,
            upper_num=upper_num_exec_unittest,
            concurr_num=concurr_num_unittest,
        )
        print(f'single repocode verify code executed, results are stored in {agent_CodeVerifier.current_log_file}. ')

    elif isexec and not isgencode:
        # Single_repocode_judge_info = agent_CodeVerifier.Verify_code_single_repocode_execution(
        #     user_model=user_model_codeverifier,
        #     current_sandbox_dir=sandbox_dir,
        #     upper_num=upper_num_exec_unittest,
        # )
        Single_repocode_judge_info = await agent_CodeVerifier.Verify_code_single_repocode_execution_concurr(
            user_model=user_model_codeverifier,
            current_sandbox_dir=sandbox_dir,
            upper_num=upper_num_exec_unittest,
            concurr_num=concurr_num_unittest,
        )
        print(f'single repocode verify code executed, results are stored in {agent_CodeVerifier.current_log_file}. ')


async def main():
    args = parse_args()
    task_data = OtherTools.load_task_parameters(args.task_name)

    await judge_unittest(
        pdf_name=task_data.get("pdf_name"),
        subplot_name=task_data.get("subplot_name"),
        repo_dir=args.repo_dir,
        CodeVerifier_library_dir=args.CodeVerifier_library_dir,
        idname_proj=args.idname_proj,
        idname_unittest=args.idname_unittest,
        upper_num_gencode_unittest=args.upper_num_gencode_unittest,
        upper_num_exec_unittest=args.upper_num_exec_unittest,
        isgencode=args.isgencode,
        ismcp=args.ismcp,
        isexec=args.isexec,
        user_model_codeverifier=args.user_model_codeverifier,
        sandbox_dir=args.sandbox_dir,
        concurr_num_unittest=args.concurr_num_unittest,
    )


async def task_unittest(args, repo_dir: str):
    task_data = OtherTools.load_task_parameters(args.task_name)

    await judge_unittest(
        pdf_name=task_data.get("pdf_name"),
        subplot_name=task_data.get("subplot_name"),
        repo_dir=repo_dir, # from external call
        CodeVerifier_library_dir=args.CodeVerifier_library_dir,
        idname_proj=args.idname_proj,
        idname_unittest=args.idname_unittest,
        upper_num_gencode_unittest=args.upper_num_gencode_unittest,
        upper_num_exec_unittest=args.upper_num_exec_unittest,
        isgencode=True, # always generate verify code
        ismcp=False, # do not use MCP in unittest
        isexec=True, # always execute verify code
        user_model_codeverifier=args.user_model_codeverifier,
        sandbox_dir="../CodeVerifier_sandbox", # not used in this case
        concurr_num_unittest=args.concurr_num_unittest,
    )


if __name__ == "__main__":
    asyncio.run(main())

