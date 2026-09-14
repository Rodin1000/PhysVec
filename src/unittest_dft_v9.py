# judge_code: verify and grade code (DFT/ORCA unittest)
# Merged from unittest_dft_v4 (DFT_agents, isgencode/isexec, ORCA) and unittest_dft_v9 (task_unittest, output_dir/sandbox_dir, optional report).
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
from src import DFT_agents, OtherTools


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
        description="Run author program to conduct unittest for an implementation (DFT/ORCA)."
    )
    parser.add_argument(
        "--topic",
        default="dmrg",
        help="Name of the topic to process (e.g., dmrg, nnwf, dft_qc, etc.).",
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
        help="Path to the sandbox directory (used when isexec=True and isgencode=False).",
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
        help="Whether to generate unittest report (if agent supports it).",
    )
    parser.add_argument(
        "--output_dir",
        default="../logs",
        help="Path to the output directory for logs.",
    )
    parser.add_argument(
        "--output_sandbox_dir",
        default="../CodeVerifier_sandbox",
        help="Path to the output directory for sandbox.",
    )
    parser.set_defaults(include_unit=True)
    return parser.parse_args()


async def judge_unittest(
    topic: str,
    pdf_name: str,
    subplot_name: str,
    code_file: str,
    repo_dir: str,
    idname_proj: str,
    idname_unittest: str,
    upper_num_gencode: int,
    upper_num_exec: int,
    isgencode: bool,
    isexec: bool,
    user_model_codeverifier: str,
    judge_model_codeverifier: str,
    sandbox_dir: str,
    concurr_num_unittest: int,
    output_dir: str,
    output_sandbox_dir: str,
):
    if not isgencode and not isexec:
        raise ValueError("At least one of isgencode or isexec must be True.")

    # Create log and sandbox directories (flat log layout like QMB unittest_v9)
    current_unittest_file, _ = DFT_agents.create_run_log(
        pdf_name=pdf_name,
        subplotname=subplot_name,
        output_dir=output_dir,
        topic=None,
        issubfile=False,
        iscreatlog=True,
        headname_dir="",
        headname_log="log_unittest",
        idname_dir=idname_proj,
        idname_log=idname_unittest,
    )
    agent_CodeVerifier = DFT_agents.CodeVerifier(
        topic=topic, current_log_file=current_unittest_file
    )

    Verify_single_repocode_info = None
    Single_repocode_judge_info = None
    Single_repocode_refine_judge_info = None
    current_unittest_dir = None

    # Process: Generate verify code (DFT/ORCA params)
    if isgencode:
        current_unittest_dir = DFT_agents.create_run_log(
            pdf_name=pdf_name,
            subplotname=subplot_name,
            output_dir=output_sandbox_dir,
            topic=topic,
            issubfile=True,
            iscreatlog=False,
            headname_dir="unittest",
            idname_dir=idname_unittest,
        )
        Verify_single_repocode_info = await agent_CodeVerifier.Verify_code_single_repocode_generation_autolibrary_split_concurr(
            user_model=user_model_codeverifier,
            current_sandbox_dir=current_unittest_dir,
            code_file=code_file,
            repo_dir=repo_dir,
            upper_num=upper_num_gencode,
            concurr_num=concurr_num_unittest,
            iscaltoken=True,
        )
        print(f"single repocode verify code generated at {current_unittest_dir}.")

    # Process: Execute verify code
    if isexec and isgencode:
        Single_repocode_judge_info = await agent_CodeVerifier.Verify_code_single_repocode_execution_concurr(
            user_model=judge_model_codeverifier,
            current_sandbox_dir=current_unittest_dir,
            upper_num=upper_num_exec,
            concurr_num=concurr_num_unittest,
            use_mcp_run_orca=False,
            mcp_orca_timeout=60,
            iscaltoken=True,
        )
        Single_repocode_refine_judge_info = await agent_CodeVerifier.Verify_code_single_repocode_split_refine_concurr(
            user_model=judge_model_codeverifier,
            current_sandbox_dir=current_unittest_dir,
            upper_num=0,
            concurr_num=concurr_num_unittest,
            iscaltoken=True,
        )
        print(f"single repocode verify code executed, results are stored in {agent_CodeVerifier.current_log_file}.")

    elif isexec and not isgencode:
        Single_repocode_judge_info = await agent_CodeVerifier.Verify_code_single_repocode_execution_concurr(
            user_model=judge_model_codeverifier,
            current_sandbox_dir=sandbox_dir,
            upper_num=upper_num_exec,
            concurr_num=concurr_num_unittest,
            use_mcp_run_orca=False,
            mcp_orca_timeout=60,
            iscaltoken=True,
        )
        Single_repocode_refine_judge_info = await agent_CodeVerifier.Verify_code_single_repocode_split_refine_concurr(
            user_model=judge_model_codeverifier,
            current_sandbox_dir=sandbox_dir,
            upper_num=0,
            concurr_num=concurr_num_unittest,
            iscaltoken=True,
        )
        print(f"single repocode verify code executed, results are stored in {agent_CodeVerifier.current_log_file}.")

    # Optional: token statistics (when agent returns token_stats)
    gen_token = (Verify_single_repocode_info or {}).get("token_stats") or {}
    exec_token = (Single_repocode_judge_info or {}).get("token_stats") or {}
    refine_token = (Single_repocode_refine_judge_info or {}).get("token_stats") or {}
    judge_total_input = (
        gen_token.get("input_tokens", 0)
        + exec_token.get("input_tokens", 0)
        + refine_token.get("input_tokens", 0)
    )
    judge_total_output = (
        gen_token.get("output_tokens", 0)
        + exec_token.get("output_tokens", 0)
        + refine_token.get("output_tokens", 0)
    )
    judge_total = judge_total_input + judge_total_output
    judge_token_stats = (
        {
            "input_tokens": judge_total_input,
            "output_tokens": judge_total_output,
            "total": judge_total,
        }
        if judge_total > 0
        else None
    )

    # Generate report when agent has Verify_code_generate_report and we ran execution (log has executability judgement)
    report_file = None
    if isexec and hasattr(agent_CodeVerifier, "Verify_code_generate_report"):
        report_info = agent_CodeVerifier.Verify_code_generate_report(
            user_model=judge_model_codeverifier,
            judge_token_stats=judge_token_stats,
        )
        report_file = (report_info or {}).get("report_file")
        if report_file:
            print(f"unittest report generated at {report_file}.")

    return {
        "report_file": report_file,
        "code_file": code_file,
        "repo_dir": repo_dir,
    }


async def main():
    args = parse_args()
    task_data = OtherTools.load_task_parameters(args.topic, args.task_name)
    sandbox_dir = getattr(args, "sandbox_dir", args.output_sandbox_dir)

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
        isexec=args.isexec,
        user_model_codeverifier=args.user_model_codeverifier,
        judge_model_codeverifier=args.judge_model_codeverifier,
        sandbox_dir=sandbox_dir,
        concurr_num_unittest=args.concurr_num_unittest,
        output_dir=args.output_dir,
        output_sandbox_dir=args.output_sandbox_dir,
    )


async def task_unittest(args):
    """Entry point for batch/workflow: same as main but takes an args object and returns result dict."""
    task_data = OtherTools.load_task_parameters(args.topic, args.task_name)
    sandbox_dir = getattr(args, "sandbox_dir", getattr(args, "output_sandbox_dir", "../CodeVerifier_sandbox"))
    output_dir = getattr(args, "output_dir", "../logs")
    output_sandbox_dir = getattr(args, "output_sandbox_dir", "../CodeVerifier_sandbox")
    isgencode = getattr(args, "isgencode", True)
    isexec = getattr(args, "isexec", False)

    result = await judge_unittest(
        topic=args.topic,
        pdf_name=task_data.get("pdf_name"),
        subplot_name=task_data.get("subplot_name"),
        code_file=args.code_file,
        repo_dir=args.repo_dir,
        idname_proj=args.idname_proj,
        idname_unittest=args.idname_unittest,
        upper_num_gencode=args.upper_num_gencode,
        upper_num_exec=args.upper_num_exec,
        isgencode=isgencode,
        isexec=isexec,
        user_model_codeverifier=args.user_model_codeverifier,
        judge_model_codeverifier=args.judge_model_codeverifier,
        sandbox_dir=sandbox_dir,
        concurr_num_unittest=args.concurr_num_unittest,
        output_dir=output_dir,
        output_sandbox_dir=output_sandbox_dir,
    )
    return result


if __name__ == "__main__":
    asyncio.run(main())
