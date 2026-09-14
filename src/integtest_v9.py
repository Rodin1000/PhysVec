# judge_code: verify and grade code
# This program is copied from integtest_v2.5.py, only focus on the integration test part.
import os
import sys
import asyncio
import json
import argparse
import shutil
import traceback
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
        help="Name of the topic to process (e.g., dmrg, tdmrg).",
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
        "--idname_proj",
        default="untagged",
        help="Identifier for the project.",
    )
    parser.add_argument(
        "--idname_integtest",
        default="untagged",
        help="Identifier for the integration test.",
    )
    parser.add_argument(
        "--upper_num_integcode",
        type=int,
        default=2,
        help="Number of run functions to be tested in this integtest.",
    )
    parser.add_argument(
        "--level_num",
        type=int,
        default=2,
        help="Number of levels to be re-constructed for each run function.",
    )
    parser.add_argument(
        "--upper_num_execode",
        type=int,
        default=2,
        help="Number of code executions for each run function in this integtest.",
    )
    parser.add_argument(
        "--isintegcode",
        type=str2bool,
        default=True,
        help="Whether to generate integcodes.",
    )
    parser.add_argument(
        "--isexecode",
        type=str2bool,
        default=False,
        help="Whether to execute the integcodes.",
    )
    parser.add_argument(
        "--user_model_codeintegrator",
        default="deepseek/deepseek-v3.1-terminus",
        help="User model for the code integrator.",
    )
    parser.add_argument(
        "--judge_model_codeintegrator",
        default="deepseek/deepseek-v3.1-terminus",
        help="Judge model for evaluating integration code execution results.",
    )
    parser.add_argument(
        "--sandbox_dir",
        default="../CodeIntegrator_sandbox",
        help="Path to the sandbox directory.",
    )
    parser.add_argument(
        "--concurr_num_integtest",
        type=int,
        default=5,
        help="Number of concurrent integration tests to run.",
    )
    parser.add_argument(
        "--isreport",
        type=str2bool,
        default=False,
        help="Whether to generate integration test report.",
    )
    parser.add_argument(
        "--output_dir",
        default="../logs",
        help="Path to the output directory for logs.",
    )
    parser.add_argument(
        "--output_sandbox_dir",
        default="../CodeIntegrator_sandbox",
        help="Path to the output directory for sandbox.",
    )
    parser.set_defaults(include_unit=True)
    return parser.parse_args()


async def judge_integtest(topic: str, pdf_name: str, subplot_name: str, repo_dir: str, idname_proj: str, idname_integtest: str, upper_num_integcode: int, level_num: int, upper_num_execode: int, user_model_codeintegrator: str, judge_model_codeintegrator: str, concurr_num_integtest: int, output_dir: str, output_sandbox_dir: str):

    # Create log and sandbox directories-------------------------------------------------
    current_integtest_file, _ = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir=output_dir, topic=None, issubfile=False, iscreatlog=True, headname_dir="", headname_log="log_integtest", idname_dir=idname_proj, idname_log=idname_integtest)
    # Define agents-----------------------------------------------------------------------
    agent_CodeIntegrator = QMBagents.CodeIntegrator(topic=topic, current_log_file=current_integtest_file)

    # Generate integration code----------------------------------------------------------
    current_integtest_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir=output_sandbox_dir, topic=topic, issubfile=True, iscreatlog=False, headname_dir="integtest", idname_dir=idname_integtest)
    _p = Path(current_integtest_dir)
    if _p.exists():
        shutil.rmtree(_p)
        _p.mkdir(parents=True, exist_ok=True)

    # generate guidelines
    guideline_info = await agent_CodeIntegrator.Generate_integration_guidelines(
        user_model=user_model_codeintegrator,
        current_sandbox_dir=current_integtest_dir,
        repo_dir=repo_dir,
        upper_num=upper_num_integcode,
        iscaltoken=True,
    )
    print(f'integration guidelines generated at {current_integtest_dir}. ')

    #generate integration codes
    try:
        integration_code_info = await agent_CodeIntegrator.Generate_integration_code_split_concurr(
            user_model=user_model_codeintegrator,
            current_sandbox_dir=current_integtest_dir,
            repo_dir=repo_dir,
            upper_num=level_num,
            concurr_num=concurr_num_integtest,
            iscaltoken=True,
        )
    except Exception as e:
        agent_CodeIntegrator._log_event({
            "Content": "Integration code generation failed",
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        raise
    print(f'integration code generated at {current_integtest_dir}. ')

    # Execute integration code and evaluate----------------------------------------------
    integration_execution_output = await agent_CodeIntegrator.Integrate_code_single_run_execution_concurr(
        user_model=judge_model_codeintegrator,
        current_sandbox_dir=current_integtest_dir,
        upper_num=upper_num_execode,
        concurr_num=concurr_num_integtest,
        iscaltoken=True,
    )
    print(f'integration code executed and evaluated at {current_integtest_dir}. ')

    # Collect token statistics
    guideline_token = guideline_info.get("token_stats") or {}
    code_token = integration_code_info.get("token_stats") or {}
    
    # Aggregate execution token from all run functions
    exec_total_input = 0
    exec_total_output = 0
    if isinstance(integration_execution_output, list):
        for run_output in integration_execution_output:
            run_token = run_output.get("token_stats") or {}
            exec_total_input += run_token.get("input_tokens", 0)
            exec_total_output += run_token.get("output_tokens", 0)
    
    judge_total_input = (
        guideline_token.get("input_tokens", 0) +
        code_token.get("input_tokens", 0) +
        exec_total_input
    )
    judge_total_output = (
        guideline_token.get("output_tokens", 0) +
        code_token.get("output_tokens", 0) +
        exec_total_output
    )
    judge_total = judge_total_input + judge_total_output
    
    judge_token_stats = {
        "input_tokens": judge_total_input,
        "output_tokens": judge_total_output,
        "total": judge_total
    } if judge_total > 0 else None

    # Generate report----------------------------------------------------------------------
    report_info = agent_CodeIntegrator.Integrate_code_generate_report(
        user_model=judge_model_codeintegrator,
        current_sandbox_dir=current_integtest_dir,
        judge_token_stats=judge_token_stats
    )
    report_file = report_info.get("report_file")
    print(f'integration test report generated at {report_file}. ')
    
    # Return result dictionary
    return {
        "report_file": report_file,
        "repo_dir": repo_dir
    }


async def main():
    args = parse_args()
    task_data = OtherTools.load_task_parameters(args.topic, args.task_name)

    await judge_integtest(
        topic=args.topic,
        pdf_name=task_data.get("pdf_name"),
        subplot_name=task_data.get("subplot_name"),
        repo_dir=args.repo_dir,
        idname_proj=args.idname_proj,
        idname_integtest=args.idname_integtest,
        upper_num_integcode=args.upper_num_integcode,
        level_num=args.level_num,
        upper_num_execode=args.upper_num_execode,
        user_model_codeintegrator=args.user_model_codeintegrator,
        judge_model_codeintegrator=args.judge_model_codeintegrator,
        concurr_num_integtest=args.concurr_num_integtest,
        output_dir=args.output_dir,
        output_sandbox_dir=args.output_sandbox_dir,
    )


async def task_integtest(args):
    task_data = OtherTools.load_task_parameters(args.topic, args.task_name)

    result = await judge_integtest(
        topic=args.topic,
        pdf_name=task_data.get("pdf_name"),
        subplot_name=task_data.get("subplot_name"),
        repo_dir=args.repo_dir,
        idname_proj=args.idname_proj,
        idname_integtest=args.idname_integtest,
        upper_num_integcode=args.upper_num_integcode,
        level_num=args.level_num,
        upper_num_execode=args.upper_num_execode,
        user_model_codeintegrator=args.user_model_codeintegrator,
        judge_model_codeintegrator=args.judge_model_codeintegrator,
        concurr_num_integtest=args.concurr_num_integtest,
        output_dir=args.output_dir,
        output_sandbox_dir=args.output_sandbox_dir,
    )
    
    return result

if __name__ == "__main__":
    asyncio.run(main())