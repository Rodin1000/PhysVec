# task_workflow_programtest: batch execution of unittest_v7 and integtest_v7
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
from src import unittest_v7, integtest_v7


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
        description="Run batch program test workflow to execute unittest and integtest for multiple tasks."
    )
    parser.add_argument(
        "--topic",
        default="dmrg",
        help="Topic of the paper to process (e.g., dmrg, nnwf, etc.).",
    )
    parser.add_argument(
        "--result_dir",
        default="../results",
        help="Result directory containing task subdirectories.",
    )
    parser.add_argument(
        "--isall",
        type=str2bool,
        default=False,
        help="Whether to execute all tasks in the result_dir/topic directory.",
    )
    parser.add_argument(
        "--task_list",
        default="",
        help="Comma-separated list of task names (without .json extension) to execute when isall is False.",
    )
    parser.add_argument(
        "--user_model_codeverifier",
        default="deepseek/deepseek-v3.1-terminus",
        help="Model identifier for the unittest code verifier agent.",
    )
    parser.add_argument(
        "--judge_model_codeverifier",
        default="deepseek/deepseek-v3.1-terminus",
        help="Model identifier for the unittest judge agent.",
    )
    parser.add_argument(
        "--user_model_codeintegrator",
        default="deepseek/deepseek-v3.1-terminus",
        help="Model identifier for the integtest code integrator agent.",
    )
    parser.add_argument(
        "--judge_model_codeintegrator",
        default="deepseek/deepseek-v3.1-terminus",
        help="Model identifier for the integtest judge agent.",
    )
    parser.add_argument(
        "--output_sandbox_dir_unittest",
        default="../CodeVerifier_sandbox",
        help="Path to the output directory for unittest sandbox.",
    )
    parser.add_argument(
        "--output_sandbox_dir_integtest",
        default="../CodeIntegrator_sandbox",
        help="Path to the output directory for integtest sandbox.",
    )
    parser.add_argument(
        "--idname_run",
        default="untagged",
        help="Identifier for the test run (used for both unittest and integtest).",
    )
    parser.set_defaults(include_unit=True)
    return parser.parse_args()


def get_task_list(result_dir: str, topic: str, isall: bool, task_list: str) -> list[str]:
    """Get list of task names to execute."""
    tasks_dir = Path(result_dir) / topic
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")
    
    if isall:
        # Get all subdirectories
        task_dirs = [d for d in tasks_dir.iterdir() if d.is_dir()]
        return [d.name for d in task_dirs]
    else:
        # Parse task_list
        if not task_list:
            raise ValueError("task_list must be provided when isall is False")
        task_names = [t.strip() for t in task_list.split(",")]
        valid_tasks = []
        for task_name in task_names:
            task_path = tasks_dir / task_name
            if task_path.exists() and task_path.is_dir():
                valid_tasks.append(task_name)
            else:
                print(f"Warning: Task directory not found: {task_path}")
        return valid_tasks


def load_statistics_author(task_dir: Path) -> dict | None:
    """Load statistics_author.jsonl file from task directory."""
    statistics_file = task_dir / "statistics_author.jsonl"
    if not statistics_file.exists():
        return None
    
    try:
        content = statistics_file.read_text(encoding="utf-8").strip()
        if not content:
            return None
        data = json.loads(content, strict=False)
        return data
    except Exception as e:
        print(f"Warning: Failed to read statistics_author.jsonl from {task_dir}: {e}")
        return None


def create_unittest_args(base_args, task_name: str, code_file: str, repo_dir: str, pdf_name: str, subplot_name: str, idname_run: str, output_dir_unittest: str, output_sandbox_dir_unittest: str) -> argparse.Namespace:
    """Create arguments for task_unittest function."""
    args = argparse.Namespace()
    # Copy base arguments
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    # Set unittest-specific parameters
    args.task_name = task_name
    args.code_file = code_file
    args.repo_dir = repo_dir
    args.idname_proj = None
    args.idname_unittest = idname_run
    args.upper_num_gencode = 0
    args.upper_num_exec = 0
    args.concurr_num_unittest = 8
    args.output_dir = output_dir_unittest
    args.output_sandbox_dir = output_sandbox_dir_unittest
    return args


def create_integtest_args(base_args, task_name: str, repo_dir: str, pdf_name: str, subplot_name: str, idname_run: str, output_dir_integtest: str, output_sandbox_dir_integtest: str) -> argparse.Namespace:
    """Create arguments for task_integtest function."""
    args = argparse.Namespace()
    # Copy base arguments
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    # Set integtest-specific parameters
    args.task_name = task_name
    args.repo_dir = repo_dir
    args.idname_proj = None
    args.idname_integtest = idname_run
    args.upper_num_integcode = 0
    args.level_num = 0
    args.upper_num_execode = 0
    args.concurr_num_integtest = 8
    args.output_dir = output_dir_integtest
    args.output_sandbox_dir = output_sandbox_dir_integtest
    return args


async def main():
    args = parse_args()
    
    # Get task list
    task_list = get_task_list(args.result_dir, args.topic, args.isall, args.task_list)
    print(f"Found {len(task_list)} task(s) to process: {task_list}")
    
    # Process each task
    for task_name in task_list:
        print(f"\n{'='*60}")
        print(f"Processing task: {task_name}")
        print(f"{'='*60}")
        
        # Build task directory path
        task_dir = Path(args.result_dir) / args.topic / task_name
        
        # Load statistics_author.jsonl
        statistics = load_statistics_author(task_dir)
        if not statistics:
            print(f"Skipping task {task_name}: statistics_author.jsonl not found or invalid")
            continue
        
        flag_obey = statistics.get("flag_obey", False)
        if not flag_obey:
            print(f"Skipping task {task_name}: flag_obey is False (no suitable code for testing)")
            continue
        
        # Extract required information
        code_file = statistics.get("code_file")
        repo_dir = statistics.get("repo_dir")
        pdf_name = statistics.get("pdf_name")
        subplot_name = statistics.get("subplot_name")
        task_name_json = statistics.get("task_name", f"{task_name}.json")
        
        if not code_file or not repo_dir:
            print(f"Skipping task {task_name}: code_file or repo_dir is missing in statistics_author.jsonl")
            continue
        
        # Extract output_dir from code_file path (parent directory of code_file)
        code_file_path = Path(code_file)
        output_dir_check = str(code_file_path.parent)
        
        # Execute unittest
        print(f"\n--- Running unittest for task {task_name} ---")
        try:
            unittest_args = create_unittest_args(
                base_args=args,
                task_name=task_name_json,
                code_file=code_file,
                repo_dir=repo_dir,
                pdf_name=pdf_name,
                subplot_name=subplot_name,
                idname_run=args.idname_run,
                output_dir_unittest=output_dir_check,
                output_sandbox_dir_unittest=args.output_sandbox_dir_unittest
            )
            unittest_result = await unittest_v7.task_unittest(unittest_args)
            print(f"Unittest completed for task {task_name}")
        except Exception as e:
            print(f"Error in unittest for task {task_name}: {e}")
            continue
        
        # Execute integtest
        print(f"\n--- Running integtest for task {task_name} ---")
        try:
            integtest_args = create_integtest_args(
                base_args=args,
                task_name=task_name_json,
                repo_dir=repo_dir,
                pdf_name=pdf_name,
                subplot_name=subplot_name,
                idname_run=args.idname_run,
                output_dir_integtest=output_dir_check,
                output_sandbox_dir_integtest=args.output_sandbox_dir_integtest
            )
            integtest_result = await integtest_v7.task_integtest(integtest_args)
            print(f"Integtest completed for task {task_name}")
        except Exception as e:
            print(f"Error in integtest for task {task_name}: {e}")
            continue
        
        print(f"\nTask {task_name} completed successfully.")
    
    print(f"\n{'='*60}")
    print(f"All tasks completed. Processed {len(task_list)} task(s).")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

