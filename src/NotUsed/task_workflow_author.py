# task_workflow_author: batch execution of author_v5
import os
import sys
import asyncio
import json
import argparse
import re
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Import from this project
from src import QMBagents, OtherTools, author_v5


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
        description="Run batch author workflow to generate code and repository for multiple tasks."
    )
    parser.add_argument(
        "--topic",
        default="dmrg",
        help="Topic of the paper to process (e.g., dmrg, nnwf, etc.).",
    )
    parser.add_argument(
        "--result_dir",
        default="../logs",
        help="Result directory for storing outputs.",
    )
    parser.add_argument(
        "--output_repo_dir",
        default="../Output_repo",
        help="Output directory for repositories.",
    )
    parser.add_argument(
        "--isall",
        type=str2bool,
        default=False,
        help="Whether to execute all tasks in the tasks directory.",
    )
    parser.add_argument(
        "--task_list",
        default="",
        help="Comma-separated list of task names (without .json extension) to execute when isall is False.",
    )
    parser.add_argument(
        "--user_model_planner",
        default="deepseek/deepseek-v3.1-terminus",
        help="Model identifier for the planner agent.",
    )
    parser.add_argument(
        "--user_model_coder",
        default="deepseek/deepseek-chat-v3.1",
        help="Model identifier for the coder agent.",
    )
    parser.add_argument(
        "--user_model_repogenerator",
        default="deepseek/deepseek-v3.1-terminus",
        help="Model identifier for the repo generator agent.",
    )
    parser.add_argument(
        "--user_model_coderefiner",
        default="deepseek/deepseek-v3.1-terminus",
        help="Model identifier for the code refiner agent.",
    )
    parser.add_argument(
        "--user_model_codejudge",
        default="deepseek/deepseek-v3.1-terminus",
        help="Model identifier for the code judge agent.",
    )
    parser.add_argument(
        "--max_iter_refinecode_rules",
        type=int,
        default=5,
        help="Maximum iterations for refining code to obey the pre-defined rules.",
    )
    parser.add_argument(
        "--max_iter_retry",
        type=int,
        default=4,
        help="Maximum number of retry iterations for each task.",
    )
    parser.set_defaults(include_unit=True)
    return parser.parse_args()


def get_task_list(topic: str, isall: bool, task_list: str) -> list[str]:
    """Get list of task names to execute."""
    tasks_dir = Path("../Paper_dataset") / topic / "tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")
    
    if isall:
        # Get all .json files
        task_files = list(tasks_dir.glob("*.json"))
        return [f.name for f in task_files]
    else:
        # Parse task_list and add .json extension
        if not task_list:
            raise ValueError("task_list must be provided when isall is False")
        task_names = [t.strip() for t in task_list.split(",")]
        return [f"{task_name}.json" for task_name in task_names]


def create_author_args(base_args, task_name: str, retry_iter: int, output_dir: str, output_repo_dir: str) -> argparse.Namespace:
    """Create arguments for task_author function."""
    args = argparse.Namespace()
    # Copy base arguments
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    # Set author-specific parameters
    args.task_name = task_name
    args.idname_proj = f"retry{retry_iter}"
    args.idname_code = args.idname_proj
    args.idname_repo = "check"
    args.output_dir = output_dir
    args.output_repo_dir = output_repo_dir
    return args


def rename_retry_to_check(current_code_dir: str, retry_iter: int) -> None:
    """Rename directory from retry{iter} to check."""
    code_dir_path = Path(current_code_dir)
    new_name = code_dir_path.name.replace(f"_retry{retry_iter}", "_check")
    new_path = code_dir_path.parent / new_name
    code_dir_path.rename(new_path)
    print(f"Renamed directory: {current_code_dir} -> {new_path}")


def generate_statistics_report(task_dir: Path, task_name: str, pdf_name: str, subplot_name: str, count_retry: int) -> None:
    """Generate statistics report for a task."""
    # Find all directories ending with retry or check
    retry_dirs = []
    check_dirs = []
    
    for item in task_dir.iterdir():
        if item.is_dir():
            name = item.name
            if name.endswith("_check"):
                check_dirs.append(item)
            elif "_retry" in name:
                retry_dirs.append(item)
    
    # Read report_author.jsonl from all directories
    total_count_iter = 0
    all_dirs = retry_dirs + check_dirs
    
    for dir_path in all_dirs:
        report_file = dir_path / "report_author.jsonl"
        if report_file.exists():
            try:
                content = report_file.read_text(encoding="utf-8").strip()
                if content:
                    data = json.loads(content, strict=False)
                    count_iter = data.get("count_iter", 0)
                    total_count_iter += count_iter
            except Exception as e:
                print(f"Warning: Failed to read report from {report_file}: {e}")
    
    # Check if check directory exists
    flag_obey = len(check_dirs) > 0
    
    # Extract code_file and repo_dir from check directory
    code_file = None
    repo_dir = None
    if check_dirs:
        check_dir = check_dirs[0]  # Use the first check directory
        report_file = check_dir / "report_author.jsonl"
        if report_file.exists():
            try:
                content = report_file.read_text(encoding="utf-8").strip()
                if content:
                    data = json.loads(content, strict=False)
                    code_file = data.get("code_file")
                    repo_dir = data.get("repo_dir")
                    
                    # Fix code_file path: replace _retryX with _check
                    if code_file:
                        code_file = re.sub(r'_retry\d+', '_check', code_file)
            except Exception as e:
                print(f"Warning: Failed to read report from {report_file}: {e}")
    
    # Create statistics report
    statistics = {
        "task_name": task_name,
        "pdf_name": pdf_name,
        "subplot_name": subplot_name,
        "count_retry": count_retry,
        "total_count_iter": total_count_iter,
        "flag_obey": flag_obey,
        "code_file": code_file,
        "repo_dir": repo_dir
    }
    
    statistics_file = task_dir / "statistics_author.jsonl"
    pretty_json = json.dumps(statistics, ensure_ascii=False, indent=2)
    with open(statistics_file, "w", encoding="utf-8") as f:
        f.write(pretty_json + "\n\n")
    
    print(f"Statistics report saved at {statistics_file}")


async def main():
    args = parse_args()
    
    # Get task list
    task_list = get_task_list(args.topic, args.isall, args.task_list)
    print(f"Found {len(task_list)} task(s) to process: {task_list}")
    
    # Process each task
    for task_name in task_list:
        print(f"\n{'='*60}")
        print(f"Processing task: {task_name}")
        print(f"{'='*60}")
        
        # Load task parameters
        task_data = OtherTools.load_task_parameters(args.topic, task_name)
        pdf_name = task_data.get("pdf_name")
        subplot_name = task_data.get("subplot_name")
        
        # Create task directories: result_dir/[topic]/[task_name] and output_repo_dir/[topic]/[task_name]
        task_dir = Path(args.result_dir) / args.topic / Path(task_name).stem
        task_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(task_dir)
        
        repo_dir = Path(args.output_repo_dir) / args.topic / Path(task_name).stem
        repo_dir.mkdir(parents=True, exist_ok=True)
        output_repo_dir = str(repo_dir)
        
        # Retry loop
        retry_iter = 0
        flag_obey = False
        current_code_dir = None
        
        while retry_iter < args.max_iter_retry:
            retry_iter += 1
            print(f"\n--- Retry iteration {retry_iter}/{args.max_iter_retry} ---")
            
            # Create author arguments
            author_args = create_author_args(base_args=args, task_name=task_name, retry_iter=retry_iter, output_dir=output_dir, output_repo_dir=output_repo_dir)
            
            # Call task_author
            try:
                result = await author_v5.task_author(author_args)
                flag_obey = result.get("flag_obey", False)
                current_code_dir = result.get("current_code_dir")
                
                if flag_obey:
                    print(f"Code obeys rules at iteration {retry_iter}")
                    # Rename directory from retry to check
                    if current_code_dir:
                        rename_retry_to_check(current_code_dir, retry_iter)
                    break
                else:
                    print(f"Code does not obey rules at iteration {retry_iter}, continuing...")
            except Exception as e:
                print(f"Error in iteration {retry_iter}: {e}")
                continue
        
        # Generate statistics report
        generate_statistics_report(task_dir, task_name, pdf_name, subplot_name, retry_iter)
        print(f"\nTask {task_name} completed. Total iterations: {retry_iter}, flag_obey: {flag_obey}")
    
    print(f"\n{'='*60}")
    print(f"All tasks completed. Processed {len(task_list)} task(s).")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

