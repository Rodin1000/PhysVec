# task_workflow_rubrics: batch execution of rubrics_v7
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
from src import QMBagents, OtherTools, rubrics_v7


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
        description="Run batch rubrics workflow to grade code files for multiple tasks."
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
        "--user_model_rubricsgrader",
        default="deepseek/deepseek-v3.1-terminus",
        help="Model identifier for the rubrics grader agent.",
    )
    parser.add_argument(
        "--idname_rubrics",
        default="untagged",
        help="Identifier for the rubrics.",
    )
    parser.add_argument(
        "--num_rubrics_grade",
        type=int,
        default=1,
        help="Number of times to run Grade_rubrics_scorecard per task; mean and std written to report.",
    )
    parser.add_argument(
        "--tag_name",
        default=None,
        help="Tag name for matching report files (if not provided, will use idname_rubrics).",
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


def create_rubrics_args(base_args, task_name: str, pdf_name: str, subplot_name: str, idname_rubrics: str, output_dir: str, tag_name: str = None) -> argparse.Namespace:
    """Create arguments for task_rubrics function."""
    args = argparse.Namespace()
    # Copy base arguments
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    # Set rubrics-specific parameters
    args.task_name = task_name
    args.idname_proj = None
    args.idname_rubrics = idname_rubrics
    args.user_model_rubricsgrader = base_args.user_model_rubricsgrader
    args.output_dir = output_dir
    args.tag_name = tag_name
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
        
        # Extract required information
        code_file = statistics.get("code_file")
        pdf_name = statistics.get("pdf_name")
        subplot_name = statistics.get("subplot_name")
        task_name_json = statistics.get("task_name", f"{task_name}.json")
        
        if not code_file:
            print(f"Skipping task {task_name}: code_file is missing in statistics_author.jsonl")
            continue
        
        # Normalize code_file path (handle Windows-style paths)
        code_file = code_file.replace("\\", "/")
        code_file_path = Path(code_file)
        # Verify code_file exists
        if not code_file_path.exists():
            print(f"Skipping task {task_name}: code_file does not exist: {code_file}")
            continue
        # Extract output_dir from code_file path (parent directory of code_file)
        output_dir = str(code_file_path.parent)
        
        # Create rubrics arguments
        # tag_name will default to idname_rubrics in rubrics_v7.judge_rubrics if not provided
        rubrics_args = create_rubrics_args(
            base_args=args,
            task_name=task_name_json,
            pdf_name=pdf_name,
            subplot_name=subplot_name,
            idname_rubrics=args.idname_rubrics,
            output_dir=output_dir,
            tag_name=getattr(args, 'tag_name', None)
        )
        
        # Execute rubrics grading
        print(f"\n--- Running rubrics grading for task {task_name} ---")
        try:
            await rubrics_v7.task_rubrics(rubrics_args, code_file)
            print(f"Rubrics grading completed for task {task_name}")
        except Exception as e:
            print(f"Error in rubrics grading for task {task_name}: {e}")
            continue

        print(f"\nTask {task_name} completed successfully.")
    
    print(f"\n{'='*60}")
    print(f"All tasks completed. Processed {len(task_list)} task(s).")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

