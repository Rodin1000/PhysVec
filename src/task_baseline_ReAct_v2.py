# task_baseline_ReAct_v2: batch ReAct baseline, independent implementation
import os
import sys
import asyncio
import argparse
import json
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src import OtherTools, ReAct_v1


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
        description="Run batch ReAct baseline: for each task, run ReAct workflow in ReAct_{tag_name} subdir.",
    )
    parser.add_argument("--topic", default="dmrg", help="Topic (e.g., dmrg, nnwf).")
    parser.add_argument("--result_dir", default="../results", help="Result directory.")
    parser.add_argument("--isall", type=str2bool, default=False, help="Execute all tasks.")
    parser.add_argument("--task_list", default="", help="Comma-separated task names when isall is False.")
    parser.add_argument("--tag_name", default="run1", help="Suffix for ReAct subdir: ReAct_{tag_name}.")
    parser.add_argument("--user_model_planner", default="deepseek/deepseek-v3.1-terminus", help="Planner model.")
    parser.add_argument("--user_model_coder", default="deepseek/deepseek-chat-v3.1", help="Coder model.")
    parser.add_argument("--user_model_suggest", default="deepseek/deepseek-v3.1-terminus", help="Suggest model.")
    parser.add_argument("--max_iter_ReAct", type=int, default=5, help="Max Execute-Suggest-Repair iterations.")
    parser.add_argument("--iscodefile", type=str2bool, default=False, help="Skip Phase 1, use code from *_check/code_LLM_loop1 (resolved in ReAct_v1).")
    parser.add_argument("--isauthorrag", type=str2bool, default=False, help="Enable Retrieve (RAG) between Plan and Generate.")
    parser.add_argument("--isrepairrag", type=str2bool, default=False, help="Enable Retrieve (RAG) between Suggest and Repair.")
    parser.add_argument("--concurr_num_retrieve", type=int, default=5, help="Concurrent retrieval in repair.")
    return parser.parse_args()


def get_task_list(
    topic: str,
    isall: bool,
    task_list: str,
    result_dir: str | None = None,
    tag_name: str | None = None,
) -> list[str]:
    tasks_dir = Path(PROJECT_ROOT) / "Paper_dataset" / topic / "tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")
    if isall:
        task_files = list(tasks_dir.glob("*.json"))
        all_tasks = [f.name for f in task_files]
        if result_dir and tag_name:
            rd = Path(result_dir)
            if not rd.is_absolute():
                rd = (Path.cwd() / result_dir).resolve()
            topic_dir = rd / topic
            skip_stems = set()
            if topic_dir.exists():
                for p in topic_dir.iterdir():
                    if not p.is_dir():
                        continue
                    if any(
                        c.is_dir() and c.name.startswith("ReAct") and c.name.endswith(tag_name)
                        for c in p.iterdir()
                    ):
                        skip_stems.add(p.name)
            all_tasks = [t for t in all_tasks if Path(t).stem not in skip_stems]
        return all_tasks
    if not task_list:
        raise ValueError("task_list must be provided when isall is False")
    names = [t.strip() for t in task_list.split(",")]
    return [f"{n}.json" if not n.endswith(".json") else n for n in names]


def _args_to_serializable(args: argparse.Namespace) -> dict:
    d = {}
    for k, v in vars(args).items():
        try:
            json.dumps(v)
            d[k] = v
        except (TypeError, ValueError):
            d[k] = str(v)
    return d


def write_settings_jsonl(work_dir: Path, args: argparse.Namespace) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    settings = _args_to_serializable(args)
    path = work_dir / "settings.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(settings, ensure_ascii=False, indent=2) + "\n\n")
    print(f"Settings saved at {path}")


async def run_one_task(args: argparse.Namespace, task_name: str) -> bool:
    task_data = OtherTools.load_task_parameters(args.topic, task_name)
    pdf_name = task_data.get("pdf_name", "")
    subplot_name = task_data.get("subplot_name", "")

    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = (Path.cwd() / result_dir).resolve()
    task_dir = result_dir / args.topic / Path(task_name).stem
    task_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(task_dir)

    react_args = argparse.Namespace()
    for k, v in vars(args).items():
        setattr(react_args, k, v)
    react_args.task_name = task_name
    react_args.output_dir = output_dir
    react_args.idname_proj = args.tag_name
    react_args.idname_code = args.tag_name
    react_args.code_file = None

    try:
        result = await ReAct_v1.task_ReAct(
            output_dir=react_args.output_dir,
            topic=react_args.topic,
            task_name=react_args.task_name,
            user_model_planner=react_args.user_model_planner,
            user_model_coder=react_args.user_model_coder,
            user_model_suggest=react_args.user_model_suggest,
            max_iter_ReAct=react_args.max_iter_ReAct,
            iscodefile=react_args.iscodefile,
            code_file=react_args.code_file,
            idname_proj=react_args.idname_proj,
            idname_code=react_args.idname_code,
            isauthorrag=react_args.isauthorrag,
            isrepairrag=react_args.isrepairrag,
            concurr_num_retrieve=react_args.concurr_num_retrieve,
        )
        work_dir = result.get("work_dir")
        if work_dir:
            write_settings_jsonl(Path(work_dir), args)
        print(f"Task {task_name} completed. exit_reason={result.get('exit_reason')}, loop_index={result.get('loop_index')}")
        return True
    except Exception as e:
        print(f"Error in task {task_name}: {e}")
        return False


async def main():
    args = parse_args()
    tasks = get_task_list(
        args.topic, args.isall, args.task_list,
        result_dir=args.result_dir, tag_name=args.tag_name,
    )
    print(f"Found {len(tasks)} task(s) to process: {tasks}")
    failed_tasks: list[str] = []
    for task_name in tasks:
        print(f"\n{'='*60}")
        print(f"Processing task: {task_name}")
        print(f"{'='*60}")
        ok = await run_one_task(args, task_name)
        if not ok:
            failed_tasks.append(task_name)
    print(f"\n{'='*60}")
    print(f"All tasks completed. Processed {len(tasks)} task(s).")
    if failed_tasks:
        print(f"Skipped/Failed tasks: {failed_tasks}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
