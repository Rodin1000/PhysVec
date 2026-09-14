#!/usr/bin/env python3
"""
Clean result_dir (v2): for tasks with no _check or _check fullcode mismatch,
remove task top-level files and subdirs ending with _check or retry+digits;
remove task dir only if it becomes empty.
"""
import argparse
import re
import shutil
from pathlib import Path

CODE_LOOP_PATTERN = re.compile(r"^code_LLM_loop(\d+)\.(py|jl)$")
REPORT_LOOP_PATTERN = re.compile(r"^report_fullcode_loop(\d+)\.jsonl$")
RETRY_PATTERN = re.compile(r"retry\d+$")


def get_loop_ids_from_check_dir(check_path: Path) -> tuple[set[int], set[int]]:
    """From a _check dir, return (code_loop_ids, report_loop_ids) from filenames only."""
    code_loop_ids: set[int] = set()
    report_loop_ids: set[int] = set()
    for f in check_path.iterdir():
        if not f.is_file():
            continue
        m = CODE_LOOP_PATTERN.match(f.name)
        if m:
            code_loop_ids.add(int(m.group(1)))
            continue
        m = REPORT_LOOP_PATTERN.match(f.name)
        if m:
            report_loop_ids.add(int(m.group(1)))
    return code_loop_ids, report_loop_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean tasks (v2): remove top-level files and _check/retry* subdirs; remove task only if empty."
    )
    parser.add_argument("--result_dir", required=True, help="Result root directory.")
    parser.add_argument("--topic_list", required=True, help="Comma-separated topic list (e.g. dmrg,nnwf,qcmb).")
    parser.add_argument("--dry_run", action="store_true", help="Only print what would be deleted.")
    parser.add_argument(
        "--isexecode",
        action="store_true",
        help="Also clean tasks where any _check has code_LLM_loop* vs report_fullcode_loop* mismatch.",
    )
    return parser.parse_args()


def _should_remove_subdir(name: str) -> bool:
    return name.endswith("_check") or bool(RETRY_PATTERN.search(name))


def clean_task(task_path: Path, result_dir: Path, dry_run: bool) -> bool:
    """
    Remove top-level files and _check/retry* subdirs in task_path. If empty afterward, remove task_path.
    Return True if task was removed (or would be in dry_run).
    """
    task_path = task_path.resolve()
    result_dir = result_dir.resolve()
    if result_dir not in task_path.parents:
        return False
    if not task_path.exists() or not task_path.is_dir():
        return False

    top_level_files = [f for f in task_path.iterdir() if f.is_file()]
    subdirs_to_remove = [d for d in task_path.iterdir() if d.is_dir() and _should_remove_subdir(d.name)]

    if dry_run:
        for f in top_level_files:
            print(f"[dry_run] Would remove file: {f}")
        for d in subdirs_to_remove:
            print(f"[dry_run] Would remove dir: {d}")
        remaining = set(task_path.iterdir()) - set(top_level_files) - set(subdirs_to_remove)
        if not remaining:
            print(f"[dry_run] Would remove empty task dir: {task_path}")
        return False

    for f in top_level_files:
        f.unlink()
    for d in subdirs_to_remove:
        shutil.rmtree(d)
    remaining = list(task_path.iterdir())
    if not remaining:
        shutil.rmtree(task_path)
        return True
    return False


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).resolve()
    if not result_dir.exists():
        raise FileNotFoundError(f"result_dir not found: {result_dir}")
    topics = [t.strip() for t in args.topic_list.split(",") if t.strip()]
    if not topics:
        raise ValueError("topic_list is empty after parsing")

    to_clean: list[tuple[str, Path, str]] = []

    for topic in topics:
        topic_dir = result_dir / topic
        if not topic_dir.exists() or not topic_dir.is_dir():
            if topic_dir.exists():
                print(f"Warning: not a directory, skip: {topic_dir}")
            continue
        for task_path in sorted(topic_dir.iterdir()):
            if not task_path.is_dir():
                continue
            check_dirs = [c for c in task_path.iterdir() if c.is_dir() and c.name.endswith("_check")]
            if not check_dirs:
                to_clean.append((topic, task_path, "no _check"))
                continue
            if args.isexecode:
                for check_dir in check_dirs:
                    code_ids, report_ids = get_loop_ids_from_check_dir(check_dir)
                    if code_ids != report_ids:
                        to_clean.append((topic, task_path, "loop mismatch in _check"))
                        break

    removed_count = 0
    for topic, task_path, reason in to_clean:
        if args.dry_run:
            print(f"[dry_run] Would clean: {topic}/{task_path.name} ({reason})")
            clean_task(task_path, result_dir, dry_run=True)
        else:
            removed = clean_task(task_path, result_dir, dry_run=False)
            if removed:
                removed_count += 1
                print(f"Removed empty task: {topic}/{task_path.name} ({reason})")
            else:
                print(f"Cleaned (task kept): {topic}/{task_path.name} ({reason})")

    n = len(to_clean)
    if n == 0:
        print("No tasks to clean.")
    else:
        topics_affected = sorted({t for t, _, _ in to_clean})
        print(f"Summary: cleaned {n} task(s) under topic(s): {', '.join(topics_affected)}; removed {removed_count} empty task dir(s).")


if __name__ == "__main__":
    main()
