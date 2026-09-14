"""
Postprocess report_unittest using report_integtest: add integtest and final_judge per
judge_result entry, recompute correct_ratio and undetermined_ratio. Backs up unittest
to _origin (or _origin2, ...) before modifying.

Usage:
  python unittest_postprocess.py --result_dir results --batch_tag BATCH2 \\
    --model_tags "15000token_18_1" --tag_name progt_15 [--topics "dmrg,nnwf,qcmb"]
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Postprocess report_unittest from report_integtest (add integtest/final_judge, recompute ratios)."
    )
    p.add_argument("--result_dir", required=True, help="Root of result dirs (e.g. results/)")
    p.add_argument("--batch_tag", required=True, help="Batch tag (e.g. BATCH2)")
    p.add_argument("--model_tags", required=True, help="Comma-separated model_tag list")
    p.add_argument("--tag_name", required=True, help="Report suffix (e.g. progt_15)")
    p.add_argument("--topics", default="", help="Comma-separated topics; empty = all.")
    args = p.parse_args()
    args.model_tags_list = [s.strip() for s in args.model_tags.split(",") if s.strip()]
    args.topics_list = [s.strip() for s in args.topics.split(",") if s.strip()]
    return args


def _read_file_fallback(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, OSError):
            continue
    raise OSError(f"Could not read {path}")


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        text = _read_file_fallback(path).strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            first = text.split("\n")[0].strip()
            obj = json.loads(first) if first else None
        return obj if isinstance(obj, dict) else None
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}", file=sys.stderr)
        return None


def successful_element_functions_union(integtest_data: dict) -> set[str]:
    out = set()
    for rf in integtest_data.get("run_functions") or []:
        if not isinstance(rf, dict):
            continue
        for name in rf.get("successful_element_functions") or []:
            if isinstance(name, str):
                out.add(name)
    return out


# verify_<func>_unittest.<ext> -> func
_FUNC_NAME_RE = re.compile(r"^verify_(.+)_unittest\.\w+$")


def function_name_from_target(target_file_name: str) -> str | None:
    if not target_file_name or not isinstance(target_file_name, str):
        return None
    m = _FUNC_NAME_RE.match(target_file_name.strip())
    return m.group(1) if m else None


def backup_path(unittest_path: Path) -> Path:
    stem = unittest_path.stem
    parent = unittest_path.parent
    suffix = "_origin"
    n = 1
    while True:
        candidate = parent / f"{stem}{suffix}.jsonl"
        if not candidate.exists():
            return candidate
        n += 1
        suffix = f"_origin{n}"


def process_one_check(check_dir: Path, tag_name: str) -> bool:
    integtest_pattern = f"report_integtest_*_{tag_name}.jsonl"
    unittest_pattern = f"report_unittest_*_{tag_name}.jsonl"
    integtest_files = list(check_dir.glob(integtest_pattern))
    unittest_files = list(check_dir.glob(unittest_pattern))
    if not unittest_files:
        print(f"Warning: no unittest report in {check_dir}", file=sys.stderr)
        return False
    if not integtest_files:
        print(f"Warning: no integtest report in {check_dir}, using unittest judge as final_judge", file=sys.stderr)
    unittest_path = unittest_files[0]
    integtest_data = _load_json(integtest_files[0]) if integtest_files else None
    unittest_data = _load_json(unittest_path)
    if not unittest_data:
        return False
    judge_result = unittest_data.get("judge_result")
    if not isinstance(judge_result, list):
        return False

    successful_set = successful_element_functions_union(integtest_data) if integtest_data else set()

    for entry in judge_result:
        if not isinstance(entry, dict):
            continue
        target = entry.get("target_file_name", "")
        func_name = function_name_from_target(target)
        integtest_val = "correct" if (func_name and func_name in successful_set) else None
        entry["integtest"] = integtest_val
        entry["final_judge"] = "correct" if integtest_val == "correct" else entry.get("judge", "")

    total = len(judge_result)
    correct_count = sum(1 for r in judge_result if r.get("final_judge") == "correct")
    undetermined_count = sum(1 for r in judge_result if r.get("final_judge") == "undetermined")
    unittest_data["correct_ratio"] = (correct_count / total) if total > 0 else 0.0
    unittest_data["undetermined_ratio"] = (undetermined_count / total) if total > 0 else 0.0

    backup = backup_path(unittest_path)
    shutil.copy2(unittest_path, backup)
    with open(unittest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(unittest_data, ensure_ascii=False, indent=2) + "\n")
    return True


def postprocess_unittest_report(check_dir: Path, tag_name: str) -> bool:
    """
    Postprocess unittest report using integtest report for the given check directory and tag.
    Callable wrapper for process_one_check; does not change existing implementation.

    Args:
        check_dir: Path to the check directory containing report_unittest_*_{tag_name}.jsonl and report_integtest_*_{tag_name}.jsonl
        tag_name: Tag suffix used in report filenames (e.g., "loop1")

    Returns:
        True if postprocessing succeeded, False otherwise.
    """
    return process_one_check(check_dir, tag_name)


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    if not result_dir.is_dir():
        print(f"Error: result_dir is not a directory: {result_dir}", file=sys.stderr)
        sys.exit(1)

    n_processed = 0
    for model_tag in args.model_tags_list:
        run_dir = result_dir / f"results_{args.batch_tag}_{model_tag}"
        if not run_dir.is_dir():
            print(f"Warning: skipping missing dir {run_dir}", file=sys.stderr)
            continue
        topics = args.topics_list
        if not topics:
            topics = [d.name for d in run_dir.iterdir() if d.is_dir()]
        for topic in topics:
            topic_dir = run_dir / topic
            if not topic_dir.is_dir():
                continue
            for task_dir in topic_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                check_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.endswith("_check")]
                if not check_dirs:
                    continue
                check_dir = check_dirs[0]
                if process_one_check(check_dir, args.tag_name):
                    n_processed += 1
    print(f"Processed {n_processed} check directories.")


if __name__ == "__main__":
    main()
