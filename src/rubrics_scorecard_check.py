# rubrics_scorecard_check: check JSON scorecards in Paper_dataset/Rubrics for NormCheck (weight sum ~ 1).
# Standalone program: same weight logic as QMBagents.Generate_rubrics_scorecard, no import from QMBagents.
import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path


def collect_issues_with_weight(obj, cumulative_weight=Decimal("1")):
    """Recursively collect all dicts with 'issue' and set each unit's weight to cumulative (product along path). Same logic as QMBagents.Generate_rubrics_scorecard."""
    issues = []

    def _collect(o, cw):
        if isinstance(o, dict):
            local_weight = o.get("weight") if "weight" in o else None
            try:
                new_cum = cw * Decimal(str(local_weight)) if isinstance(local_weight, (int, float, str)) else cw
            except (InvalidOperation, Exception):
                new_cum = cw
            if "issue" in o:
                unit = dict(o)
                try:
                    unit["weight"] = float(new_cum)
                except Exception:
                    unit["weight"] = str(new_cum)
                issues.append(unit)
            for v in o.values():
                _collect(v, new_cum)
        elif isinstance(o, list):
            for item in o:
                _collect(item, cw)

    _collect(obj, cumulative_weight)
    return issues


def check_rubrics_norm(rubrics_obj: dict) -> tuple[bool, float]:
    """Return (norm_ok, total). norm_ok = |sum(issue weights) - 1| <= 1e-4. Same as QMBagents NormCheck."""
    if not isinstance(rubrics_obj, dict):
        return False, 0.0
    issues = collect_issues_with_weight(rubrics_obj)
    total = Decimal("0")
    for u in issues:
        try:
            total += Decimal(str(u.get("weight", 0)))
        except Exception:
            pass
    norm_ok = abs(float(total - Decimal("1"))) <= 1e-4
    return norm_ok, float(total)


def parse_args():
    parser = argparse.ArgumentParser(description="Check rubrics JSON scorecards for NormCheck (weight sum ~ 1).")
    parser.add_argument("--mode", default="json", choices=["json", "task"], help="json: scan Rubrics/topic/*.json; task: scan topic/tasks/*.json and check each task's rubrics_name scorecard.")
    parser.add_argument("--topic", default="all", choices=["dmrg", "nnwf", "qcmb", "all"], help="Topic subdir or all.")
    parser.add_argument("--rubrics_dir", default="../Paper_dataset/Rubrics", help="Path to Rubrics root.")
    parser.add_argument("--paper_dataset", default="../Paper_dataset", help="Path to Paper_dataset root (used in task mode for topic/tasks).")
    return parser.parse_args()


def run_json_mode(rubrics_root: Path, topics: list[str]) -> tuple[list, list]:
    """Scan Rubrics/topic/*.json, run NormCheck. Return (load_errors, norm_failures) as list of (name, err_or_None)."""
    load_errors = []
    norm_failures = []
    for topic in topics:
        topic_dir = rubrics_root / topic
        if not topic_dir.is_dir():
            continue
        for path in sorted(topic_dir.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                load_errors.append((path.name, str(e)))
                continue
            norm_ok, _ = check_rubrics_norm(data)
            if not norm_ok:
                norm_failures.append(path.name)
    return load_errors, norm_failures


def run_task_mode(paper_dataset: Path, rubrics_root: Path, topics: list[str]) -> tuple[list, list]:
    """Read topic/tasks/*.json, resolve rubrics_name to scorecard; collect not_found and norm_fail. Return (not_found_list, norm_fail_list)."""
    not_found = []  # (task_name, rubrics_name)
    norm_fail = []  # rubrics_name (dedupe)
    seen_norm_fail = set()
    for topic in topics:
        tasks_dir = paper_dataset / topic / "tasks"
        if not tasks_dir.is_dir():
            continue
        for task_path in sorted(tasks_dir.glob("*.json")):
            try:
                with open(task_path, "r", encoding="utf-8") as f:
                    task = json.load(f)
            except Exception:
                not_found.append((task_path.name, None))
                continue
            rn = task.get("rubrics_name")
            if not rn:
                not_found.append((task_path.name, "(no rubrics_name)"))
                continue
            scorecard_path = rubrics_root / topic / rn
            if not scorecard_path.is_file():
                not_found.append((task_path.name, rn))
                continue
            try:
                with open(scorecard_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                if rn not in seen_norm_fail:
                    seen_norm_fail.add(rn)
                    norm_fail.append(rn)
                continue
            norm_ok, _ = check_rubrics_norm(data)
            if not norm_ok and rn not in seen_norm_fail:
                seen_norm_fail.add(rn)
                norm_fail.append(rn)
    return not_found, norm_fail


def main():
    args = parse_args()
    rubrics_root = Path(args.rubrics_dir).resolve()
    if not rubrics_root.is_dir():
        print(f"Error: rubrics_dir not found: {rubrics_root}")
        return
    topics = ["dmrg", "nnwf", "qcmb"] if args.topic == "all" else [args.topic]

    if args.mode == "json":
        load_errors, norm_failures = run_json_mode(rubrics_root, topics)
        for name, err in load_errors:
            print(f"{name}  (load error: {err})")
        for name in norm_failures:
            print(name)
        if not load_errors and not norm_failures:
            print("All checked scorecards passed NormCheck.")
        return

    paper_dataset = Path(args.paper_dataset).resolve()
    if not paper_dataset.is_dir():
        print(f"Error: paper_dataset not found: {paper_dataset}")
        return
    not_found, norm_fail = run_task_mode(paper_dataset, rubrics_root, topics)
    for task_name, rubrics_name in not_found:
        print(f"NOT_FOUND: {task_name} -> {rubrics_name}")
    for rn in norm_fail:
        print(f"NORM_FAIL: {rn}")
    if not not_found and not norm_fail:
        print("All task scorecards found and passed NormCheck.")


if __name__ == "__main__":
    main()
