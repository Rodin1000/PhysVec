# task_workflow_compprog_v2: author + (unittest + integtest + postprocess + coderepair) loop
# Batch execution; integrates task_workflow_author_v2, task_workflow_programtest_v2, coderepair_v7.
import os
import sys
import asyncio
import json
import argparse
import re
import shutil
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src import QMBagents, OtherTools, author_v7, unittest_v9, integtest_v9, coderepair_v7, execute_check
from src.unittest_postprocess import postprocess_unittest_report


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
        description="Run batch compprog workflow: author then loop (unittest + integtest + postprocess + coderepair) until pass or max_workflow_iter."
    )
    # From task_workflow_author_v2
    parser.add_argument("--topic", default="dmrg", help="Topic (e.g., dmrg, nnwf, etc.).")
    parser.add_argument("--result_dir", default="../results", help="Result directory for outputs.")
    parser.add_argument("--output_repo_dir", default="../Output_repo", help="Output directory for repositories.")
    parser.add_argument("--isall", type=str2bool, default=False, help="Execute all tasks in tasks directory.")
    parser.add_argument("--task_list", default="", help="Comma-separated task names (no .json) when isall is False.")
    # Author step models
    parser.add_argument("--model_step_author_role_planner", default="deepseek/deepseek-v3.1-terminus", help="Author step: planner model.")
    parser.add_argument("--model_step_author_role_coder", default="deepseek/deepseek-chat-v3.1", help="Author step: coder model.")
    parser.add_argument("--model_step_author_role_repogenerator", default="deepseek/deepseek-v3.1-terminus", help="Author step: repo generator model.")
    parser.add_argument("--model_step_author_role_coderefiner", default="deepseek/deepseek-v3.1-terminus", help="Author step: code refiner model.")
    parser.add_argument("--model_step_author_role_codejudge", default="deepseek/deepseek-v3.1-terminus", help="Author step: code judge model.")
    parser.add_argument("--max_iter_refinecode_rules", type=int, default=5, help="Max format-check iterations.")
    parser.add_argument("--max_iter_retry", type=int, default=4, help="Max author retry per task.")
    # Unittest step models
    parser.add_argument("--model_step_unittest_role_codeverifier", default="deepseek/deepseek-v3.1-terminus", help="Unittest step: code verifier model.")
    parser.add_argument("--model_step_unittest_role_judge", default="deepseek/deepseek-v3.1-terminus", help="Unittest step: judge model.")
    # Integtest step models
    parser.add_argument("--model_step_integtest_role_codeintegrator", default="deepseek/deepseek-v3.1-terminus", help="Integtest step: code integrator model.")
    parser.add_argument("--model_step_integtest_role_judge", default="deepseek/deepseek-v3.1-terminus", help="Integtest step: judge model.")
    parser.add_argument("--output_sandbox_dir_unittest", default="../CodeVerifier_sandbox", help="Unittest sandbox dir.")
    parser.add_argument("--output_sandbox_dir_integtest", default="../CodeIntegrator_sandbox", help="Integtest sandbox dir.")
    # Repair step models
    parser.add_argument("--model_step_repair_role_suggest", default="deepseek/deepseek-v3.1-terminus", help="Repair step: suggest model.")
    parser.add_argument("--model_step_repair_role_author", default="deepseek/deepseek-v3.1-terminus", help="Repair step: author model.")
    parser.add_argument("--model_step_repair_role_judge", default="deepseek/deepseek-v3.1-terminus", help="Repair step: format judge model.")
    parser.add_argument("--concurr_num_retrieve", type=int, default=5, help="Concurrent retrieval in repair.")
    # New
    parser.add_argument("--max_workflow_iter", type=int, default=5, help="Max (unittest+integtest+coderepair) loops per task.")
    parser.set_defaults(include_unit=True)
    return parser.parse_args()


def get_task_list(topic: str, isall: bool, task_list: str, result_dir: str | None = None) -> list[str]:
    """Get list of task names from Paper_dataset/topic/tasks (same as author_v2).
    When isall is True and result_dir is set, exclude only tasks whose dir under result_dir/topic/
    contains at least one subdir ending with _check."""
    tasks_dir = Path("../Paper_dataset") / topic / "tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")
    if isall:
        task_files = list(tasks_dir.glob("*.json"))
        all_tasks = [f.name for f in task_files]
        if result_dir:
            topic_dir = Path(result_dir) / topic
            skip_stems = set()
            if topic_dir.exists():
                for p in topic_dir.iterdir():
                    if not p.is_dir():
                        continue
                    if any(c.is_dir() and c.name.endswith("_check") for c in p.iterdir()):
                        skip_stems.add(p.name)
            all_tasks = [t for t in all_tasks if Path(t).stem not in skip_stems]
        return all_tasks
    if not task_list:
        raise ValueError("task_list must be provided when isall is False")
    task_names = [t.strip() for t in task_list.split(",")]
    return [f"{t}.json" if not t.endswith(".json") else t for t in task_names]


def _args_to_serializable(args: argparse.Namespace) -> dict:
    """Convert args to a dict suitable for JSON (strip non-serializable)."""
    d = {}
    for k, v in vars(args).items():
        try:
            json.dumps(v)
            d[k] = v
        except (TypeError, ValueError):
            d[k] = str(v)
    return d


def write_settings_jsonl(task_dir: Path, args: argparse.Namespace) -> None:
    """Write settings.jsonl in task_dir with current program input parameters."""
    settings = _args_to_serializable(args)
    settings_path = task_dir / "settings.jsonl"
    with open(settings_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(settings, ensure_ascii=False, indent=2) + "\n\n")
    print(f"Settings saved at {settings_path}")


def create_author_args(base_args: argparse.Namespace, task_name: str, retry_iter: int, output_dir: str, output_repo_dir: str) -> argparse.Namespace:
    """Create arguments for author_v7.task_author; map model_step_* to user_model_* / judge_model_* expected by author_v7."""
    args = argparse.Namespace()
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    args.user_model_planner = base_args.model_step_author_role_planner
    args.user_model_coder = base_args.model_step_author_role_coder
    args.user_model_repogenerator = base_args.model_step_author_role_repogenerator
    args.user_model_coderefiner = base_args.model_step_author_role_coderefiner
    args.user_model_codejudge = base_args.model_step_author_role_codejudge
    args.task_name = task_name
    args.idname_proj = f"retry{retry_iter}"
    args.idname_code = args.idname_proj
    args.idname_repo = "check"
    args.output_dir = output_dir
    args.output_repo_dir = output_repo_dir
    args.issmallscale = True
    return args


def rename_retry_to_check(current_code_dir: str, retry_iter: int) -> None:
    """Rename directory from retry{iter} to check (same as author_v2)."""
    code_dir_path = Path(current_code_dir)
    new_name = code_dir_path.name.replace(f"_retry{retry_iter}", "_check")
    new_path = code_dir_path.parent / new_name
    code_dir_path.rename(new_path)
    print(f"Renamed directory: {current_code_dir} -> {new_path}")


def generate_statistics_report(task_dir: Path, task_name: str, pdf_name: str, subplot_name: str, count_retry: int) -> None:
    """Generate statistics_author.jsonl for the task (same as author_v2)."""
    retry_dirs = []
    check_dirs = []
    for item in task_dir.iterdir():
        if item.is_dir():
            if item.name.endswith("_check"):
                check_dirs.append(item)
            elif "_retry" in item.name:
                retry_dirs.append(item)
    total_count_iter = 0
    total_author_input = total_author_output = total_judge_input = total_judge_output = 0
    total_tool_calls = 0
    per_tool_calls = {}
    for dir_path in retry_dirs + check_dirs:
        report_file = dir_path / "report_author.jsonl"
        if not report_file.exists():
            continue
        try:
            content = report_file.read_text(encoding="utf-8").strip()
            if not content:
                continue
            data = json.loads(content, strict=False)
            total_count_iter += data.get("count_iter", 0)
            st = data.get("author_token_stats") or {}
            total_author_input += st.get("input_tokens", 0)
            total_author_output += st.get("output_tokens", 0)
            st = data.get("judge_token_stats") or {}
            total_judge_input += st.get("input_tokens", 0)
            total_judge_output += st.get("output_tokens", 0)
            tool_stats = data.get("author_tool_stats") or {}
            total_tool_calls += tool_stats.get("total_tool_calls", 0)
            for tn, cnt in tool_stats.get("per_tool_calls", {}).items():
                per_tool_calls[tn] = per_tool_calls.get(tn, 0) + cnt
        except Exception as e:
            print(f"Warning: Failed to read {report_file}: {e}")
    flag_obey = len(check_dirs) > 0
    code_file = repo_dir = None
    if check_dirs:
        report_file = check_dirs[0] / "report_author.jsonl"
        if report_file.exists():
            try:
                data = json.loads(report_file.read_text(encoding="utf-8").strip(), strict=False)
                code_file = data.get("code_file")
                repo_dir = data.get("repo_dir")
                if code_file:
                    code_file = re.sub(r"_retry\d+", "_check", code_file)
            except Exception as e:
                print(f"Warning: Failed to read {report_file}: {e}")
    total_author_token_stats = None
    if total_author_input + total_author_output > 0:
        total_author_token_stats = {"input_tokens": total_author_input, "output_tokens": total_author_output, "total": total_author_input + total_author_output}
    total_judge_token_stats = None
    if total_judge_input + total_judge_output > 0:
        total_judge_token_stats = {"input_tokens": total_judge_input, "output_tokens": total_judge_output, "total": total_judge_input + total_judge_output}
    total_author_tool_stats = None
    if total_tool_calls > 0:
        total_author_tool_stats = {"total_tool_calls": total_tool_calls, "per_tool_calls": per_tool_calls}
    statistics = {
        "task_name": task_name,
        "pdf_name": pdf_name,
        "subplot_name": subplot_name,
        "count_retry": count_retry,
        "total_count_iter": total_count_iter,
        "flag_obey": flag_obey,
        "code_file": code_file,
        "repo_dir": repo_dir,
        "total_author_token_stats": total_author_token_stats,
        "total_judge_token_stats": total_judge_token_stats,
        "total_author_tool_stats": total_author_tool_stats,
    }
    statistics_file = task_dir / "statistics_author.jsonl"
    with open(statistics_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(statistics, ensure_ascii=False, indent=2) + "\n\n")
    print(f"Statistics report saved at {statistics_file}")


def load_statistics_author(task_dir: Path) -> dict | None:
    """Load statistics_author.jsonl from task directory."""
    p = task_dir / "statistics_author.jsonl"
    if not p.exists():
        return None
    try:
        content = p.read_text(encoding="utf-8").strip()
        if not content:
            return None
        return json.loads(content, strict=False)
    except Exception as e:
        print(f"Warning: Failed to read statistics_author.jsonl from {task_dir}: {e}")
        return None


def find_report_files(work_dir: Path, tag_name: str) -> tuple[Path | None, Path | None]:
    """Find unittest and integtest report files for tag."""
    unittest_report = next(work_dir.glob(f"report_unittest_*_{tag_name}.jsonl"), None)
    integtest_report = next(work_dir.glob(f"report_integtest_*_{tag_name}.jsonl"), None)
    return unittest_report, integtest_report


def check_reports_pass(work_dir: Path, tag_name: str) -> tuple[bool, dict]:
    """Check if unittest and integtest reports for tag_name both pass (correct_ratio and ave_correct_ratio == 1)."""
    unittest_report, integtest_report = find_report_files(work_dir, tag_name)
    report_info = {
        "unittest_report": str(unittest_report) if unittest_report else None,
        "integtest_report": str(integtest_report) if integtest_report else None,
        "unittest_correct_ratio": None,
        "integtest_ave_correct_ratio": None,
    }
    all_pass = True
    if unittest_report and unittest_report.exists():
        try:
            data = json.loads(unittest_report.read_text(encoding="utf-8").strip(), strict=False)
            cr = data.get("correct_ratio", 0.0)
            report_info["unittest_correct_ratio"] = cr
            if cr != 1.0:
                all_pass = False
        except Exception as e:
            print(f"Warning: Failed to read unittest report {unittest_report}: {e}")
            all_pass = False
    else:
        print(f"Warning: Unittest report not found for tag_name: {tag_name}")
        all_pass = False
    if integtest_report and integtest_report.exists():
        try:
            data = json.loads(integtest_report.read_text(encoding="utf-8").strip(), strict=False)
            acr = data.get("ave_correct_ratio", 0.0)
            report_info["integtest_ave_correct_ratio"] = acr
            if acr != 1.0:
                all_pass = False
        except Exception as e:
            print(f"Warning: Failed to read integtest report {integtest_report}: {e}")
            all_pass = False
    else:
        print(f"Warning: Integtest report not found for tag_name: {tag_name}")
        all_pass = False
    return all_pass, report_info


def create_unittest_args(base_args: argparse.Namespace, task_name_json: str, code_file: str, repo_dir: str, pdf_name: str, subplot_name: str, idname_run: str, output_dir_check: str, output_sandbox_dir_unittest: str) -> argparse.Namespace:
    """Create arguments for unittest_v9.task_unittest; map model_step_* to user_model_codeverifier / judge_model_codeverifier."""
    args = argparse.Namespace()
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    args.user_model_codeverifier = base_args.model_step_unittest_role_codeverifier
    args.judge_model_codeverifier = base_args.model_step_unittest_role_judge
    args.task_name = task_name_json
    args.code_file = code_file
    args.repo_dir = repo_dir
    args.idname_proj = None
    args.idname_unittest = idname_run
    args.upper_num_gencode = 0
    args.upper_num_exec = 0
    args.concurr_num_unittest = 8
    args.output_dir = output_dir_check
    args.output_sandbox_dir = output_sandbox_dir_unittest
    return args


def create_integtest_args(base_args: argparse.Namespace, task_name_json: str, repo_dir: str, pdf_name: str, subplot_name: str, idname_run: str, output_dir_check: str, output_sandbox_dir_integtest: str) -> argparse.Namespace:
    """Create arguments for integtest_v9.task_integtest; map model_step_* to user_model_codeintegrator / judge_model_codeintegrator."""
    args = argparse.Namespace()
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    args.user_model_codeintegrator = base_args.model_step_integtest_role_codeintegrator
    args.judge_model_codeintegrator = base_args.model_step_integtest_role_judge
    args.task_name = task_name_json
    args.repo_dir = repo_dir
    args.idname_proj = None
    args.idname_integtest = idname_run
    args.upper_num_integcode = 0
    args.level_num = 0
    args.upper_num_execode = 0
    args.concurr_num_integtest = 8
    args.output_dir = output_dir_check
    args.output_sandbox_dir = output_sandbox_dir_integtest
    return args


def create_coderepair_args(base_args: argparse.Namespace, task_name: str, task_dir: str, output_repo_dir: str, tag_name_for_report: str, tag_name: str, code_original_name: str, code_repair_name: str, pdf_name: str, subplot_name: str, code_file: str, repo_dir: str, isonlyexe: bool = False) -> argparse.Namespace:
    """Create arguments for coderepair_v7.task_coderepair; map model_step_* to suggest_model / author_model / judge_model."""
    args = argparse.Namespace()
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    args.suggest_model = base_args.model_step_repair_role_suggest
    args.author_model = base_args.model_step_repair_role_author
    args.judge_model = base_args.model_step_repair_role_judge
    args.task_name = task_name
    args.idname_proj = "check"
    args.idname_repair = tag_name
    args.tag_name_for_report = tag_name_for_report
    args.tag_name = tag_name
    args.tag_query_dir = tag_name
    args.output_dir = task_dir
    args.output_repo_dir = output_repo_dir
    args.code_original_name = code_original_name
    args.code_repair_name = code_repair_name
    args.code_file = code_file
    args.repo_dir = repo_dir
    args.isonlyexe = isonlyexe
    return args


def _agg_token(stats_list: list) -> dict | None:
    """Aggregate token stats from list of dicts."""
    total_in = total_out = 0
    for s in stats_list:
        if s and isinstance(s, dict):
            total_in += s.get("input_tokens", 0) or 0
            total_out += s.get("output_tokens", 0) or 0
    if total_in == 0 and total_out == 0:
        return None
    return {"input_tokens": total_in, "output_tokens": total_out, "total": total_in + total_out}


def _agg_tool(tool_stats_list: list) -> dict | None:
    """Aggregate tool_stats from list of dicts."""
    total_calls = 0
    per_tool = {}
    for s in tool_stats_list:
        if not s or not isinstance(s, dict):
            continue
        total_calls += s.get("total_tool_calls", 0)
        for k, v in (s.get("per_tool_calls") or {}).items():
            per_tool[k] = per_tool.get(k, 0) + v
    if total_calls == 0 and not per_tool:
        return None
    return {"total_tool_calls": total_calls, "per_tool_calls": per_tool}


def _read_judge_token_from_report(report_path: Path) -> dict | None:
    """Read judge_token_stats from a report file if present."""
    if not report_path or not report_path.exists():
        return None
    try:
        data = json.loads(report_path.read_text(encoding="utf-8").strip(), strict=False)
        return data.get("judge_token_stats") if isinstance(data.get("judge_token_stats"), dict) else None
    except Exception:
        return None


def write_statistics_compprog(
    task_dir: Path,
    check_dir: Path,
    task_name: str,
    pdf_name: str,
    subplot_name: str,
    workflow_loop_count: int,
    exit_reason: str,
    final_code_file: str | None,
    final_repo_dir: str | None,
    author_statistics: dict | None,
    repair_returns: list,
    loop_tags: list[str],
) -> None:
    """Write statistics_compprog.jsonl aggregating author + repair + judge + format token/tool stats.
    Author phase uses statistics_author (aggregated across retries), not report_author."""
    # Author phase (from statistics_author: aggregated over all retries)
    author_token_stats = (author_statistics or {}).get("total_author_token_stats")
    author_tool_stats = (author_statistics or {}).get("total_author_tool_stats")
    author_format_judge = (author_statistics or {}).get("total_judge_token_stats")

    # Repair phase (aggregate over all coderepair returns)
    repair_author_tokens = [r.get("author_token_stats") for r in repair_returns if r.get("author_token_stats")]
    repair_tool_stats_list = [r.get("repair_tool_stats") for r in repair_returns if r.get("repair_tool_stats")]
    repair_suggest_tokens = [r.get("suggest_token_stats") for r in repair_returns if r.get("suggest_token_stats")]
    repair_judge_tokens = [r.get("judge_token_stats") for r in repair_returns if r.get("judge_token_stats")]

    author_token_stats_agg = _agg_token([author_token_stats] + repair_author_tokens)
    author_tool_stats_agg = _agg_tool([author_tool_stats] + repair_tool_stats_list)
    format_token_stats = _agg_token([author_format_judge] + repair_judge_tokens)

    # Judge token: unittest + integtest judge from reports (all loop tags) + repair suggest (suggest归为judge)
    judge_token_list = []
    for tag in loop_tags:
        u_report = next(check_dir.glob(f"report_unittest_*_{tag}.jsonl"), None)
        i_report = next(check_dir.glob(f"report_integtest_*_{tag}.jsonl"), None)
        for rp in (u_report, i_report):
            jt = _read_judge_token_from_report(rp)
            if jt:
                judge_token_list.append(jt)
    judge_token_list.extend(repair_suggest_tokens)
    judge_token_stats = _agg_token(judge_token_list)

    statistics = {
        "task_name": task_name,
        "pdf_name": pdf_name,
        "subplot_name": subplot_name,
        "workflow_loop_count": workflow_loop_count,
        "exit_reason": exit_reason,
        "final_code_file": final_code_file,
        "final_repo_dir": final_repo_dir,
        "author_token_stats": author_token_stats_agg,
        "author_tool_stats": author_tool_stats_agg,
        "judge_token_stats": judge_token_stats,
        "format_token_stats": format_token_stats,
    }
    statistics_path = task_dir / "statistics_compprog.jsonl"
    with open(statistics_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(statistics, ensure_ascii=False, indent=2) + "\n\n")
    print(f"Statistics compprog saved at {statistics_path}")


async def run_one_task(args: argparse.Namespace, task_name: str) -> None:
    """Run full workflow for one task: settings -> author -> (rename code_LLM to code_LLM_loop1) -> loop (unittest -> integtest -> postprocess -> check -> coderepair) -> statistics_compprog. No final rename."""
    task_data = OtherTools.load_task_parameters(args.topic, task_name)
    pdf_name = task_data.get("pdf_name")
    subplot_name = task_data.get("subplot_name")

    task_dir = Path(args.result_dir) / args.topic / Path(task_name).stem
    task_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(task_dir)
    repo_dir_path = Path(args.output_repo_dir) / args.topic / Path(task_name).stem
    repo_dir_path.mkdir(parents=True, exist_ok=True)
    output_repo_dir = str(repo_dir_path)

    # Step A: settings.jsonl
    write_settings_jsonl(task_dir, args)

    # Step B: Author
    retry_iter = 0
    flag_obey = False
    current_code_dir = None
    while retry_iter < args.max_iter_retry:
        retry_iter += 1
        print(f"\n--- Author retry {retry_iter}/{args.max_iter_retry} ---")
        try:
            author_args = create_author_args(base_args=args, task_name=task_name, retry_iter=retry_iter, output_dir=output_dir, output_repo_dir=output_repo_dir)
            result = await author_v7.task_author(author_args)
            flag_obey = result.get("flag_obey", False)
            current_code_dir = result.get("current_code_dir")
            if flag_obey:
                if current_code_dir:
                    rename_retry_to_check(current_code_dir, retry_iter)
                break
        except Exception as e:
            print(f"Error in author iteration {retry_iter}: {e}")
            continue

    generate_statistics_report(task_dir, task_name, pdf_name, subplot_name, retry_iter)
    if not flag_obey:
        print(f"Task {task_name}: author did not obey rules after {retry_iter} retries, skipping workflow loop.")
        return

    statistics = load_statistics_author(task_dir)
    if not statistics:
        print(f"Task {task_name}: statistics_author.jsonl not found, skipping.")
        return
    code_file = statistics.get("code_file")
    repo_dir = statistics.get("repo_dir")
    task_name_json = statistics.get("task_name", task_name)
    if not code_file or not repo_dir:
        print(f"Task {task_name}: code_file or repo_dir missing, skipping.")
        return

    check_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.endswith("_check")]
    if not check_dirs:
        print(f"Task {task_name}: no check directory found, skipping.")
        return
    check_dir = check_dirs[0]
    output_dir_check = str(check_dir)

    # Sandbox paths: output_sandbox_dir_base / tag1 / topic / task_stem (same structure as task_workflow_programtest_v2, plus task layer)
    result_dir_name = Path(args.result_dir).name
    tag1 = result_dir_name.replace("results_", "") if result_dir_name.startswith("results_") else result_dir_name
    task_stem = Path(task_name).stem
    output_sandbox_dir_unittest_full = str(Path(args.output_sandbox_dir_unittest) / tag1 / args.topic / task_stem)
    output_sandbox_dir_integtest_full = str(Path(args.output_sandbox_dir_integtest) / tag1 / args.topic / task_stem)

    # Clear task-level sandbox dirs before loop (overwrite if exist)
    for sandbox_dir in (output_sandbox_dir_unittest_full, output_sandbox_dir_integtest_full):
        p = Path(sandbox_dir)
        if p.exists():
            shutil.rmtree(p)
            print(f"Cleared sandbox dir: {sandbox_dir}")

    # Step C: Loop
    loop_index = 0
    repair_returns = []
    loop_tags = []
    exit_reason = "max_iter"
    while loop_index < args.max_workflow_iter:
        loop_index += 1
        tag = f"loop{loop_index}"
        loop_tags.append(tag)
        print(f"\n--- Workflow loop {loop_index}/{args.max_workflow_iter} (tag={tag}) ---")

        # Unittest
        try:
            unittest_args = create_unittest_args(
                base_args=args,
                task_name_json=task_name_json,
                code_file=code_file,
                repo_dir=repo_dir,
                pdf_name=pdf_name,
                subplot_name=subplot_name,
                idname_run=tag,
                output_dir_check=output_dir_check,
                output_sandbox_dir_unittest=output_sandbox_dir_unittest_full,
            )
            await unittest_v9.task_unittest(unittest_args)
        except Exception as e:
            print(f"Error in unittest for {task_name} loop {loop_index}: {e}")
            # Continue: skip postprocess this loop; repair will run without unittest report.

        # Integtest
        try:
            integtest_args = create_integtest_args(
                base_args=args,
                task_name_json=task_name_json,
                repo_dir=repo_dir,
                pdf_name=pdf_name,
                subplot_name=subplot_name,
                idname_run=tag,
                output_dir_check=output_dir_check,
                output_sandbox_dir_integtest=output_sandbox_dir_integtest_full,
            )
            await integtest_v9.task_integtest(integtest_args)
        except Exception as e:
            print(f"Error in integtest for {task_name} loop {loop_index}: {e}")
            # Continue: postprocess will use unittest judge as final_judge; repair will use unittest-only.

        # Postprocess unittest report only when report_unittest exists for this tag
        u_report, _ = find_report_files(check_dir, tag)
        if u_report and u_report.exists():
            postprocess_unittest_report(check_dir, tag)

        # Check pass disabled: loop exits only on max_workflow_iter or fullcode exitcode=0
        # all_pass, report_info = check_reports_pass(check_dir, tag)
        # if all_pass:
        #     print(f"All tests pass at loop {loop_index}.")
        #     exit_reason = "all_pass"
        #     break

        # Coderepair: read tag_name_for_report=loop{index}, write tag_name=loop{index+1}
        tag_name_for_report = tag
        tag_name_next = f"loop{loop_index + 1}"
        code_original_name = f"code_LLM_loop{loop_index}"
        code_repair_name = f"code_LLM_loop{loop_index + 1}"
        isonlyexe = loop_index >= args.max_workflow_iter

        try:
            coderepair_args = create_coderepair_args(
                base_args=args,
                task_name=task_name,
                task_dir=output_dir,
                output_repo_dir=output_repo_dir,
                tag_name_for_report=tag_name_for_report,
                tag_name=tag_name_next,
                code_original_name=code_original_name,
                code_repair_name=code_repair_name,
                pdf_name=pdf_name,
                subplot_name=subplot_name,
                code_file=code_file,
                repo_dir=repo_dir,
                isonlyexe=isonlyexe,
            )
            repair_result = await coderepair_v7.task_coderepair(coderepair_args)
            repair_returns.append(repair_result)
            if isonlyexe:
                break
            if repair_result.get("no_repair_needed"):
                print(f"No repair needed at loop {loop_index}. Ending task (same as check pass).")
                exit_reason = "no_repair_needed"
                break
            code_file = repair_result.get("code_file")
            repo_dir = repair_result.get("repo_dir")
            if not code_file or not repo_dir:
                print(f"Task {task_name}: coderepair did not return code_file/repo_dir, stopping loop.")
                break
        except Exception as e:
            print(f"Error in coderepair for {task_name} loop {loop_index}: {e}")
            return

    # Step D: final_code_file is the last code file (no rename to code_LLM_final)
    final_code_file = code_file

    # Step E: statistics_compprog.jsonl
    write_statistics_compprog(
        task_dir=task_dir,
        check_dir=check_dir,
        task_name=task_name,
        pdf_name=pdf_name,
        subplot_name=subplot_name,
        workflow_loop_count=loop_index,
        exit_reason=exit_reason,
        final_code_file=final_code_file,
        final_repo_dir=repo_dir,
        author_statistics=statistics,
        repair_returns=repair_returns,
        loop_tags=loop_tags,
    )

    # Step F: execute_check disabled (execute results written during coderepair)
    # try:
    #     report_path = await execute_check.run_execute_check_async(check_dir, timeout=300, concurr_num=8)
    #     print(f"Execute check report saved at {report_path}")
    # except Exception as e:
    #     print(f"Warning: execute_check failed: {e}")

    print(f"Task {task_name} completed. Loops: {loop_index}, exit_reason: {exit_reason}")


async def main():
    args = parse_args()
    task_list = get_task_list(args.topic, args.isall, args.task_list, result_dir=args.result_dir)
    print(f"Found {len(task_list)} task(s) to process: {task_list}")

    for task_name in task_list:
        print(f"\n{'='*60}")
        print(f"Processing task: {task_name}")
        print(f"{'='*60}")
        try:
            await run_one_task(args, task_name)
        except Exception as e:
            print(f"Error processing task {task_name}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"All tasks completed. Processed {len(task_list)} task(s).")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
