"""Continue task_workflow_compprog_v2 from an existing loop into an isolated run."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean like true/false/1/0/yes/no")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue task_workflow_compprog_v2 from a previous loop while writing "
            "all new artifacts to an isolated result/repo/sandbox run."
        )
    )

    # Keep the original task_workflow_compprog_v2.py CLI contract.
    parser.add_argument("--topic", default="dmrg")
    parser.add_argument("--result_dir", default="../results")
    parser.add_argument("--output_repo_dir", default="../Output_repo")
    parser.add_argument("--isall", type=str2bool, default=False)
    parser.add_argument("--task_list", default="")
    parser.add_argument("--model_step_author_role_planner", default="deepseek/deepseek-v3.1-terminus")
    parser.add_argument("--model_step_author_role_coder", default="deepseek/deepseek-chat-v3.1")
    parser.add_argument("--model_step_author_role_repogenerator", default="deepseek/deepseek-v3.1-terminus")
    parser.add_argument("--model_step_author_role_coderefiner", default="deepseek/deepseek-v3.1-terminus")
    parser.add_argument("--model_step_author_role_codejudge", default="deepseek/deepseek-v3.1-terminus")
    parser.add_argument("--max_iter_refinecode_rules", type=int, default=5)
    parser.add_argument("--max_iter_retry", type=int, default=4)
    parser.add_argument("--model_step_unittest_role_codeverifier", default="deepseek/deepseek-v3.1-terminus")
    parser.add_argument("--model_step_unittest_role_judge", default="deepseek/deepseek-v3.1-terminus")
    parser.add_argument("--model_step_integtest_role_codeintegrator", default="deepseek/deepseek-v3.1-terminus")
    parser.add_argument("--model_step_integtest_role_judge", default="deepseek/deepseek-v3.1-terminus")
    parser.add_argument("--output_sandbox_dir_unittest", default="../CodeVerifier_sandbox")
    parser.add_argument("--output_sandbox_dir_integtest", default="../CodeIntegrator_sandbox")
    parser.add_argument("--model_step_repair_role_suggest", default="deepseek/deepseek-v3.1-terminus")
    parser.add_argument("--model_step_repair_role_author", default="deepseek/deepseek-v3.1-terminus")
    parser.add_argument("--model_step_repair_role_judge", default="deepseek/deepseek-v3.1-terminus")
    parser.add_argument("--concurr_num_retrieve", type=int, default=5)
    parser.add_argument(
        "--max_workflow_iter",
        type=int,
        default=5,
        help="Number of additional workflow loops to test after the resume loop.",
    )

    # Continuation-only inputs and safety switches.
    parser.add_argument("--resume_result_dir", required=True)
    parser.add_argument("--resume_repo_dir", required=True)
    parser.add_argument("--resume_from_loop", type=int, required=True)
    parser.add_argument("--only_failed_fullcode", type=str2bool, default=True)
    parser.add_argument("--include_missing_loop_report", type=str2bool, default=False)
    parser.add_argument("--allow_existing_result_dir", type=str2bool, default=False)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    parser.set_defaults(include_unit=True)
    return parser.parse_args()


def resolve_input_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [
        Path.cwd() / path,
        PROJECT_ROOT / path,
        PROJECT_ROOT / "run_sh" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_output_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8").strip(), strict=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "resume_result_dir": resolve_input_path(args.resume_result_dir),
        "result_dir": resolve_output_path(args.result_dir),
        "output_repo_dir": resolve_output_path(args.output_repo_dir),
        "unittest_sandbox": resolve_output_path(args.output_sandbox_dir_unittest),
        "integtest_sandbox": resolve_output_path(args.output_sandbox_dir_integtest),
    }
    paths["resume_repo_dir"] = resolve_input_path(args.resume_repo_dir)

    source_topic = paths["resume_result_dir"] / args.topic
    if not source_topic.is_dir():
        raise FileNotFoundError(f"Resume topic directory not found: {source_topic}")
    if not paths["resume_repo_dir"].is_dir():
        raise FileNotFoundError(f"Resume repo directory not found: {paths['resume_repo_dir']}")
    if paths["resume_result_dir"] == paths["result_dir"]:
        raise ValueError("resume_result_dir and result_dir must be different")
    if paths["resume_repo_dir"] == paths["output_repo_dir"]:
        raise ValueError("resume_repo_dir and output_repo_dir must be different")
    if paths["unittest_sandbox"] == paths["integtest_sandbox"]:
        raise ValueError("Unittest and integtest sandbox roots must be different")
    for name in ("result_dir", "output_repo_dir", "unittest_sandbox", "integtest_sandbox"):
        if is_within(paths[name], paths["resume_result_dir"]):
            raise ValueError(f"{name} must not be inside resume_result_dir")
    source_tag = paths["resume_result_dir"].name.removeprefix("results_")
    for name in ("unittest_sandbox", "integtest_sandbox"):
        if source_tag in paths[name].parts:
            raise ValueError(f"{name} points at the source run tag and is not isolated")

    result_dir = paths["result_dir"]
    if result_dir.exists() and any(result_dir.iterdir()) and not args.allow_existing_result_dir:
        raise FileExistsError(
            f"Destination result_dir is non-empty: {result_dir}. "
            "Choose a new run directory or explicitly set --allow_existing_result_dir true."
        )
    if args.resume_from_loop < 1:
        raise ValueError("resume_from_loop must be >= 1")
    if args.max_workflow_iter < 1:
        raise ValueError("max_workflow_iter must be >= 1")
    return paths


def requested_task_names(args: argparse.Namespace, source_topic: Path) -> list[str]:
    available = sorted(path.name for path in source_topic.iterdir() if path.is_dir())
    if args.isall:
        return available
    if not args.task_list.strip():
        raise ValueError("task_list must be provided when isall is false")
    requested = [item.strip().removesuffix(".json") for item in args.task_list.split(",") if item.strip()]
    missing = [name for name in requested if name not in available]
    if missing:
        print(f"Warning: task directories not found and will be skipped: {missing}")
    return [name for name in requested if name in available]


def find_check_dir(task_dir: Path, statistics: dict[str, Any]) -> Path | None:
    final_code_raw = statistics.get("final_code_file")
    if final_code_raw:
        final_code = resolve_input_path(final_code_raw)
        if final_code.exists() and final_code.parent.name.endswith("_check"):
            return final_code.parent
    check_dirs = sorted(path for path in task_dir.iterdir() if path.is_dir() and path.name.endswith("_check"))
    return check_dirs[0] if len(check_dirs) == 1 else None


def inspect_candidate(args: argparse.Namespace, source_task_dir: Path) -> tuple[dict[str, Any] | None, str]:
    statistics_path = source_task_dir / "statistics_compprog.jsonl"
    if not statistics_path.is_file():
        return None, "missing statistics_compprog.jsonl"
    try:
        statistics = load_json(statistics_path)
    except Exception as exc:
        return None, f"invalid statistics_compprog.jsonl: {exc}"
    if statistics.get("workflow_loop_count") != args.resume_from_loop:
        return None, f"workflow_loop_count={statistics.get('workflow_loop_count')}"

    check_dir = find_check_dir(source_task_dir, statistics)
    if check_dir is None:
        return None, "cannot identify exactly one source _check directory"
    code_raw = statistics.get("final_code_file")
    repo_raw = statistics.get("final_repo_dir")
    if not code_raw or not repo_raw:
        return None, "missing final_code_file/final_repo_dir"
    code_file = resolve_input_path(code_raw)
    repo_dir = resolve_input_path(repo_raw)
    if not code_file.is_file():
        return None, f"final code not found: {code_file}"
    if not repo_dir.is_dir():
        return None, f"final repo not found: {repo_dir}"

    tag = f"loop{args.resume_from_loop}"
    fullcode_report = check_dir / f"report_fullcode_{tag}.jsonl"
    exitcode = None
    if fullcode_report.is_file():
        try:
            exitcode = load_json(fullcode_report).get("exitcode")
        except Exception as exc:
            if not args.include_missing_loop_report:
                return None, f"invalid {fullcode_report.name}: {exc}"
    elif not args.include_missing_loop_report:
        return None, f"missing {fullcode_report.name}"

    if args.only_failed_fullcode and exitcode == 0:
        return None, "source full-code exitcode=0"
    if args.only_failed_fullcode and exitcode is None and not args.include_missing_loop_report:
        return None, "source full-code exitcode is unavailable"

    return {
        "task_name": source_task_dir.name,
        "task_name_json": statistics.get("task_name", f"{source_task_dir.name}.json"),
        "pdf_name": statistics.get("pdf_name"),
        "subplot_name": statistics.get("subplot_name"),
        "source_task_dir": source_task_dir,
        "source_check_dir": check_dir,
        "source_statistics_path": statistics_path,
        "source_statistics": statistics,
        "source_code_file": code_file,
        "source_repo_dir": repo_dir,
        "source_fullcode_report": fullcode_report if fullcode_report.exists() else None,
        "source_exitcode": exitcode,
    }, "selected"


def select_candidates(args: argparse.Namespace, paths: dict[str, Path]) -> list[dict[str, Any]]:
    source_topic = paths["resume_result_dir"] / args.topic
    selected: list[dict[str, Any]] = []
    for task_name in requested_task_names(args, source_topic):
        candidate, reason = inspect_candidate(args, source_topic / task_name)
        print(f"[{('SELECT' if candidate else 'SKIP'):6}] {task_name}: {reason}")
        if candidate:
            selected.append(candidate)
    return selected


def seed_files(candidate: dict[str, Any], destination_check: Path, resume_tag: str) -> list[dict[str, str]]:
    source_check: Path = candidate["source_check_dir"]
    sources = [candidate["source_code_file"]]
    patterns = [
        f"report_unittest_*_{resume_tag}.jsonl",
        f"report_unittest_*_{resume_tag}_origin*.jsonl",
        f"report_integtest_*_{resume_tag}.jsonl",
        f"report_fullcode_{resume_tag}.jsonl",
    ]
    for pattern in patterns:
        sources.extend(sorted(source_check.glob(pattern)))

    unique_sources: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        resolved = source.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_sources.append(source)

    destination_check.mkdir(parents=True, exist_ok=False)
    copied = []
    for source in unique_sources:
        destination = destination_check / source.name
        shutil.copy2(source, destination)
        copied.append({
            "source": str(source.resolve()),
            "destination": str(destination.resolve()),
            "sha256": sha256_file(destination),
        })
    return copied


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value if isinstance(value, (str, int, float, bool, type(None))) else str(value) for key, value in vars(args).items()}


def aggregate_token_stats(items: list[dict[str, Any]]) -> dict[str, int] | None:
    input_tokens = sum((item.get("input_tokens", 0) or 0) for item in items if isinstance(item, dict))
    output_tokens = sum((item.get("output_tokens", 0) or 0) for item in items if isinstance(item, dict))
    if input_tokens == 0 and output_tokens == 0:
        return None
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total": input_tokens + output_tokens}


async def run_candidate(args: argparse.Namespace, paths: dict[str, Path], candidate: dict[str, Any]) -> dict[str, Any]:
    # Delay heavyweight project imports so dry-run stays read-only and lightweight.
    from src import coderepair_v7, integtest_v9, unittest_v9
    from src.task_workflow_compprog_v2 import (
        create_coderepair_args,
        create_integtest_args,
        create_unittest_args,
        find_report_files,
    )
    from src.unittest_postprocess import postprocess_unittest_report

    task_name = candidate["task_name"]
    task_dir = paths["result_dir"] / args.topic / task_name
    check_dir = task_dir / candidate["source_check_dir"].name
    manifest_path = task_dir / "continuation_manifest.json"
    statistics_path = task_dir / "statistics_compprog_continue.jsonl"
    if task_dir.exists():
        existing_check_dirs = sorted(
            path for path in task_dir.iterdir()
            if path.is_dir() and path.name.endswith("_check")
        )
        conflicts = [path for path in (manifest_path, statistics_path) if path.exists()]
        conflicts.extend(existing_check_dirs)
        if conflicts:
            conflict_text = ", ".join(str(path) for path in conflicts)
            raise FileExistsError(
                "Destination already contains compprog continuation artifacts; "
                f"refusing to overwrite: {conflict_text}"
            )
    resume_tag = f"loop{args.resume_from_loop}"

    output_repo_task = paths["output_repo_dir"] / args.topic / task_name
    unittest_sandbox = paths["unittest_sandbox"] / args.topic / task_name
    integtest_sandbox = paths["integtest_sandbox"] / args.topic / task_name
    for label, path in (
        ("destination repo", output_repo_task),
        ("unittest sandbox", unittest_sandbox),
        ("integration sandbox", integtest_sandbox),
    ):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Non-empty {label} already exists; refusing to overwrite: {path}")
    for write_path, root in (
        (task_dir, paths["result_dir"]),
        (output_repo_task, paths["output_repo_dir"]),
        (unittest_sandbox, paths["unittest_sandbox"]),
        (integtest_sandbox, paths["integtest_sandbox"]),
    ):
        if not is_within(write_path, root):
            raise ValueError(f"Unsafe write path outside configured root: {write_path}")

    copied_files = seed_files(candidate, check_dir, resume_tag)

    manifest = {
        "task_name": task_name,
        "task_name_json": candidate["task_name_json"],
        "source_result_dir": str(paths["resume_result_dir"]),
        "source_task_dir": str(candidate["source_task_dir"].resolve()),
        "source_check_dir": str(candidate["source_check_dir"].resolve()),
        "source_code_file": str(candidate["source_code_file"].resolve()),
        "source_repo_dir": str(candidate["source_repo_dir"].resolve()),
        "source_loop": args.resume_from_loop,
        "source_exitcode": candidate["source_exitcode"],
        "destination_result_dir": str(paths["result_dir"]),
        "destination_repo_dir": str(paths["output_repo_dir"]),
        "additional_loops": args.max_workflow_iter,
        "author_phase_skipped": True,
        "settings": serializable_args(args),
        "seed_files": copied_files,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    pdf_name = candidate["pdf_name"]
    subplot_name = candidate["subplot_name"]
    code_file = str(check_dir / candidate["source_code_file"].name)
    repo_dir = str(candidate["source_repo_dir"])
    repair_returns: list[dict[str, Any]] = []
    loop_tags: list[str] = []
    exit_reason = "max_iter"

    # Bootstrap repair: consume copied loopN reports and produce code/repo for loopN+1.
    bootstrap_next = args.resume_from_loop + 1
    bootstrap_args = create_coderepair_args(
        base_args=args,
        task_name=candidate["task_name_json"],
        task_dir=str(task_dir),
        output_repo_dir=str(output_repo_task),
        tag_name_for_report=resume_tag,
        tag_name=f"loop{bootstrap_next}",
        code_original_name=f"code_LLM_{resume_tag}",
        code_repair_name=f"code_LLM_loop{bootstrap_next}",
        pdf_name=pdf_name,
        subplot_name=subplot_name,
        code_file=code_file,
        repo_dir=repo_dir,
        isonlyexe=False,
    )
    bootstrap_result = await coderepair_v7.task_coderepair(bootstrap_args)
    repair_returns.append(bootstrap_result)
    if bootstrap_result.get("no_repair_needed"):
        exit_reason = "source_code_now_passes"
    else:
        code_file = bootstrap_result.get("code_file")
        repo_dir = bootstrap_result.get("repo_dir")
        if not code_file or not repo_dir:
            raise RuntimeError("Bootstrap repair did not return code_file/repo_dir")

        for offset in range(1, args.max_workflow_iter + 1):
            loop_number = args.resume_from_loop + offset
            tag = f"loop{loop_number}"
            loop_tags.append(tag)
            print(f"\n--- Continuation loop {offset}/{args.max_workflow_iter} (tag={tag}) ---")

            try:
                unittest_args = create_unittest_args(
                    base_args=args,
                    task_name_json=candidate["task_name_json"],
                    code_file=code_file,
                    repo_dir=repo_dir,
                    pdf_name=pdf_name,
                    subplot_name=subplot_name,
                    idname_run=tag,
                    output_dir_check=str(check_dir),
                    output_sandbox_dir_unittest=str(unittest_sandbox),
                )
                await unittest_v9.task_unittest(unittest_args)
            except Exception as exc:
                print(f"Error in unittest for {task_name} {tag}: {exc}")

            try:
                integtest_args = create_integtest_args(
                    base_args=args,
                    task_name_json=candidate["task_name_json"],
                    repo_dir=repo_dir,
                    pdf_name=pdf_name,
                    subplot_name=subplot_name,
                    idname_run=tag,
                    output_dir_check=str(check_dir),
                    output_sandbox_dir_integtest=str(integtest_sandbox),
                )
                await integtest_v9.task_integtest(integtest_args)
            except Exception as exc:
                print(f"Error in integtest for {task_name} {tag}: {exc}")

            unittest_report, _ = find_report_files(check_dir, tag)
            if unittest_report and unittest_report.exists():
                postprocess_unittest_report(check_dir, tag)

            is_last = offset >= args.max_workflow_iter
            repair_args = create_coderepair_args(
                base_args=args,
                task_name=candidate["task_name_json"],
                task_dir=str(task_dir),
                output_repo_dir=str(output_repo_task),
                tag_name_for_report=tag,
                tag_name=f"loop{loop_number + 1}",
                code_original_name=f"code_LLM_{tag}",
                code_repair_name=f"code_LLM_loop{loop_number + 1}",
                pdf_name=pdf_name,
                subplot_name=subplot_name,
                code_file=code_file,
                repo_dir=repo_dir,
                isonlyexe=is_last,
            )
            repair_result = await coderepair_v7.task_coderepair(repair_args)
            repair_returns.append(repair_result)
            if repair_result.get("no_repair_needed"):
                exit_reason = "no_repair_needed"
                break
            if is_last:
                break
            code_file = repair_result.get("code_file")
            repo_dir = repair_result.get("repo_dir")
            if not code_file or not repo_dir:
                raise RuntimeError(f"Repair for {tag} did not return code_file/repo_dir")

    token_stats = aggregate_token_stats(
        [
            stats
            for result in repair_returns
            for stats in (
                result.get("author_token_stats"),
                result.get("suggest_token_stats"),
                result.get("judge_token_stats"),
            )
            if stats
        ]
    )
    statistics = {
        "task_name": candidate["task_name_json"],
        "pdf_name": pdf_name,
        "subplot_name": subplot_name,
        "resume_from_loop": args.resume_from_loop,
        "requested_additional_loops": args.max_workflow_iter,
        "completed_continuation_loops": len(loop_tags),
        "loop_tags": loop_tags,
        "exit_reason": exit_reason,
        "source_code_file": str(candidate["source_code_file"].resolve()),
        "source_repo_dir": str(candidate["source_repo_dir"].resolve()),
        "final_code_file": code_file,
        "final_repo_dir": repo_dir,
        "continuation_repair_token_stats": token_stats,
    }
    statistics_path.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return statistics


async def async_main(args: argparse.Namespace) -> int:
    paths = validate_paths(args)
    candidates = select_candidates(args, paths)
    print(f"Selected {len(candidates)} task(s).")
    next_loop = args.resume_from_loop + 1
    final_loop = args.resume_from_loop + args.max_workflow_iter
    print(f"Continuation test range: loop{next_loop}..loop{final_loop}")
    print(f"Destination results: {paths['result_dir']}")
    print(f"Destination repos: {paths['output_repo_dir']}")
    print(f"Unittest sandbox: {paths['unittest_sandbox']}")
    print(f"Integtest sandbox: {paths['integtest_sandbox']}")
    if args.dry_run:
        print("Dry run complete: no files were created and no API calls were made.")
        return 0
    if not candidates:
        print("No eligible tasks; nothing to run.")
        return 0

    summaries = {"succeeded": [], "failed": []}
    for candidate in candidates:
        task_name = candidate["task_name"]
        print(f"\n{'=' * 60}\nContinuing task: {task_name}\n{'=' * 60}")
        try:
            await run_candidate(args, paths, candidate)
            summaries["succeeded"].append(task_name)
        except Exception as exc:
            print(f"Continuation failed for {task_name}: {exc}")
            summaries["failed"].append({"task_name": task_name, "error": str(exc)})

    paths["result_dir"].mkdir(parents=True, exist_ok=True)
    (paths["result_dir"] / "continuation_summary_compprog.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 1 if summaries["failed"] else 0


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
