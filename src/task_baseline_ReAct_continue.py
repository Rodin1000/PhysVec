"""Continue a ReAct baseline from an existing loop into an isolated result run."""

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
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from src import ReAct_agents, repair_agents  # noqa: E402


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
            "Continue task_baseline_ReAct_v2 from a previous loop while writing "
            "all new artifacts to an isolated result run."
        )
    )

    # Keep the original task_baseline_ReAct_v2.py CLI contract.
    parser.add_argument("--topic", default="dmrg", help="Topic (e.g., dmrg, nnwf).")
    parser.add_argument("--result_dir", default="../results", help="New result directory.")
    parser.add_argument("--isall", type=str2bool, default=False, help="Process all source tasks.")
    parser.add_argument("--task_list", default="", help="Comma-separated task names when isall is false.")
    parser.add_argument("--tag_name", default="run1", help="New output ReAct subdirectory tag.")
    parser.add_argument("--user_model_planner", default="deepseek/deepseek-v3.1-terminus")
    parser.add_argument("--user_model_coder", default="deepseek/deepseek-chat-v3.1")
    parser.add_argument("--user_model_suggest", default="deepseek/deepseek-v3.1-terminus")
    parser.add_argument(
        "--max_iter_ReAct",
        type=int,
        default=5,
        help="Number of additional code versions to execute after resume_from_loop.",
    )
    parser.add_argument("--iscodefile", type=str2bool, default=False, help="Retained for CLI compatibility; continuation skips Phase 1.")
    parser.add_argument("--isauthorrag", type=str2bool, default=False, help="Retained for CLI compatibility; continuation skips Phase 1.")
    parser.add_argument("--isrepairrag", type=str2bool, default=False)
    parser.add_argument("--concurr_num_retrieve", type=int, default=5)

    # Continuation-only inputs and safety switches.
    parser.add_argument("--resume_result_dir", required=True, help="Read-only source result run root.")
    parser.add_argument("--resume_tag_name", required=True, help="Source ReAct subdirectory tag.")
    parser.add_argument("--resume_from_loop", type=int, required=True, help="Source loop used as the continuation seed.")
    parser.add_argument("--only_failed_at_resume_loop", type=str2bool, default=True)
    parser.add_argument("--include_missing_execute_report", type=str2bool, default=False)
    parser.add_argument("--allow_existing_result_dir", type=str2bool, default=False)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser.parse_args()


def resolve_input_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (Path.cwd() / path, PROJECT_ROOT / path, PROJECT_ROOT / "run_sh" / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_output_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    source_root = resolve_input_path(args.resume_result_dir)
    result_root = resolve_output_path(args.result_dir)
    source_topic = source_root / args.topic
    if not source_topic.is_dir():
        raise FileNotFoundError(f"Resume topic directory not found: {source_topic}")
    if source_root == result_root:
        raise ValueError("resume_result_dir and result_dir must be different")
    if is_within(result_root, source_root):
        raise ValueError("result_dir must not be inside resume_result_dir")
    if args.resume_from_loop < 1:
        raise ValueError("resume_from_loop must be >= 1")
    if args.max_iter_ReAct < 1:
        raise ValueError("max_iter_ReAct must be >= 1 in continuation mode")
    if args.concurr_num_retrieve < 1:
        raise ValueError("concurr_num_retrieve must be >= 1")
    if result_root.exists() and any(result_root.iterdir()) and not args.allow_existing_result_dir:
        raise FileExistsError(
            f"Destination result_dir is non-empty: {result_root}. Choose a new run directory "
            "or explicitly set --allow_existing_result_dir true."
        )
    return source_root, result_root


def requested_task_names(args: argparse.Namespace, source_topic: Path) -> list[str]:
    available = sorted(path.name for path in source_topic.iterdir() if path.is_dir())
    if args.isall:
        return available
    requested = [item.strip().removesuffix(".json") for item in args.task_list.split(",") if item.strip()]
    if not requested:
        raise ValueError("task_list must be provided when isall is false")
    missing = [name for name in requested if name not in available]
    if missing:
        print(f"Warning: source task directories not found and will be skipped: {missing}")
    return [name for name in requested if name in available]


def load_execute_records(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return records, ["missing report_execute_ReAct.jsonl"]
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line, strict=False)
            if isinstance(value, dict):
                records.append(value)
            else:
                errors.append(f"line {line_number} is not an object")
        except Exception as exc:
            errors.append(f"line {line_number}: {exc}")
    return records, errors


def normalize_exitcode(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def inspect_candidate(args: argparse.Namespace, source_task_dir: Path, result_root: Path) -> tuple[dict[str, Any] | None, str]:
    suffix = f"_{args.resume_tag_name}"
    source_dirs = sorted(
        path for path in source_task_dir.iterdir()
        if path.is_dir() and path.name.startswith("ReAct_") and path.name.endswith(suffix)
    )
    if len(source_dirs) != 1:
        return None, f"expected exactly one source ReAct *{suffix} directory, found {len(source_dirs)}"
    source_dir = source_dirs[0]
    prefix = source_dir.name[:-len(suffix)]
    target_dir = result_root / args.topic / source_task_dir.name / f"{prefix}_{args.tag_name}"
    if target_dir.exists():
        return None, f"destination ReAct directory already exists: {target_dir}"

    stems = [source_dir / f"code_ReAct_loop{args.resume_from_loop}{ext}" for ext in (".py", ".jl")]
    code_files = [path for path in stems if path.is_file()]
    if len(code_files) != 1:
        return None, f"expected exactly one loop{args.resume_from_loop} .py/.jl code file, found {len(code_files)}"
    seed_code = code_files[0]

    records, report_errors = load_execute_records(source_dir / "report_execute_ReAct.jsonl")
    matching = [record for record in records if record.get("name") == seed_code.name]
    seed_record = matching[-1] if matching else None
    if seed_record is None:
        if not args.include_missing_execute_report:
            detail = "; ".join(report_errors) if report_errors else "no matching record"
            return None, f"missing loop{args.resume_from_loop} execute record ({detail})"
        exitcode = None
    else:
        exitcode = normalize_exitcode(seed_record.get("exitcode"))
        if args.only_failed_at_resume_loop and exitcode == 0:
            return None, f"loop{args.resume_from_loop} already succeeded (exitcode=0)"
        if args.only_failed_at_resume_loop and exitcode is None:
            return None, f"loop{args.resume_from_loop} execute record has no valid integer exitcode"

    return {
        "task_name": source_task_dir.name,
        "source_dir": source_dir,
        "target_dir": target_dir,
        "seed_code": seed_code,
        "seed_sha256": sha256_file(seed_code),
        "seed_record": seed_record,
        "seed_exitcode": exitcode,
        "seed_record_count": len(matching),
        "report_parse_errors": report_errors,
    }, "selected"


def aggregate_token_stats(items: list[dict[str, Any] | None]) -> dict[str, int] | None:
    input_tokens = sum((item or {}).get("input_tokens", 0) for item in items)
    output_tokens = sum((item or {}).get("output_tokens", 0) for item in items)
    if input_tokens + output_tokens == 0:
        return None
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total": input_tokens + output_tokens}


def aggregate_tool_stats(items: list[dict[str, Any] | None]) -> dict[str, Any]:
    total = 0
    per_tool: dict[str, int] = {}
    for item in items:
        if not item:
            continue
        total += item.get("total_tool_calls", 0)
        for name, count in (item.get("per_tool_calls") or {}).items():
            per_tool[name] = per_tool.get(name, 0) + count
    return {"total_tool_calls": total, "per_tool_calls": per_tool}


def append_execute_report(work_dir: Path, exec_result: dict[str, Any], code_name: str, source: str) -> None:
    record = {
        "name": code_name,
        "exitcode": exec_result.get("exitcode"),
        "stdout": exec_result.get("stdout", ""),
        "stderr": exec_result.get("stderr", ""),
        "source": source,
    }
    with (work_dir / "report_execute_ReAct.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_one_task(args: argparse.Namespace, candidate: dict[str, Any]) -> dict[str, Any]:
    source_dir: Path = candidate["source_dir"]
    target_dir: Path = candidate["target_dir"]
    seed_code: Path = candidate["seed_code"]
    if sha256_file(seed_code) != candidate["seed_sha256"]:
        raise RuntimeError(f"Source seed changed after selection; refusing to continue: {seed_code}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir()
    copied_seed = target_dir / seed_code.name
    shutil.copy2(seed_code, copied_seed)

    log_path = target_dir / f"log_ReAct_continue_{args.tag_name}.jsonl"
    log_path.touch()
    settings = vars(args).copy()
    settings.update({
        "resume_result_dir": str(resolve_input_path(args.resume_result_dir)),
        "result_dir": str(resolve_output_path(args.result_dir)),
        "phase1_skipped": True,
        "max_iter_ReAct_semantics": "additional_code_versions",
    })
    json_dump(target_dir / "settings.jsonl", settings)

    manifest: dict[str, Any] = {
        "task_name": candidate["task_name"],
        "source_react_dir": str(source_dir),
        "target_react_dir": str(target_dir),
        "resume_tag_name": args.resume_tag_name,
        "tag_name": args.tag_name,
        "resume_from_loop": args.resume_from_loop,
        "additional_code_versions": args.max_iter_ReAct,
        "seed_code": str(seed_code),
        "copied_seed_code": str(copied_seed),
        "seed_sha256": candidate["seed_sha256"],
        "seed_exitcode": candidate["seed_exitcode"],
        "seed_execute_record_count": candidate["seed_record_count"],
        "source_report_parse_errors": candidate["report_parse_errors"],
        "status": "running",
    }
    json_dump(target_dir / "continuation_manifest.json", manifest)

    prompt_dir = str(PROJECT_ROOT / "prompts")
    react_agent = ReAct_agents.ReAct_agent(
        prompt_dir=prompt_dir,
        output_dir=str(target_dir),
        current_log_file=str(log_path),
        current_code_dir=str(target_dir),
        topic=args.topic,
    )

    author_tokens: list[dict[str, Any] | None] = []
    judge_tokens: list[dict[str, Any] | None] = []
    tool_stats: list[dict[str, Any] | None] = []
    current_loop = args.resume_from_loop
    current_code = copied_seed
    seed_record = candidate["seed_record"]
    final_loop_limit = args.resume_from_loop + args.max_iter_ReAct
    print(
        f"[ReAct Continue] Phase 2: resume after loop {args.resume_from_loop}; "
        f"execute up to loop {final_loop_limit}",
        flush=True,
    )
    if seed_record is None:
        print(
            f"[ReAct Continue] Loop {current_loop}: execute report missing; re-executing seed code...",
            flush=True,
        )
        seed_exec = react_agent.Execute_code_ReAct(str(current_code))
        append_execute_report(target_dir, seed_exec, current_code.name, "resume_seed_reexecuted")
        manifest["seed_exitcode"] = seed_exec.get("exitcode")
        if seed_exec.get("exitcode") == 0:
            exit_reason = "exec_success"
            print(
                f"[ReAct Continue] Execution succeeded (exitcode=0) at seed loop {current_loop}, exiting.",
                flush=True,
            )
            report = {
                "code_file": str(current_code), "work_dir": str(target_dir),
                "exit_reason": exit_reason, "loop_index": current_loop,
                "author_token_stats": None, "judge_token_stats": None,
                "author_tool_stats": aggregate_tool_stats(tool_stats),
                "continuation": manifest,
            }
            json_dump(target_dir / "report_ReAct.jsonl", report)
            manifest.update({"status": "completed", "exit_reason": exit_reason, "final_loop": current_loop})
            json_dump(target_dir / "continuation_manifest.json", manifest)
            print(
                f"[ReAct Continue] Workflow finished. exit_reason={exit_reason}, loop_index={current_loop}",
                flush=True,
            )
            print(f"ReAct report saved at {target_dir / 'report_ReAct.jsonl'}", flush=True)
            return manifest
    else:
        # Match ReAct_v1 exactly: the suggestion model receives only the
        # in-memory execution result, not persistence/provenance fields such
        # as the report record's `name`.
        seed_exec = {
            "exitcode": candidate["seed_exitcode"],
            "stdout": seed_record.get("stdout", ""),
            "stderr": seed_record.get("stderr", ""),
        }
        append_execute_report(target_dir, seed_exec, current_code.name, "resume_seed")

    exit_reason = "unknown"
    executed_new_versions = 0

    async def repair_to_next(exec_result: dict[str, Any], loop_number: int, code_path: Path) -> Path | None:
        next_loop = loop_number + 1
        tag = f"loop{next_loop}"
        print(f"[ReAct Continue] Loop {loop_number}: generating repair suggestions...", flush=True)
        suggest_out = react_agent.Generate_repair_suggest_by_execution_result(
            user_model=args.user_model_suggest,
            execution_result=exec_result,
            code_content=code_path.read_text(encoding="utf-8"),
            tag_name_for_save=tag,
            iscaltoken=True,
        )
        suggestions = suggest_out.get("repair_suggestions", [])
        if not suggestions:
            print(
                f"[ReAct Continue] No repair suggestions at loop {loop_number}; "
                f"copying current code to loop {next_loop}.",
                flush=True,
            )
            next_path = target_dir / f"code_ReAct_loop{next_loop}{code_path.suffix}"
            shutil.copy2(code_path, next_path)
            judge_tokens.append(suggest_out.get("token_stats"))
            return next_path

        suggest_token = suggest_out.get("token_stats")
        if args.isrepairrag:
            print(f"[ReAct Continue] Loop {loop_number}: retrieving knowledge for repair...", flush=True)
            code_repairer = repair_agents.Code_repairer(
                prompt_dir=prompt_dir,
                output_dir=str(target_dir),
                current_log_file=str(log_path),
                current_code_dir=str(target_dir),
                topic=args.topic,
            )
            retrieve_out = await code_repairer.Retrieve_knowledge_for_repair_mcp_use(
                user_model=args.user_model_coder,
                tag_query_dir=tag,
                tag_name=tag,
                concurr_num=args.concurr_num_retrieve,
                iscaltoken=True,
                iscaltool=True,
            )
            author_tokens.append(retrieve_out.get("token_stats"))
            tool_stats.append(retrieve_out.get("tool_stats"))
            print(f"[ReAct Continue] Loop {loop_number}: refining repair suggestions by query...", flush=True)
            refine_out = code_repairer.Refine_repair_suggestions_by_query(
                user_model=args.user_model_suggest,
                tag_name=tag,
                tag_query_dir=tag,
                iscaltoken=True,
            )
            suggest_token = aggregate_token_stats([suggest_token, refine_out.get("token_stats")])
        judge_tokens.append(suggest_token)

        print(f"[ReAct Continue] Loop {loop_number}: repairing code...", flush=True)
        repair_out = react_agent.Repair_code_ReAct_by_suggestions(
            user_model=args.user_model_coder,
            tag_name=tag,
            code_original_name=f"code_ReAct_loop{loop_number}",
            code_repair_name=f"code_ReAct_loop{next_loop}",
            iscaltoken=True,
            tag_query_dir=tag if args.isrepairrag else None,
        )
        author_tokens.append(repair_out.get("token_stats"))
        repaired = repair_out.get("code_file")
        repaired_path = Path(repaired) if repaired and Path(repaired).is_file() else None
        if repaired_path is None:
            print(f"[ReAct Continue] Repair failed at loop {loop_number}.", flush=True)
        else:
            print(
                f"[ReAct Continue] Loop {loop_number} completed. Code saved to {repaired_path}",
                flush=True,
            )
        return repaired_path

    try:
        # Bootstrap loopN -> loopN+1 from the imported failure record.
        print(
            f"[ReAct Continue] Bootstrap repair: loop {current_loop} -> loop {current_loop + 1}",
            flush=True,
        )
        next_code = await repair_to_next(seed_exec, current_loop, current_code)
        if next_code is None:
            exit_reason = "repair_failed"
        else:
            current_loop += 1
            current_code = next_code
            while executed_new_versions < args.max_iter_ReAct:
                print(
                    f"[ReAct Continue] Loop {current_loop}/{final_loop_limit}: executing code...",
                    flush=True,
                )
                exec_result = react_agent.Execute_code_ReAct(str(current_code))
                append_execute_report(target_dir, exec_result, current_code.name, "continue_run")
                executed_new_versions += 1
                if exec_result.get("exitcode") == 0:
                    exit_reason = "exec_success"
                    print(
                        f"[ReAct Continue] Execution succeeded (exitcode=0) at loop {current_loop}, exiting.",
                        flush=True,
                    )
                    break
                if executed_new_versions >= args.max_iter_ReAct:
                    exit_reason = "max_iter"
                    print(
                        f"[ReAct Continue] Reached additional iteration limit ({args.max_iter_ReAct}) "
                        f"after loop {current_loop}, exiting before suggest.",
                        flush=True,
                    )
                    break
                next_code = await repair_to_next(exec_result, current_loop, current_code)
                if next_code is None:
                    exit_reason = "repair_failed"
                    break
                current_loop += 1
                current_code = next_code
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "final_loop": current_loop,
            "executed_new_versions": executed_new_versions,
            "source_seed_unchanged": sha256_file(seed_code) == candidate["seed_sha256"],
        })
        json_dump(target_dir / "continuation_manifest.json", manifest)
        raise

    source_unchanged = sha256_file(seed_code) == candidate["seed_sha256"]
    continuation_info = {
        "source_react_dir": str(source_dir),
        "source_seed_code": str(seed_code),
        "source_seed_sha256": candidate["seed_sha256"],
        "source_seed_unchanged": source_unchanged,
        "resume_from_loop": args.resume_from_loop,
        "executed_new_versions": executed_new_versions,
        "max_additional_code_versions": args.max_iter_ReAct,
    }
    report = {
        "code_file": str(current_code),
        "work_dir": str(target_dir),
        "exit_reason": exit_reason,
        "loop_index": current_loop,
        "author_token_stats": aggregate_token_stats(author_tokens),
        "judge_token_stats": aggregate_token_stats(judge_tokens),
        "author_tool_stats": aggregate_tool_stats(tool_stats),
        "continuation": continuation_info,
    }
    json_dump(target_dir / "report_ReAct.jsonl", report)
    manifest.update({
        "status": "completed",
        "exit_reason": exit_reason,
        "final_loop": current_loop,
        "final_code_file": str(current_code),
        "executed_new_versions": executed_new_versions,
        "source_seed_unchanged": source_unchanged,
    })
    json_dump(target_dir / "continuation_manifest.json", manifest)
    print(
        f"[ReAct Continue] Workflow finished. exit_reason={exit_reason}, loop_index={current_loop}",
        flush=True,
    )
    print(f"ReAct report saved at {target_dir / 'report_ReAct.jsonl'}", flush=True)
    return manifest


async def main() -> None:
    args = parse_args()
    source_root, result_root = validate_paths(args)
    source_topic = source_root / args.topic
    task_names = requested_task_names(args, source_topic)
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for task_name in task_names:
        candidate, reason = inspect_candidate(args, source_topic / task_name, result_root)
        if candidate is None:
            skipped.append({"task_name": task_name, "reason": reason})
        else:
            selected.append(candidate)

    preview = {
        "source_result_dir": str(source_root),
        "destination_result_dir": str(result_root),
        "topic": args.topic,
        "resume_tag_name": args.resume_tag_name,
        "tag_name": args.tag_name,
        "resume_from_loop": args.resume_from_loop,
        "additional_code_versions": args.max_iter_ReAct,
        "selected": [
            {
                "task_name": item["task_name"],
                "source_dir": str(item["source_dir"]),
                "target_dir": str(item["target_dir"]),
                "seed_code": str(item["seed_code"]),
                "seed_exitcode": item["seed_exitcode"],
            }
            for item in selected
        ],
        "skipped": skipped,
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run complete: no files or directories were created.")
        return

    result_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for candidate in selected:
        print(f"\n{'=' * 60}\nContinuing task: {candidate['task_name']}\n{'=' * 60}")
        try:
            result = await run_one_task(args, candidate)
            completed.append(result)
            print(
                f"Task {candidate['task_name']} completed. "
                f"exit_reason={result.get('exit_reason')}, final_loop={result.get('final_loop')}",
                flush=True,
            )
        except Exception as exc:
            print(f"Error in task {candidate['task_name']}: {exc}")
            failed.append({"task_name": candidate["task_name"], "error": str(exc)})

    summary = {
        **{key: value for key, value in preview.items() if key not in {"selected", "skipped"}},
        "selected_tasks": [item["task_name"] for item in selected],
        "skipped": skipped,
        "completed": completed,
        "failed": failed,
    }
    json_dump(result_root / "continuation_summary_react.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
