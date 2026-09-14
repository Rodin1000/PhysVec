# task_baseline_compscit_v4: baseline workflow - Merge + Modify_code_plotsave + Execute_and_refine.
# Independent implementation (does not import task_workflow_compscit_v3).
# Work directory: subdir ending with baseline tag; input: code_ReAct_smallscale + code_ReAct_fullparas.
import os
import re
import sys
import json
import shutil
import asyncio
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src import repair_agents, scitest_agents


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
        description="Run baseline compscit workflow: Merge + Modify_code_plotsave + Execute_and_refine in baseline tag subdir.",
    )
    parser.add_argument("--topic", default="dmrg", help="Topic (e.g., dmrg, nnwf).")
    parser.add_argument("--result_dir", default="../results", help="Result directory.")
    parser.add_argument("--ref_result_dir", default="", help="Reference result dir; copy fullparas and loopx (max x) from ref ReAct subdirs to result. Empty = skip.")
    parser.add_argument("--tag_name", required=True, help="Baseline tag; work dir = subdir ending with this tag.")
    parser.add_argument("--isall", type=str2bool, default=False, help="Process all tasks that have baseline tag subdir.")
    parser.add_argument("--task_list", default="", help="Comma-separated task names when isall is False.")
    parser.add_argument("--model_step_repair_role_author", default="deepseek/deepseek-v3.1-terminus", help="Model for Merge, plotsave, Execute_and_refine.")
    parser.add_argument("--max_iter_refine", type=int, default=3, help="Max refine iterations in Execute_and_refine_code.")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout (seconds) for code execution.")
    return parser.parse_args()


def get_task_list(
    topic: str,
    isall: bool,
    task_list: str,
    result_dir: str,
    tag_name: str,
) -> list[str]:
    """Get tasks to process. isall=True: scan result_dir/topic for task dirs that have subdir ending with tag_name."""
    rd = Path(result_dir)
    if not rd.is_absolute():
        rd = (Path.cwd() / result_dir).resolve()
    topic_dir = rd / topic
    if not topic_dir.exists():
        raise FileNotFoundError(f"Topic directory not found: {topic_dir}")

    if isall:
        tasks = []
        for p in topic_dir.iterdir():
            if not p.is_dir():
                continue
            if any(d.is_dir() and d.name.endswith(tag_name) for d in p.iterdir()):
                tasks.append(f"{p.name}.json")
        return tasks

    if not task_list or not task_list.strip():
        raise ValueError("task_list must be provided when isall is False")
    names = [t.strip() for t in task_list.split(",") if t.strip()]
    return [f"{Path(n).stem}.json" if not n.endswith(".json") else n for n in names]


def copy_from_ref_one_task(
    ref_result_dir: Path,
    result_dir: Path,
    topic: str,
    tag_name: str,
    task_name: str,
) -> int:
    """Copy code_ReAct_fullparas and code_ReAct_loopx (max x) from ref ReAct_xxx_{tag} to result.
    Target dir: _xxx_{tag} (no ReAct prefix). ref_result_dir is strictly read-only.
    On any missing file/dir: sys.exit(1)."""
    task_stem = Path(task_name).stem
    ref_task_dir = ref_result_dir / topic / task_stem
    if not ref_task_dir.exists() or not ref_task_dir.is_dir():
        print(f"Error: ref task dir not found: {ref_task_dir}", flush=True)
        sys.exit(1)

    ref_subdirs = [d for d in ref_task_dir.iterdir() if d.is_dir() and d.name.endswith(tag_name)]
    if not ref_subdirs:
        print(f"Error: no ReAct_xxx_{tag_name} subdirs in ref {ref_task_dir}", flush=True)
        sys.exit(1)

    result_task_dir = result_dir / topic / task_stem
    result_task_dir.mkdir(parents=True, exist_ok=True)
    loop_pat = re.compile(r"^code_ReAct_loop(\d+)\.(jl|py)$")
    copied = 0

    for ref_sub in ref_subdirs:
        fullparas_src = None
        for ext in (".jl", ".py"):
            fp = ref_sub / f"code_ReAct_fullparas{ext}"
            if fp.exists():
                fullparas_src = fp
                break
        if not fullparas_src:
            print(f"Error: code_ReAct_fullparas not found in {ref_sub}", flush=True)
            sys.exit(1)

        loop_nums = []
        for f in ref_sub.iterdir():
            if f.is_file():
                m = loop_pat.match(f.name)
                if m:
                    loop_nums.append((int(m.group(1)), f, m.group(2)))
        if not loop_nums:
            print(f"Error: no code_ReAct_loopN in {ref_sub}", flush=True)
            sys.exit(1)
        loop_nums.sort(key=lambda x: x[0], reverse=True)
        _, loop_max_src, loop_ext = loop_nums[0]

        target_name = "_" + ref_sub.name[6:]  # ReAct_ -> _
        target_sub = result_task_dir / target_name
        if target_sub.exists():
            shutil.rmtree(target_sub)
        target_sub.mkdir(parents=True)

        shutil.copy2(fullparas_src, target_sub / fullparas_src.name)
        shutil.copy2(loop_max_src, target_sub / f"code_ReAct_smallscale.{loop_ext}")
        print(f"[{task_name}] Copied {ref_sub.name} -> {target_name}: fullparas + loop{loop_nums[0][0]} -> smallscale")
        copied += 1

    return copied


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


def write_report_ReAct(
    work_dir: Path,
    task_name: str,
    exit_reason: str,
    final_code_file: str | None,
    merge_token_stats,
    plotsave_token_stats,
    exec_refine_token_stats,
    refine_iterations: int,
) -> None:
    """Write report_ReAct.jsonl to work_dir with author_token_stats aggregated from all steps."""
    author_token_stats = _agg_token([merge_token_stats, plotsave_token_stats, exec_refine_token_stats])
    report = {
        "task_name": task_name,
        "exit_reason": exit_reason,
        "final_code_file": final_code_file,
        "author_token_stats": author_token_stats,
        "refine_iterations": refine_iterations,
    }
    report_path = work_dir / "report_ReAct.jsonl"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n\n")
    print(f"Report saved at {report_path}")


async def run_one_task(args: argparse.Namespace, task_name: str) -> bool:
    """Run baseline workflow for one task. Returns True on success, False on skip/failure."""
    task_stem = Path(task_name).stem
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = (Path.cwd() / args.result_dir).resolve()
    task_dir = result_dir / args.topic / task_stem

    if not task_dir.exists() or not task_dir.is_dir():
        print(f"Warning: Task directory not found: {task_dir}, skipping task {task_name}.")
        return False

    work_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.endswith(args.tag_name)]
    if not work_dirs:
        print(f"Warning: No subdir ending with '{args.tag_name}' in {task_dir}, skipping task {task_name}.")
        return False
    work_dir = work_dirs[0]
    print(f"[{task_name}] Work dir: {work_dir}")

    smallscale_path = fullparas_path = None
    for ext in (".jl", ".py"):
        small_path = work_dir / f"code_ReAct_smallscale{ext}"
        full_path = work_dir / f"code_ReAct_fullparas{ext}"
        if small_path.exists() and full_path.exists():
            smallscale_path, fullparas_path = small_path, full_path
            break
    if smallscale_path is None or fullparas_path is None:
        print(f"Warning: code_ReAct_smallscale + code_ReAct_fullparas not found in {work_dir}, skipping task {task_name}.")
        return False
    print(f"[{task_name}] Input: {smallscale_path.name} + {fullparas_path.name}")

    agent = repair_agents.Code_repairer(
        output_dir=str(work_dir),
        topic=args.topic,
        current_log_file=str(work_dir / "agent_log.jsonl"),
    )

    # Step 1: Merge
    print(f"[{task_name}] Step 1: Merge_fullparas_into_smallscale ...", flush=True)
    try:
        merge_result = agent.Merge_fullparas_into_smallscale(
            user_model=args.model_step_repair_role_author,
            iscaltoken=True,
            smallscale_path=smallscale_path,
            fullparas_path=fullparas_path,
            output_stem="code_ReAct",
        )
    except Exception as e:
        print(f"Task {task_name}: Merge failed: {e}, skipping.")
        return False

    code_file = merge_result.get("code_file")
    if not code_file or not Path(code_file).exists():
        print(f"Task {task_name}: Merge did not produce code_ReAct, skipping.")
        return False
    print(f"[{task_name}] Step 1 done -> {Path(code_file).name}")

    # Step 2: Modify_code_plotsave
    print(f"[{task_name}] Step 2: Modify_code_plotsave ...", flush=True)
    agent_scitest = scitest_agents.Convtest_runner(
        output_dir=str(work_dir),
        topic=args.topic,
        current_log_file=None,
    )
    plotsave_result = agent_scitest.Modify_code_plotsave(
        code_file=code_file,
        user_model=args.model_step_repair_role_author,
        output_code_stem="code_ReAct",
        iscaltoken=True,
    )
    if plotsave_result.get("code_file") is None:
        print(f"Task {task_name}: Modify_code_plotsave failed, skipping.")
        return False
    code_file = plotsave_result.get("code_file")
    print(f"[{task_name}] Step 2 done -> {Path(code_file).name}")

    # Step 3: Execute_and_refine
    print(f"[{task_name}] Step 3: Execute_and_refine_code (max_iter={args.max_iter_refine}, timeout={args.timeout}s) ...", flush=True)
    exec_refine_out = agent.Execute_and_refine_code(
        code_file=code_file,
        user_model=args.model_step_repair_role_author,
        max_refine_iter=args.max_iter_refine,
        timeout=args.timeout,
        iscaltoken=True,
    )
    code_file = exec_refine_out.get("code_file") or code_file
    exit_reason = "exec_success" if exec_refine_out.get("exitcode") == 0 else "max_iter"
    refine_iter = exec_refine_out.get("refine_iterations", 0)
    if refine_iter > 0:
        print(f"[{task_name}] Step 3: refined {refine_iter} time(s), exitcode={exec_refine_out.get('exitcode')}")
    print(f"[{task_name}] Step 3 done -> exit_reason={exit_reason}")

    # Write report
    print(f"[{task_name}] Writing report_ReAct.jsonl ...")
    write_report_ReAct(
        work_dir=work_dir,
        task_name=task_stem,
        exit_reason=exit_reason,
        final_code_file=code_file,
        merge_token_stats=merge_result.get("token_stats"),
        plotsave_token_stats=plotsave_result.get("token_stats"),
        exec_refine_token_stats=exec_refine_out.get("token_stats"),
        refine_iterations=refine_iter,
    )

    print(f"[{task_name}] Task completed. exit_reason={exit_reason}, refine_iterations={refine_iter}")
    return True


async def main():
    args = parse_args()
    print(f"Baseline workflow: topic={args.topic}, tag={args.tag_name}, result_dir={args.result_dir}", flush=True)
    try:
        tasks = get_task_list(
            topic=args.topic,
            isall=args.isall,
            task_list=args.task_list,
            result_dir=args.result_dir,
            tag_name=args.tag_name,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    if not tasks:
        print("No tasks to process.")
        return

    print(f"Found {len(tasks)} task(s) to process: {tasks}")
    failed_tasks: list[str] = []

    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = (Path.cwd() / args.result_dir).resolve()
    ref_dir: Path | None = None
    if args.ref_result_dir and args.ref_result_dir.strip():
        ref_dir = Path(args.ref_result_dir.strip())
        if not ref_dir.is_absolute():
            ref_dir = (Path.cwd() / ref_dir).resolve()
        print(f"ref_result_dir: {ref_dir} (copy before workflow, read-only)", flush=True)

    for i, task_name in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(tasks)}] Processing task: {task_name}")
        print(f"{'='*60}", flush=True)
        if ref_dir is not None:
            copy_from_ref_one_task(
                ref_result_dir=ref_dir,
                result_dir=result_dir,
                topic=args.topic,
                tag_name=args.tag_name,
                task_name=task_name,
            )
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
