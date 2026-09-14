# task_workflow_compscit_v2: merge smallscale+fullparas -> code_LLM_loop1, then loop rubrics_v9 -> suggest -> retrieve -> refine -> repair (no repo).
# Batch execution; uses repair_agents (Merge, Generate_repair_suggest_by_rubrics_report, Retrieve, Refine, Repair), rubrics_v9.
import os
import sys
import re
import shutil
import asyncio
import json
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src import OtherTools, rubrics_v9, repair_agents, convtest_v1, QMBagents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compscit workflow: merge smallscale+fullparas, then loop rubrics -> suggest -> retrieve -> refine -> repair."
    )
    parser.add_argument("--result_dir", default="../results", help="Result directory for outputs.")
    parser.add_argument("--ref_result_dir", default="", help="Reference result dir; copy fullparas and loopx (max x) from ref xxx_check to result. Empty = skip.")
    parser.add_argument("--topic", required=True, help="Topic (e.g., qcmb, dmrg).")
    parser.add_argument("--task_list", required=True, help="Comma-separated task names (no .json), e.g. task_nnwf_4_5,task_nnwf_4_6.")
    parser.add_argument("--max_workflow_iter", type=int, default=5, help="Max rubrics+repair loops per task.")
    parser.add_argument("--user_model_rubricsgrader", default="deepseek/deepseek-v3.1-terminus", help="Rubrics grader model.")
    parser.add_argument("--model_step_repair_role_suggest", default="deepseek/deepseek-v3.1-terminus", help="Repair suggest model.")
    parser.add_argument("--model_step_repair_role_author", default="deepseek/deepseek-v3.1-terminus", help="Repair author model.")
    parser.add_argument("--concurr_num_retrieve", type=int, default=5, help="Concurrent retrieval in repair.")
    parser.add_argument("--num_rubrics_grade", type=int, default=1, help="Number of Grade_rubrics_scorecard runs per loop.")
    parser.add_argument("--concurr_num_limiting", type=int, default=3, help="Concurrency limit for Test_LimitingCases in rubrics.")
    parser.add_argument("--max_iter_limiting", type=int, default=3, help="Max refine iterations per limiting case and for Execute_and_refine_code.")
    parser.add_argument("--max_convtest_loop", type=int, default=3, help="Max convtest refinement loops (Step 2, future).")
    parser.add_argument("--concurr_num_convtest", type=int, default=3, help="Concurrency limit for convtest Modify per case.")
    parser.add_argument("--max_convtest_refine", type=int, default=3, help="Max refine iterations inside Run_convtest_code_and_refine (Step 2.1).")
    parser.add_argument("--convtest_timeout", type=int, default=30000, help="Timeout (seconds) for code execution in Run_convtest_code_and_refine.")
    parser.add_argument("--exec_refine_timeout", type=int, default=30000, help="Timeout (seconds) for Execute_and_refine_code in rubrics repair loop.")
    return parser.parse_args()


def _safe_idname(s: str) -> str:
    """Safe string for idname_proj (log/report naming)."""
    return re.sub(r"[^\w\-]", "_", s) if s else "untagged"


def get_task_list(task_list: str, result_dir: str, topic: str) -> list[str]:
    """Parse task_list (comma-separated); return list of task names (with .json for load_task_parameters)."""
    if not task_list or not task_list.strip():
        return []
    names = [t.strip() for t in task_list.split(",") if t.strip()]
    return [f"{Path(n).stem}.json" if not n.endswith(".json") else n for n in names]


def copy_from_ref_check_one_task(
    ref_result_dir: Path,
    result_dir: Path,
    topic: str,
    task_name: str,
) -> int:
    """Copy code_LLM_fullparas and code_LLM_loopx (max x) from ref xxx_check to result.
    Target: same dir name (xxx_check). Rename loopx -> code_LLM_smallscale. ref_result_dir is strictly read-only.
    On any missing file/dir: sys.exit(1)."""
    task_stem = Path(task_name).stem
    ref_task_dir = ref_result_dir / topic / task_stem
    if not ref_task_dir.exists() or not ref_task_dir.is_dir():
        print(f"Error: ref task dir not found: {ref_task_dir}", flush=True)
        sys.exit(1)

    ref_check_dirs = [d for d in ref_task_dir.iterdir() if d.is_dir() and d.name.endswith("_check")]
    if not ref_check_dirs:
        print(f"Error: no xxx_check subdirs in ref {ref_task_dir}", flush=True)
        sys.exit(1)

    result_task_dir = result_dir / topic / task_stem
    result_task_dir.mkdir(parents=True, exist_ok=True)
    loop_pat = re.compile(r"^code_LLM_loop(\d+)\.(jl|py)$")
    copied = 0

    for ref_check in ref_check_dirs:
        fullparas_src = None
        for ext in (".jl", ".py"):
            fp = ref_check / f"code_LLM_fullparas{ext}"
            if fp.exists():
                fullparas_src = fp
                break
        if not fullparas_src:
            print(f"Error: code_LLM_fullparas not found in {ref_check}", flush=True)
            sys.exit(1)

        loop_nums = []
        for f in ref_check.iterdir():
            if f.is_file():
                m = loop_pat.match(f.name)
                if m:
                    loop_nums.append((int(m.group(1)), f, m.group(2)))
        if not loop_nums:
            print(f"Error: no code_LLM_loopN in {ref_check}", flush=True)
            sys.exit(1)
        loop_nums.sort(key=lambda x: x[0], reverse=True)
        _, loop_max_src, loop_ext = loop_nums[0]

        target_sub = result_task_dir / ref_check.name
        if target_sub.exists():
            shutil.rmtree(target_sub)
        target_sub.mkdir(parents=True)

        shutil.copy2(fullparas_src, target_sub / fullparas_src.name)
        shutil.copy2(loop_max_src, target_sub / f"code_LLM_smallscale.{loop_ext}")
        print(f"[{task_name}] Copied {ref_check.name}: fullparas + loop{loop_nums[0][0]} -> smallscale")
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


def _read_rubrics_report(report_path: Path) -> dict | None:
    """Read report_rubrics JSON; return None if missing or invalid (robust per plan §11)."""
    if not report_path or not report_path.exists():
        return None
    try:
        raw = report_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        return json.loads(raw, strict=False)
    except Exception:
        return None


def check_exit_from_rubrics_report(report_path: Path) -> tuple[bool, str]:
    """
    Check exit conditions from report_rubrics_*_loop{x}.jsonl.
    Exit only when: (1) scorecard success (mean=1 or error_issues empty) AND (2) LimitingCases all correct.
    Returns (should_exit, exit_reason). exit_reason: "scorecard_and_limiting_pass" or "".
    """
    data = _read_rubrics_report(report_path)
    if not data:
        return False, ""

    scorecard = data.get("Scorecard") or {}
    scorecard_ok = False
    if isinstance(scorecard, dict):
        if scorecard.get("rubrics_grade_mean") == 1 or scorecard.get("error_issues") == []:
            scorecard_ok = True
    if not scorecard_ok:
        return False, ""

    limiting = (data.get("LimitingCases") or {}).get("results") or []
    limiting_ok = True
    if limiting:
        for item in limiting:
            if isinstance(item, dict) and item.get("judge") != "correct":
                limiting_ok = False
                break
    if scorecard_ok and limiting_ok:
        return True, "scorecard_and_limiting_pass"
    return False, ""


def _read_rubrics_token_from_report(report_path: Path) -> list[dict]:
    """Collect token_stats from Scorecard and LimitingCases in one report. Returns list of dicts for _agg_token."""
    data = _read_rubrics_report(report_path)
    if not data:
        return []
    out = []
    for key in ("Scorecard", "LimitingCases"):
        block = data.get(key)
        if isinstance(block, dict) and block.get("token_stats"):
            out.append(block["token_stats"])
    return out


def create_rubrics_args(base_args: argparse.Namespace, task_name: str, pdf_name: str, subplot_name: str, idname_rubrics: str, output_dir: str, idname_proj: str) -> argparse.Namespace:
    """Create args for rubrics_v9.task_rubrics; idname_proj must be safe string."""
    args = argparse.Namespace()
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    args.task_name = task_name
    args.idname_proj = idname_proj
    args.idname_rubrics = idname_rubrics
    args.user_model_rubricsgrader = base_args.user_model_rubricsgrader
    args.output_dir = output_dir
    args.num_rubrics_grade = getattr(base_args, "num_rubrics_grade", 1)
    args.concurr_num_limiting = getattr(base_args, "concurr_num_limiting", 3)
    args.max_iter_limiting = getattr(base_args, "max_iter_limiting", 3)
    return args


def write_statistics_compscit(
    task_dir: Path,
    task_name: str,
    pdf_name: str,
    subplot_name: str,
    workflow_loop_count: int,
    exit_reason: str,
    final_code_file: str | None,
    merge_token_stats: dict | None,
    rubrics_judge_token_list: list,
    repair_returns: list,
    loop_tags: list[str],
    convtest_author_token_stats: dict | None = None,
    convtest_judge_token_stats: dict | None = None,
) -> None:
    """
    Write statistics_compscit.jsonl with author_token_stats, judge_token_stats, author_tool_stats.
    merge_token_stats: from Merge_fullparas_into_smallscale.
    rubrics_judge_token_list: list of token_stats from each report (Scorecard + LimitingCases per loop).
    repair_returns: list of dicts with author_token_stats (retrieve+repair), suggest_token_stats, repair_tool_stats.
    convtest_author_token_stats, convtest_judge_token_stats: from task_convtest.
    """
    author_tokens = [merge_token_stats] if merge_token_stats else []
    for r in repair_returns:
        if r.get("author_token_stats"):
            author_tokens.append(r["author_token_stats"])
    if convtest_author_token_stats:
        author_tokens.append(convtest_author_token_stats)
    author_token_stats = _agg_token(author_tokens)

    judge_tokens = list(rubrics_judge_token_list)
    for r in repair_returns:
        if r.get("suggest_token_stats"):
            judge_tokens.append(r["suggest_token_stats"])
    if convtest_judge_token_stats:
        judge_tokens.append(convtest_judge_token_stats)
    judge_token_stats = _agg_token(judge_tokens)

    tool_list = [r.get("repair_tool_stats") for r in repair_returns if r.get("repair_tool_stats")]
    author_tool_stats = _agg_tool(tool_list)

    statistics = {
        "task_name": task_name,
        "pdf_name": pdf_name,
        "subplot_name": subplot_name,
        "workflow_loop_count": workflow_loop_count,
        "exit_reason": exit_reason,
        "final_code_file": final_code_file,
        "author_token_stats": author_token_stats,
        "judge_token_stats": judge_token_stats,
        "author_tool_stats": author_tool_stats,
    }
    statistics_path = task_dir / "statistics_compscit.jsonl"
    with open(statistics_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(statistics, ensure_ascii=False, indent=2) + "\n\n")
    print(f"Statistics compscit saved at {statistics_path}")


async def run_one_task(args: argparse.Namespace, task_name: str) -> None:
    """
    Run compscit workflow for one task: validate task_dir and _check; Merge -> loop (rubrics -> exit? -> suggest -> retrieve -> refine -> repair); write statistics.
    """
    task_stem = Path(task_name).stem
    task_dir = Path(args.result_dir) / args.topic / task_stem
    if not task_dir.exists() or not task_dir.is_dir():
        print(f"Warning: Task directory not found or not a directory: {task_dir}, skipping task {task_name}.")
        return

    check_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.endswith("_check")]
    if not check_dirs:
        print(f"Warning: No _check subdirectory under {task_dir}, skipping task {task_name}.")
        return
    check_dir = check_dirs[0]
    output_dir_check = str(check_dir)

    # Precondition: code_LLM_smallscale and code_LLM_fullparas
    small_py = check_dir / "code_LLM_smallscale.py"
    small_jl = check_dir / "code_LLM_smallscale.jl"
    if small_py.exists():
        ext = ".py"
    elif small_jl.exists():
        ext = ".jl"
    else:
        print(f"Warning: code_LLM_smallscale.py/jl not found in {check_dir}, skipping task {task_name}.")
        return
    full_path = check_dir / f"code_LLM_fullparas{ext}"
    if not full_path.exists():
        print(f"Warning: {full_path.name} not found in {check_dir}, skipping task {task_name}.")
        return

    try:
        task_data = OtherTools.load_task_parameters(args.topic, task_name)
    except Exception as e:
        print(f"Warning: load_task_parameters failed for {task_name}: {e}, skipping.")
        return

    pdf_name = task_data.get("pdf_name") or task_stem
    subplot_name = task_data.get("subplot_name") or ""
    idname_proj = _safe_idname(pdf_name) or _safe_idname(task_stem)

    # Step 1: Merge
    agent = repair_agents.Code_repairer(
        output_dir=output_dir_check,
        topic=args.topic,
    )
    merge_result = agent.Merge_fullparas_into_smallscale(
        user_model=args.model_step_repair_role_author,
        iscaltoken=True,
    )
    merge_token_stats = merge_result.get("token_stats")
    code_file = merge_result.get("code_file")
    if not code_file or not Path(code_file).exists():
        print(f"Task {task_name}: Merge did not produce code_LLM_loop1, stopping.")
        return

    # Loop
    loop_index = 0
    repair_returns = []
    loop_tags = []
    rubrics_judge_token_list = []
    exit_reason = "max_iter"

    while loop_index < args.max_workflow_iter:
        loop_index += 1
        tag = f"loop{loop_index}"
        loop_tags.append(tag)
        print(f"\n--- Workflow loop {loop_index}/{args.max_workflow_iter} (tag={tag}) ---")

        # Rubrics
        rubrics_args = create_rubrics_args(
            base_args=args,
            task_name=task_name,
            pdf_name=pdf_name,
            subplot_name=subplot_name,
            idname_rubrics=tag,
            output_dir=output_dir_check,
            idname_proj=idname_proj,
        )
        try:
            await rubrics_v9.task_rubrics(rubrics_args, code_file)
        except Exception as e:
            print(f"Error in rubrics for {task_name} loop {loop_index}: {e}")
            # Continue: treat as not passed, will do suggest->repair or hit max

        # Rubrics token for judge_token_stats (plan §8)
        report_path = next(check_dir.glob(f"report_rubrics_*_{tag}.jsonl"), None)
        if report_path:
            rubrics_judge_token_list.extend(_read_rubrics_token_from_report(report_path))

        # Exit check (report missing/invalid -> not exit, plan §11)
        should_exit, exit_reason = check_exit_from_rubrics_report(report_path) if report_path else (False, "")
        if should_exit:
            print(f"Exit condition met: {exit_reason}")
            break

        if loop_index >= args.max_workflow_iter:
            break

        tag_next = f"loop{loop_index + 1}"
        code_original_name = f"code_LLM_loop{loop_index}"
        code_repair_name = f"code_LLM_loop{loop_index + 1}"

        # Suggest (read report loop{x}, write repair_suggest_loop{x+1})
        try:
            # suggest_out = agent.Generate_repair_suggest_by_rubrics_report(
            #     user_model=args.model_step_repair_role_suggest,
            #     tag_name=tag,
            #     tag_name_for_save=tag_next,
            #     code_file_stem=code_original_name,
            #     iscaltoken=True,
            # )
            suggest_out = agent.Generate_repair_suggest_by_rubrics_report_seperate(
                user_model=args.model_step_repair_role_suggest,
                tag_name=tag,
                tag_name_for_save=tag_next,
                code_file_stem=code_original_name,
                iscaltoken=True,
            )
        except Exception as e:
            print(f"Error in suggest for {task_name} loop {loop_index}: {e}")
            break

        original_count = suggest_out.get("original_count", 0)
        if original_count == 0:
            # Skip repair: copy code_LLM_loop{x} -> code_LLM_loop{y}
            code_original_path = None
            for ext in (".py", ".jl"):
                p = check_dir / f"{code_original_name}{ext}"
                if p.exists():
                    code_original_path = p
                    break
            if code_original_path is None:
                print(f"Task {task_name}: {code_original_name} not found in {check_dir}, stopping loop.")
                break
            print(f"Suggestions empty for loop {loop_index}; skipping repair, copying {code_original_name} -> {code_repair_name}.")
            dst_path = check_dir / f"{code_repair_name}{code_original_path.suffix}"
            shutil.copy2(code_original_path, dst_path)
            code_file = str(dst_path)
            repair_returns.append({
                "author_token_stats": None,
                "suggest_token_stats": suggest_out.get("token_stats"),
                "repair_tool_stats": None,
            })
        else:
            # # Retrieve
            # try:
            #     retrieve_out = await agent.Retrieve_knowledge_for_repair_mcp_use(
            #         user_model=args.model_step_repair_role_author,
            #         tag_query_dir=tag_next,
            #         tag_name=tag_next,
            #         concurr_num=args.concurr_num_retrieve,
            #         iscaltoken=True,
            #         iscaltool=True,
            #     )
            # except Exception as e:
            #     print(f"Error in retrieve for {task_name} loop {loop_index}: {e}")
            #     break
            retrieve_out = {}

            # # Refine
            # try:
            #     refine_out = agent.Refine_repair_suggestions_by_query(
            #         user_model=args.model_step_repair_role_suggest,
            #         tag_name=tag_next,
            #         tag_query_dir=tag_next,
            #         iscaltoken=True,
            #     )
            # except Exception as e:
            #     print(f"Error in refine for {task_name} loop {loop_index}: {e}")
            #     break
            refine_out = {}

            # Repair (no repo)
            try:
                repair_out = agent.Repair_code_by_suggestions(
                    user_model=args.model_step_repair_role_author,
                    tag_name=tag_next,
                    tag_query_dir=tag_next,
                    code_original_name=code_original_name,
                    code_repair_name=code_repair_name,
                    iscaltoken=True,
                )
            except Exception as e:
                print(f"Error in repair for {task_name} loop {loop_index}: {e}")
                break

            # Generate smallscale from repaired code, execute-and-refine on smallscale, then merge back
            code_file_fullparas = repair_out.get("code_file")
            agent_codegen = QMBagents.CodeGenerator(output_dir=output_dir_check, topic=args.topic)
            smallscale_out = agent_codegen.Generate_code_small_scale(
                user_model=args.model_step_repair_role_author,
                code_file=code_file_fullparas,
                save_name=f"code_LLM_loop{loop_index + 1}_smallscale",
                iscaltoken=True,
            )
            code_file_smallscale = smallscale_out.get("code_file")
            exec_refine_out = {}
            merge_out = {}
            if code_file_smallscale:
                exec_refine_out = agent.Execute_and_refine_code(
                    code_file=code_file_smallscale,
                    user_model=args.model_step_repair_role_author,
                    max_refine_iter=getattr(args, "max_iter_limiting", 3),
                    timeout=getattr(args, "exec_refine_timeout", 30000),
                    iscaltoken=True,
                )
                merge_out = agent.Merge_fullparas_into_smallscale(
                    user_model=args.model_step_repair_role_author,
                    iscaltoken=True,
                    smallscale_path=exec_refine_out.get("code_file") or code_file_smallscale,
                    fullparas_path=code_file_fullparas,
                    output_stem=f"code_LLM_loop{loop_index + 1}",
                )
            code_file = merge_out.get("code_file") or code_file_fullparas

            # Aggregate this loop's repair stats (author = retrieve+repair+smallscale+exec_refine+merge; judge = suggest+refine; tool = retrieve)
            # When retrieve/refine are commented out, retrieve_out and refine_out are set to {}
            suggest_ts = suggest_out.get("token_stats")
            refine_ts = refine_out.get("token_stats")
            repair_author_ts = _agg_token([
                retrieve_out.get("token_stats"),
                repair_out.get("token_stats"),
                smallscale_out.get("token_stats"),
                exec_refine_out.get("token_stats"),
                merge_out.get("token_stats"),
            ])
            repair_returns.append({
                "author_token_stats": repair_author_ts,
                "suggest_token_stats": _agg_token([suggest_ts, refine_ts]),
                "repair_tool_stats": retrieve_out.get("tool_stats"),
            })

        if not code_file or not Path(code_file).exists():
            print(f"Task {task_name}: repair did not produce {code_repair_name}, stopping loop.")
            break

    final_code_file = code_file
    convtest_author_token_stats = None
    convtest_judge_token_stats = None
    if exit_reason == "scorecard_and_limiting_pass":
        try:
            convtest_out = await convtest_v1.task_convtest(
                args,
                code_file=final_code_file,
                loop_index=loop_index,
                output_dir=output_dir_check,
                task_data=task_data,
                author_model=args.model_step_repair_role_author,
                judge_model=args.model_step_repair_role_suggest,
            )
            if isinstance(convtest_out, dict):
                convtest_author_token_stats = convtest_out.get("author_token_stats")
                convtest_judge_token_stats = convtest_out.get("judge_token_stats")
        except Exception as e:
            print(f"Convtest error for {task_name}: {e}")
    else:
        print(f"Rubrics did not pass (exit_reason={exit_reason}); skipping convtest.")
    write_statistics_compscit(
        task_dir=check_dir,
        task_name=task_stem,
        pdf_name=pdf_name,
        subplot_name=subplot_name,
        workflow_loop_count=loop_index,
        exit_reason=exit_reason,
        final_code_file=final_code_file,
        merge_token_stats=merge_token_stats,
        rubrics_judge_token_list=rubrics_judge_token_list,
        repair_returns=repair_returns,
        loop_tags=loop_tags,
        convtest_author_token_stats=convtest_author_token_stats,
        convtest_judge_token_stats=convtest_judge_token_stats,
    )
    print(f"Task {task_name} completed. Loops: {loop_index}, exit_reason: {exit_reason}")


async def main():
    args = parse_args()
    task_list = get_task_list(args.task_list, args.result_dir, args.topic)
    if not task_list:
        print("No tasks to process (task_list empty or invalid).")
        return
    print(f"Found {len(task_list)} task(s) to process: {task_list}")

    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = (Path.cwd() / args.result_dir).resolve()
    ref_dir: Path | None = None
    if args.ref_result_dir and args.ref_result_dir.strip():
        ref_dir = Path(args.ref_result_dir.strip())
        if not ref_dir.is_absolute():
            ref_dir = (Path.cwd() / ref_dir).resolve()
        print(f"ref_result_dir: {ref_dir} (copy before workflow, read-only)", flush=True)

    for task_name in task_list:
        if ref_dir is not None:
            copy_from_ref_check_one_task(
                ref_result_dir=ref_dir,
                result_dir=result_dir,
                topic=args.topic,
                task_name=task_name,
            )
        task_stem = Path(task_name).stem
        task_dir = result_dir / args.topic / task_stem
        if not task_dir.exists() or not task_dir.is_dir():
            print(f"Warning: Skipping {task_name}: directory {task_dir} does not exist.")
            continue
        check_dirs = [d for d in task_dir.iterdir() if d.is_dir() and d.name.endswith("_check")]
        if not check_dirs:
            print(f"Warning: Skipping {task_name}: no _check subdirectory in {task_dir}.")
            continue
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
