# convtest_v1: convergence-test per ConvergenceCase (Step 1 + Step 2.1~2.5).
import os
import sys
import asyncio
import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src import scitest_agents, OtherTools, QMBagents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run convtest: Step 1 Modify per case; Step 2.1~2.5 Run, Summarize, Judge, Gen+Apply."
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="Topic (e.g., qcmb, dmrg).",
    )
    parser.add_argument(
        "--task_name",
        required=True,
        help="Task name (e.g., task_dmrg_7_2.json).",
    )
    parser.add_argument(
        "--code_file",
        required=True,
        help="Path to the code file to run convtest on.",
    )
    parser.add_argument(
        "--output_dir",
        default="../logs",
        help="Output directory for convtest outputs (e.g., code_LLM_conv*_loop1).",
    )
    parser.add_argument(
        "--loop_index",
        type=int,
        default=1,
        help="Current workflow loop index (for naming; Step 2 will use this).",
    )
    parser.add_argument(
        "--author_model",
        default="deepseek/deepseek-v3.1-terminus",
        help="Author model for Modify, Run_refine, Summarize, Apply.",
    )
    parser.add_argument(
        "--judge_model",
        default="deepseek/deepseek-v3.1-terminus",
        help="Judge model for Judge_convtest_convergence and Gen_conv_repair_suggestion.",
    )
    parser.add_argument(
        "--concurr_num_convtest",
        type=int,
        default=3,
        help="Concurrency limit for convtest Modify per case.",
    )
    parser.add_argument(
        "--max_convtest_loop",
        type=int,
        default=3,
        help="Max convtest refinement loops (Step 2, future).",
    )
    parser.add_argument(
        "--max_convtest_refine",
        type=int,
        default=3,
        help="Max refine iterations inside Run_convtest_code_and_refine (Step 2.1).",
    )
    parser.add_argument(
        "--convtest_timeout",
        type=int,
        default=10000,
        help="Timeout (seconds) for code execution in Run_convtest_code_and_refine.",
    )
    return parser.parse_args()


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


async def task_convtest(args, code_file: str, loop_index: int, output_dir: str, task_data: dict, author_model: str = None, judge_model: str = None):
    """
    For each ConvergenceCase: Step 1 Modify -> code_LLM_conv{x}_loop1;
    Step 2.1 Run_convtest_code_and_refine; Step 2.2 if exitcode!=0 break.
    Step 2.3: on exitcode==0, Summarize_convtest_run and append one line to report. Step 2.4: Judge (handles lines<2 internally). Step 2.5: if isconv copy final and break; else Gen_conv_repair_suggestion + Apply_conv_repair_to_code.
    Returns dict with "results" (list of per-case results), "author_token_stats", "judge_token_stats".
    Judge_convtest_convergence and Gen_conv_repair_suggestion use judge_model; others use author_model.
    """
    convergence_cases = task_data.get("ConvergenceCases") if task_data else None
    if not convergence_cases:
        print("[convtest] No ConvergenceCases in task_data, skipping.")
        return {"results": [], "author_token_stats": None, "judge_token_stats": None}

    print(f"[convtest] Starting task_convtest: {len(convergence_cases)} ConvergenceCase(s), loop_index={loop_index}")
    pdf_name = (task_data or {}).get("pdf_name", "unknown")
    subplot_name = (task_data or {}).get("subplot_name", "unknown")
    concurr = getattr(args, "concurr_num_convtest", 3)
    max_convtest_loop = getattr(args, "max_convtest_loop", 3)
    max_refine_iter = getattr(args, "max_convtest_refine", 3)
    timeout = getattr(args, "convtest_timeout", 30000)
    sem = asyncio.Semaphore(concurr)

    author_token_list = []
    judge_token_list = []

    async def _process_with_semaphore(case_index: int, case_dict: dict):
        async with sem:
            current_convtest_file, _ = QMBagents.create_run_log(
                pdf_name=pdf_name,
                subplotname=subplot_name,
                output_dir=output_dir,
                topic=None,
                issubfile=False,
                iscreatlog=True,
                headname_dir="",
                headname_log="log_convtest",
                idname_dir=f"conv{case_index}",
                idname_log=f"conv{case_index}",
            )
            agent = scitest_agents.Convtest_runner(
                output_dir=output_dir,
                topic=args.topic,
                current_log_file=current_convtest_file,
            )
            print(f"[convtest] Case {case_index}: Step 1 Modify_code_by_convergence_cases")
            # Step 1: Modify -> code_LLM_conv{x}_loop1 (author)
            step1_result = await asyncio.to_thread(
                agent.Modify_code_by_convergence_cases,
                user_model=author_model,
                code_file=code_file,
                convergence_case=case_dict,
                output_code_stem=f"code_LLM_conv{case_index}_loop1",
                iscaltoken=True,
                case_index=case_index,
            )
            code_file_loop1 = step1_result.get("code_file") if step1_result else None
            if not code_file_loop1:
                print(f"[convtest] Case {case_index}: Step 1 failed (no code_file), skipping.")
                return {"case_index": case_index, "step1": step1_result, "step2_1": None, "exitcode": None}
            print(f"[convtest] Case {case_index}: Step 1 done -> {code_file_loop1}")
            if step1_result and step1_result.get("token_stats"):
                author_token_list.append(step1_result["token_stats"])

            # Step 2: loop 2.1~2.5
            code_file_this = code_file_loop1
            report_path = Path(output_dir) / f"report_conv{case_index}_test.jsonl"
            last_run_result = None
            for conv_loop in range(1, max_convtest_loop + 1):
                output_stem = f"code_LLM_conv{case_index}_loop{conv_loop}"
                print(f"[convtest] Case {case_index}: Step 2.{conv_loop} Run_convtest_code_and_refine (conv_loop={conv_loop})")
                run_result = await asyncio.to_thread(
                    agent.Run_convtest_code_and_refine,
                    code_file=code_file_this,
                    user_model=author_model,
                    output_code_stem=output_stem,
                    convergence_case=case_dict,
                    max_refine_iter=max_refine_iter,
                    timeout=timeout,
                    iscaltoken=True,
                    case_index=case_index,
                    conv_loop=conv_loop,
                )
                last_run_result = run_result
                if run_result.get("exitcode") != 0:
                    print(f"[convtest] Case {case_index}: Step 2.{conv_loop} exitcode={run_result.get('exitcode')}, stopping.")
                    return {"case_index": case_index, "step1": step1_result, "step2_1": run_result, "exitcode": run_result.get("exitcode")}
                if run_result.get("token_stats"):
                    author_token_list.append(run_result["token_stats"])
                code_file_this = run_result.get("code_file") or code_file_this
                # Step 2.3: summarize and write one line to report (plan §10.5) (author)
                print(f"[convtest] Case {case_index}: Step 2.{conv_loop} Summarize_convtest_run")
                sum_result = await asyncio.to_thread(
                    agent.Summarize_convtest_run,
                    code_file=code_file_this,
                    stdout=run_result.get("stdout", ""),
                    user_model=author_model,
                    convergence_case=case_dict,
                    iscaltoken=True,
                    report_path=str(report_path),
                    conv_loop=conv_loop,
                    case_index=case_index,
                )
                if sum_result and sum_result.get("token_stats"):
                    author_token_list.append(sum_result["token_stats"])
                # Step 2.4: Judge (handles lines<2 internally), update last line (judge)
                print(f"[convtest] Case {case_index}: Step 2.{conv_loop} Judge_convtest_convergence")
                judge_result = await asyncio.to_thread(
                    agent.Judge_convtest_convergence,
                    report_path=str(report_path),
                    content=case_dict.get("content", ""),
                    criteria=case_dict.get("criteria", ""),
                    user_model=judge_model,
                    iscaltoken=True,
                    case_index=case_index,
                    conv_loop=conv_loop,
                )
                isconv = judge_result.get("isconv", False)
                if judge_result.get("token_stats"):
                    judge_token_list.append(judge_result["token_stats"])
                print(f"[convtest] Case {case_index}: Step 2.{conv_loop} Judge done, isconv={isconv}")
                # Step 2.5: if converged, copy final and break; else Gen + Apply
                if isconv:
                    final_path = Path(output_dir) / f"code_LLM_conv{case_index}_final{Path(code_file_this).suffix}"
                    shutil.copy2(code_file_this, str(final_path))
                    agent.log_converged(case_index=case_index, conv_loop=conv_loop, final_path=str(final_path), source_code_file=code_file_this)
                    print(f"[convtest] Case {case_index}: converged, saved to {final_path}")
                    return {"case_index": case_index, "step1": step1_result, "step2_1": run_result, "exitcode": 0, "isconv": True, "code_file_final": str(final_path)}
                # Gen_conv_repair_suggestion (reads isconv/reason from report last line) (judge)
                suggest_path = Path(output_dir) / f"conv_suggest_conv{case_index}_loop{conv_loop + 1}.jsonl"
                print(f"[convtest] Case {case_index}: Step 2.{conv_loop} not converged, Gen_conv_repair_suggestion + Apply")
                gen_suggest_result = await asyncio.to_thread(
                    agent.Gen_conv_repair_suggestion,
                    code_file=code_file_this,
                    report_path=str(report_path),
                    case_content=case_dict.get("content", ""),
                    case_parameter=case_dict.get("parameter", ""),
                    user_model=judge_model,
                    output_path=str(suggest_path),
                    iscaltoken=True,
                )
                if gen_suggest_result and gen_suggest_result.get("token_stats"):
                    judge_token_list.append(gen_suggest_result["token_stats"])
                # Apply_conv_repair_to_code -> code_LLM_conv{x}_loop{conv_loop+1} (author)
                apply_result = await asyncio.to_thread(
                    agent.Apply_conv_repair_to_code,
                    code_file=code_file_this,
                    suggestion_file=str(suggest_path),
                    output_code_stem=f"code_LLM_conv{case_index}_loop{conv_loop + 1}",
                    user_model=author_model,
                    iscaltoken=True,
                    case_index=case_index,
                    conv_loop=conv_loop,
                )
                code_file_next = apply_result.get("code_file")
                if not code_file_next:
                    print(f"[convtest] Case {case_index}: Apply_conv_repair failed (no code_file), stopping.")
                    return {"case_index": case_index, "step1": step1_result, "step2_1": run_result, "exitcode": run_result.get("exitcode"), "isconv": False, "apply_failed": True}
                if apply_result.get("token_stats"):
                    author_token_list.append(apply_result["token_stats"])
                code_file_this = code_file_next

            agent.log_max_loop_reached(case_index=case_index, max_convtest_loop=max_convtest_loop, last_run_result=last_run_result)
            print(f"[convtest] Case {case_index}: max_convtest_loop ({max_convtest_loop}) reached, isconv=False")
            return {"case_index": case_index, "step1": step1_result, "step2_1": last_run_result, "exitcode": last_run_result.get("exitcode") if last_run_result else None, "isconv": False}

    tasks = [
        _process_with_semaphore(i, c)
        for i, c in enumerate(convergence_cases, start=1)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"[convtest] task_convtest done: {len(results)} case(s) processed")
    return {
        "results": list(results),
        "author_token_stats": _agg_token(author_token_list),
        "judge_token_stats": _agg_token(judge_token_list),
    }


async def main():
    args = parse_args()
    task_data = OtherTools.load_task_parameters(args.topic, args.task_name)
    await task_convtest(
        args,
        code_file=args.code_file,
        loop_index=args.loop_index,
        output_dir=args.output_dir,
        task_data=task_data,
        author_model=args.author_model,
        judge_model=args.judge_model,
    )


if __name__ == "__main__":
    asyncio.run(main())
