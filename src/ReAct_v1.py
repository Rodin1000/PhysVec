# ReAct_v1: Plan -> Generate -> Small Scale -> (Execute -> Suggest -> Repair) loop
import os
import sys
import asyncio
import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src import QMBagents, OtherTools, ReAct_agents, repair_agents


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
        description="ReAct workflow: Plan -> Generate -> Execute -> Suggest -> Repair",
    )
    parser.add_argument("--topic", default="dmrg", help="Topic (e.g., dmrg, nnwf).")
    parser.add_argument("--task_name", default=None, help="Task name (e.g., task_1_1.json). Required when iscodefile=False.")
    parser.add_argument("--output_dir", default="../results", help="Output directory.")
    parser.add_argument("--idname_proj", default="ReAct", help="Project id.")
    parser.add_argument("--idname_code", default="ReAct", help="Code id.")
    parser.add_argument("--user_model_planner", default="deepseek/deepseek-v3.1-terminus", help="Planner model.")
    parser.add_argument("--user_model_coder", default="deepseek/deepseek-chat-v3.1", help="Coder model.")
    parser.add_argument("--user_model_suggest", default="deepseek/deepseek-v3.1-terminus", help="Suggest model.")
    parser.add_argument("--max_iter_ReAct", type=int, default=5, help="Max Execute-Suggest-Repair iterations.")
    parser.add_argument("--iscodefile", type=str2bool, default=False, help="Skip Phase 1 and use code_file.")
    parser.add_argument("--code_file", default=None, help="Initial code file when iscodefile=True.")
    parser.add_argument("--isauthorrag", type=str2bool, default=False, help="Enable Retrieve (RAG) between Plan and Generate.")
    parser.add_argument("--isrepairrag", type=str2bool, default=False, help="Enable Retrieve (RAG) between Suggest and Repair.")
    parser.add_argument("--concurr_num_retrieve", type=int, default=5, help="Concurrent retrieval in repair.")
    return parser.parse_args()


def _append_execute_report(work_dir: Path, exec_result: dict, code_file_name: str, is_first: bool) -> None:
    """Append execution result to report_execute_ReAct.jsonl. First write overwrites existing file."""
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / "report_execute_ReAct.jsonl"
    record = {
        "name": code_file_name,
        "exitcode": exec_result.get("exitcode"),
        "stdout": exec_result.get("stdout", ""),
        "stderr": exec_result.get("stderr", ""),
    }
    mode = "w" if is_first else "a"
    with open(report_path, mode, encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_report(work_dir: Path, report_dict: dict) -> str:
    """Save report_ReAct.jsonl in work_dir."""
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / "report_ReAct.jsonl"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report_dict, ensure_ascii=False, indent=2) + "\n\n")
    return str(report_path)


def _aggregate_token_stats(stats_list: list) -> dict | None:
    """Aggregate input/output/total from a list of token_stats dicts."""
    total_in = total_out = 0
    for s in stats_list or []:
        if s:
            total_in += s.get("input_tokens", 0)
            total_out += s.get("output_tokens", 0)
    if total_in + total_out == 0:
        return None
    return {"input_tokens": total_in, "output_tokens": total_out, "total": total_in + total_out}


def _default_tool_stats() -> dict:
    """Empty tool stats when no tool-using steps."""
    return {"total_tool_calls": 0, "per_tool_calls": {}}


def _aggregate_tool_stats(tool_stats_list: list) -> dict:
    """Aggregate tool_stats from list of dicts. Returns _default_tool_stats() when empty."""
    total_calls = 0
    per_tool = {}
    for s in tool_stats_list or []:
        if not s or not isinstance(s, dict):
            continue
        total_calls += s.get("total_tool_calls", 0)
        for k, v in (s.get("per_tool_calls") or {}).items():
            per_tool[k] = per_tool.get(k, 0) + v
    return {"total_tool_calls": total_calls, "per_tool_calls": per_tool} if (total_calls or per_tool) else _default_tool_stats()


async def task_ReAct(
    output_dir: str,
    topic: str,
    task_name: str | None,
    user_model_planner: str,
    user_model_coder: str,
    user_model_suggest: str,
    max_iter_ReAct: int = 5,
    iscodefile: bool = False,
    code_file: str | None = None,
    idname_proj: str = "ReAct",
    idname_code: str = "ReAct",
    isauthorrag: bool = False,
    isrepairrag: bool = False,
    concurr_num_retrieve: int = 5,
) -> dict:
    """
    ReAct workflow: Phase 1 (Plan -> Generate -> Small Scale) + Phase 2 (Execute -> Suggest -> Repair).
    If iscodefile=True, skip Phase 1 and use code_file as code_ReAct_loop1.
    """
    prompt_dir = str(Path(PROJECT_ROOT) / "prompts")

    if not iscodefile and not task_name:
        raise ValueError("iscodefile=False requires task_name to be provided")

    work_dir: Path
    current_log_file = None
    task_data = {}
    pdf_name = subplot_name = ""

    if task_name:
        task_data = OtherTools.load_task_parameters(topic, task_name)
        pdf_name = task_data.get("pdf_name", "")
        subplot_name = task_data.get("subplot_name", "")
        safe_name = Path(pdf_name).name
        if safe_name.lower().endswith(".pdf"):
            safe_name = safe_name[:-4]
        safe_name = safe_name.replace(" ", "_")
        safe_subplot = subplot_name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")
        out_dir = Path(output_dir) / f"ReAct_{safe_name}_{safe_subplot}_{idname_proj}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        current_log_file, current_code_dir = QMBagents.create_run_log(
            pdf_name=pdf_name,
            subplotname=subplot_name,
            output_dir=output_dir,
            topic=None,
            issubfile=True,
            iscreatlog=True,
            headname_dir="ReAct",
            headname_log="log_ReAct",
            idname_dir=idname_proj,
            idname_log=idname_code,
        )
        work_dir = Path(current_code_dir)
    else:
        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    agent_ReAct = ReAct_agents.ReAct_agent(
        prompt_dir=prompt_dir,
        output_dir=output_dir,
        current_log_file=current_log_file if task_name else None,
        current_code_dir=str(work_dir),
        topic=topic,
    )
    agent_CodeGenerator = QMBagents.CodeGenerator(
        topic=topic,
        current_log_file=current_log_file if task_name else None,
        current_code_dir=str(work_dir),
    )
    agent_PaperSummerizer = QMBagents.PaperSummerizer(
        topic=topic,
        current_log_file=current_log_file if task_name else None,
    )

    author_token_stats_list = []
    judge_token_stats_list = []
    author_tool_stats_list: list = []
    final_code_file: str | None = None
    exit_reason = "unknown"

    # Phase 1: Generate (or copy when iscodefile=True)
    print("[ReAct] Phase 1: Plan -> Generate -> Small Scale")
    if iscodefile:
        src_path = None
        if code_file:
            src_path = Path(code_file)
        else:
            # output_dir = task_dir; *_check and ReAct_xxx are direct children
            # Resolve from output_dir/*_check/code_LLM_loop1
            base = Path(output_dir)
            check_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.endswith("_check")] if base.exists() else []
            for d in check_dirs:
                for ext in (".py", ".jl"):
                    p = d / f"code_LLM_loop1{ext}"
                    if p.exists():
                        src_path = p
                        break
                if src_path:
                    break
            if not src_path or not src_path.exists():
                report_dict = {
                    "code_file": None,
                    "work_dir": str(work_dir),
                    "exit_reason": "iscodefile_no_source",
                    "author_token_stats": None,
                    "judge_token_stats": None,
                    "author_tool_stats": _aggregate_tool_stats(author_tool_stats_list),
                }
                _save_report(work_dir, report_dict)
                return report_dict
        ext = src_path.suffix.lower()
        if ext not in (".py", ".jl"):
            raise ValueError(f"Unsupported code extension: {ext}")
        dest = work_dir / f"code_ReAct_loop1{ext}"
        shutil.copy2(src_path, dest)
        final_code_file = str(dest)
        print(f"[ReAct] Copied {src_path} to {dest}")
    else:
        pdf_path = task_data.get("pdf_path")
        User_requests = task_data.get("User_requests", "")
        print("[ReAct] Extracting PDF text and generating plan...")
        PDF_info = agent_PaperSummerizer.Extract_tex_text(tex_path=pdf_path, tex_name=pdf_name)
        Plan_text_info = agent_PaperSummerizer.Plan_task_by_text(
            user_model=user_model_planner,
            subplot_name=subplot_name,
            PDF_info=PDF_info,
            User_requests=User_requests,
            iscaltoken=True,
        )
        author_token_stats_list.append(Plan_text_info.get("token_stats"))
        print(f"[ReAct] Plan generated for {subplot_name}.")

        if isauthorrag:
            print("[ReAct] Retrieving knowledge by plan...")
            Query_info = await agent_CodeGenerator.Retrieve_knowledge_by_plan_mcp_use(
                user_model=user_model_coder,
                User_requests=User_requests,
                Plan_info=Plan_text_info,
                iscaltoken=True,
                iscaltool=True,
            )
            author_token_stats_list.append(Query_info.get("token_stats"))
            author_tool_stats_list.append(Query_info.get("tool_stats"))
            query_dir = Path(Query_info.get("query_dir", str(work_dir / "query")))
            summary_path = query_dir / "query_summary.jsonl"
            if not summary_path.exists() or summary_path.stat().st_size == 0:
                print("[ReAct] Summary not saved; aborting.")
                report_dict = {
                    "code_file": None,
                    "work_dir": str(work_dir),
                    "exit_reason": "retrieve_no_summary",
                    "author_token_stats": _aggregate_token_stats(author_token_stats_list),
                    "judge_token_stats": _aggregate_token_stats(judge_token_stats_list),
                    "author_tool_stats": _aggregate_tool_stats(author_tool_stats_list),
                }
                _save_report(work_dir, report_dict)
                return report_dict
            print(f"[ReAct] Knowledge retrieved to {query_dir}")

        print("[ReAct] Generating code by plan...")
        gen_out = agent_ReAct.Generate_code_ReAct_by_plan(
            user_model=user_model_coder,
            Plan_info=Plan_text_info,
            iscaltoken=True,
        )
        author_token_stats_list.append(gen_out.get("token_stats"))
        code_fullparas = gen_out.get("code_file")
        if not code_fullparas:
            print("[ReAct] Code generation failed.")
            report_dict = {
                "code_file": None,
                "work_dir": str(work_dir),
                "exit_reason": "generate_failed",
                "author_token_stats": _aggregate_token_stats(author_token_stats_list),
                "judge_token_stats": _aggregate_token_stats(judge_token_stats_list),
                "author_tool_stats": _aggregate_tool_stats(author_tool_stats_list),
            }
            _save_report(work_dir, report_dict)
            return report_dict

        print("[ReAct] Generating small-scale code...")
        small_out = agent_CodeGenerator.Generate_code_small_scale(
            user_model=user_model_coder,
            code_file=code_fullparas,
            save_name="code_ReAct_loop1",
            iscaltoken=True,
        )
        author_token_stats_list.append(small_out.get("token_stats"))
        final_code_file = small_out.get("code_file")
        print(f"[ReAct] Small-scale code saved to {final_code_file}")

    if not final_code_file or not Path(final_code_file).exists():
        report_dict = {
            "code_file": final_code_file,
            "work_dir": str(work_dir),
            "exit_reason": "phase1_failed",
            "author_token_stats": _aggregate_token_stats(author_token_stats_list),
            "judge_token_stats": _aggregate_token_stats(judge_token_stats_list),
            "author_tool_stats": _aggregate_tool_stats(author_tool_stats_list),
        }
        _save_report(work_dir, report_dict)
        return report_dict

    # Phase 2: Execute -> Suggest -> Repair loop
    loop_index = 1
    code_base = "code_ReAct_loop"
    print(f"[ReAct] Phase 2: Execute -> Suggest -> Repair loop (max {max_iter_ReAct} iterations)")

    while True:
        print(f"[ReAct] Loop {loop_index}/{max_iter_ReAct}: executing code...")
        exec_result = agent_ReAct.Execute_code_ReAct(final_code_file)
        code_file_name = Path(final_code_file).name
        _append_execute_report(work_dir, exec_result, code_file_name, is_first=(loop_index == 1))
        code_content = Path(final_code_file).read_text(encoding="utf-8")

        if exec_result.get("exitcode") == 0:
            exit_reason = "exec_success"
            print(f"[ReAct] Execution succeeded (exitcode=0) at loop {loop_index}, exiting.")
            break

        if loop_index >= max_iter_ReAct:
            exit_reason = "max_iter"
            print(f"[ReAct] Reached max_iter ({max_iter_ReAct}) after execute, exiting before suggest.")
            break

        print(f"[ReAct] Loop {loop_index}: generating repair suggestions...")
        suggest_out = agent_ReAct.Generate_repair_suggest_by_execution_result(
            user_model=user_model_suggest,
            execution_result=exec_result,
            code_content=code_content,
            tag_name_for_save=f"loop{loop_index+1}",
            iscaltoken=True,
        )
        repair_suggestions = suggest_out.get("repair_suggestions", [])
        if not repair_suggestions:
            print(f"[ReAct] No repair suggestions at loop {loop_index}, copying current code to next loop.")
            next_name = f"{code_base}{loop_index + 1}{Path(final_code_file).suffix}"
            next_path = work_dir / next_name
            shutil.copy2(final_code_file, next_path)
            final_code_file = str(next_path)
            loop_index += 1
            continue

        tag_name = f"loop{loop_index + 1}"
        suggest_token_this_loop = suggest_out.get("token_stats")
        if isrepairrag:
            print(f"[ReAct] Loop {loop_index}: retrieving knowledge for repair...")
            agent_CodeRepairer = repair_agents.Code_repairer(
                topic=topic,
                output_dir=str(work_dir),
                current_log_file=current_log_file if task_name else None,
            )
            retrieve_result = await agent_CodeRepairer.Retrieve_knowledge_for_repair_mcp_use(
                user_model=user_model_coder,
                tag_query_dir=tag_name,
                tag_name=tag_name,
                concurr_num=concurr_num_retrieve,
                iscaltoken=True,
                iscaltool=True,
            )
            author_token_stats_list.append(retrieve_result.get("token_stats"))
            author_tool_stats_list.append(retrieve_result.get("tool_stats"))
            print(f"[ReAct] Loop {loop_index}: refining repair suggestions by query...")
            refinesuggest_out = agent_CodeRepairer.Refine_repair_suggestions_by_query(
                user_model=user_model_suggest,
                tag_name=tag_name,
                tag_query_dir=tag_name,
                iscaltoken=True,
            )
            suggest_token_this_loop = _aggregate_token_stats([suggest_token_this_loop, refinesuggest_out.get("token_stats")])
        judge_token_stats_list.append(suggest_token_this_loop)

        print(f"[ReAct] Loop {loop_index}: repairing code...")
        repair_out = agent_ReAct.Repair_code_ReAct_by_suggestions(
            user_model=user_model_coder,
            tag_name=tag_name,
            code_original_name=f"{code_base}{loop_index}",
            code_repair_name=f"{code_base}{loop_index + 1}",
            iscaltoken=True,
            tag_query_dir=tag_name if isrepairrag else None,
        )
        author_token_stats_list.append(repair_out.get("token_stats"))
        final_code_file = repair_out.get("code_file")
        if not final_code_file or not Path(final_code_file).exists():
            exit_reason = "repair_failed"
            print(f"[ReAct] Repair failed at loop {loop_index}.")
            break

        print(f"[ReAct] Loop {loop_index} completed. Code saved to {final_code_file}")
        loop_index += 1

    print(f"[ReAct] Workflow finished. exit_reason={exit_reason}, loop_index={loop_index}")
    report_dict = {
        "code_file": final_code_file,
        "work_dir": str(work_dir),
        "exit_reason": exit_reason,
        "loop_index": loop_index,
        "author_token_stats": _aggregate_token_stats(author_token_stats_list),
        "judge_token_stats": _aggregate_token_stats(judge_token_stats_list),
        "author_tool_stats": _aggregate_tool_stats(author_tool_stats_list),
    }
    _save_report(work_dir, report_dict)
    print(f"ReAct report saved at {work_dir / 'report_ReAct.jsonl'}")
    return report_dict


async def main():
    args = parse_args()
    await task_ReAct(
        output_dir=args.output_dir,
        topic=args.topic,
        task_name=args.task_name,
        user_model_planner=args.user_model_planner,
        user_model_coder=args.user_model_coder,
        user_model_suggest=args.user_model_suggest,
        max_iter_ReAct=args.max_iter_ReAct,
        iscodefile=args.iscodefile,
        code_file=args.code_file,
        idname_proj=args.idname_proj,
        idname_code=args.idname_code,
        isauthorrag=args.isauthorrag,
        isrepairrag=args.isrepairrag,
        concurr_num_retrieve=args.concurr_num_retrieve,
    )


if __name__ == "__main__":
    asyncio.run(main())
