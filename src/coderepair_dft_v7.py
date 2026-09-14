# coderepair_v7: aligned with author_v7 (output_dir from args, topic_for_dir=None, no is* branches)
import json
import os
import sys
import asyncio
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src import DFT_agents, repair_agents_dft, OtherTools


def _agg_token(stats_list):
    total_in = total_out = 0
    for s in stats_list:
        if s and isinstance(s, dict):
            total_in += s.get("input_tokens", 0) or 0
            total_out += s.get("output_tokens", 0) or 0
    if total_in == 0 and total_out == 0:
        return None
    return {"input_tokens": total_in, "output_tokens": total_out, "total": total_in + total_out}


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
        description="Run repair program to generate repair suggestions by verification results (v7: no is* branches)."
    )
    parser.add_argument(
        "--topic",
        default="dmrg",
        help="Topic for load_task_parameters (e.g., dmrg, nnwf, qcmb).",
    )
    parser.add_argument(
        "--task_name",
        default="task_1_1.json",
        help="Task to process.",
    )
    parser.add_argument(
        "--idname_proj",
        default="untagged",
        help="Identifier for the project.",
    )
    parser.add_argument(
        "--idname_repair",
        default="untagged",
        help="Identifier for the repair process.",
    )
    parser.add_argument(
        "--output_dir",
        default="../logs",
        help="Output directory for logs (user-provided).",
    )
    parser.add_argument(
        "--output_repo_dir",
        default="../Output_repo",
        help="Output directory for repo.",
    )
    parser.add_argument(
        "--suggest_model",
        default="deepseek/deepseek-v3.1-terminus",
        help="LLM for repair suggestions.",
    )
    parser.add_argument(
        "--author_model",
        default="deepseek/deepseek-v3.1-terminus",
        help="LLM for retrieval and repair.",
    )
    parser.add_argument(
        "--tag_name_for_report",
        default="untagged",
        help="Tag to identify which verification reports to read (report_unittest_*_{tag}.jsonl, report_integtest_*_{tag}.jsonl).",
    )
    parser.add_argument(
        "--tag_name",
        default="untagged",
        help="Tag for saving newly generated files (e.g. repair_suggest_{tag}.jsonl) and downstream steps.",
    )
    parser.add_argument(
        "--tag_query_dir",
        default="untagged",
        help="Tag for query directory (often same as tag_name).",
    )
    parser.add_argument(
        "--concurr_num_retrieve",
        type=int,
        default=5,
        help="Concurrent retrieval tasks.",
    )
    parser.add_argument(
        "--code_original_name",
        default="code_LLM",
        help="Original code file stem.",
    )
    parser.add_argument(
        "--code_repair_name",
        default="code_repair",
        help="Repaired code file stem.",
    )
    parser.add_argument(
        "--judge_model",
        default="deepseek/deepseek-v3.1-terminus",
        help="LLM for format judge.",
    )
    parser.add_argument(
        "--max_iter_refinecode_rules",
        type=int,
        default=5,
        help="Max format-check iterations.",
    )
    parser.add_argument(
        "--isonlyexe",
        action="store_true",
        help="Only run fullcode execution, skip Suggest/Retrieve/Repair.",
    )
    parser.set_defaults(include_unit=True)
    return parser.parse_args()


async def repair_code(
    topic: str,
    pdf_name: str,
    subplot_name: str,
    idname_proj: str,
    idname_repair: str,
    suggest_model: str,
    author_model: str,
    judge_model: str,
    tag_name_for_report: str,
    tag_name: str,
    tag_query_dir: str,
    output_dir: str,
    output_repo_dir: str,
    concurr_num_retrieve: int = 5,
    code_original_name: str = "code_LLM",
    code_repair_name: str = "code_repair",
    max_iter_refinecode_rules: int = 5,
    topic_for_dir: str | None = None,
    code_file: str | None = None,
    repo_dir: str | None = None,
    isonlyexe: bool = False,
):
    current_repair_file, current_repair_dir = DFT_agents.create_run_log(
        pdf_name=pdf_name,
        subplotname=subplot_name,
        output_dir=output_dir,
        topic=topic_for_dir,
        issubfile=True,
        iscreatlog=True,
        headname_dir="",
        headname_log="log_repair",
        idname_dir=idname_proj,
        idname_log=idname_repair,
    )
    repair_agent = repair_agents_dft.Code_repairer(
        topic=topic,
        current_log_file=current_repair_file,
        output_dir=current_repair_dir,
    )

    # Step 1: Execute fullcode before Suggest (support .py, .jl, .inp for DFT/ORCA)
    code_path = Path(current_repair_dir) / f"{code_original_name}.py"
    if not code_path.exists():
        code_path = Path(current_repair_dir) / f"{code_original_name}.jl"
    if not code_path.exists():
        code_path = Path(current_repair_dir) / f"{code_original_name}.inp"
    if not code_path.exists():
        raise FileNotFoundError(f"{code_original_name}.py, .jl or .inp not found in {current_repair_dir}")
    exec_result = repair_agent.Execute_fullcode(str(code_path))
    exec_record = {"name": code_path.name, **exec_result}
    (Path(current_repair_dir) / f"report_fullcode_{tag_name_for_report}.jsonl").write_text(json.dumps(exec_record, ensure_ascii=False) + "\n", encoding="utf-8")
    report_exec_path = Path(current_repair_dir) / "report_execute_check.jsonl"
    mode = "w" if not report_exec_path.exists() else "a"
    with open(report_exec_path, mode, encoding="utf-8") as f:
        f.write(json.dumps(exec_record, ensure_ascii=False) + "\n")
    if isonlyexe:
        return {
            "code_file": code_file or str(code_path),
            "repo_dir": repo_dir or "",
            "tag_name": tag_name,
            "author_token_stats": None,
            "suggest_token_stats": None,
            "judge_token_stats": None,
            "repair_tool_stats": None,
            "no_repair_needed": False,
        }
    # Success: exitcode==0, or for ORCA "ran for some time without error" (timeout + running_successfully)
    fullcode_ok = exec_result.get("exitcode") == 0
    if not fullcode_ok and code_path.suffix.lower() == ".inp":
        fullcode_ok = bool(exec_result.get("timeout") and exec_result.get("running_successfully"))
    if fullcode_ok:
        early_code_file = code_file or str(code_path)
        early_repo_dir = repo_dir or ""
        return {
            "code_file": early_code_file or "",
            "repo_dir": early_repo_dir or "",
            "tag_name": tag_name,
            "author_token_stats": None,
            "suggest_token_stats": None,
            "judge_token_stats": None,
            "repair_tool_stats": None,
            "no_repair_needed": True,
        }

    # Step 2: Suggest (always): read reports by tag_name_for_report, save suggest by tag_name
    suggest_out = repair_agent.Generate_repair_suggest_by_verification_results(
        user_model=suggest_model,
        tag_name=tag_name_for_report,
        tag_name_for_save=tag_name,
        iscaltoken=True,
        code_file_stem=code_original_name,
    )
    print(f"Repair suggestions generated at {current_repair_dir}.")

    repair_suggestions = suggest_out.get("repair_suggestions", [])
    if not repair_suggestions:
        # Suggest empty: skip Retrieve+Repair, return unchanged code, no_repair_needed=False so loop continues
        suggest_token_stats = _agg_token([suggest_out.get("token_stats")])
        early_code_file = code_file if code_file else str(code_path)
        early_repo_dir = repo_dir if repo_dir else ""
        return {
            "code_file": early_code_file or "",
            "repo_dir": early_repo_dir or "",
            "tag_name": tag_name,
            "author_token_stats": None,
            "suggest_token_stats": suggest_token_stats,
            "judge_token_stats": None,
            "repair_tool_stats": None,
            "no_repair_needed": False,
        }

    # Step 3: Retrieve (always)
    retrieve_result = await repair_agent.Retrieve_knowledge_for_repair_mcp_use(
        user_model=author_model,
        tag_query_dir=tag_query_dir,
        tag_name=tag_name,
        concurr_num=concurr_num_retrieve,
        iscaltoken=True,
        iscaltool=True,
    )
    print(f"Knowledge retrieval completed at {retrieve_result.get('query_base_dir')}.")

    # Step 3.5: Refine repair_suggestions by query (overwrite repair_suggest file)
    refinesuggest_out = repair_agent.Refine_repair_suggestions_by_query(
        user_model=suggest_model,
        tag_name=tag_name,
        tag_query_dir=tag_query_dir,
        iscaltoken=True,
    )

    # Step 4: Repair + Repo (FormatCheck disabled)
    repair_result = repair_agent.Repair_code_by_suggestions(
        user_model=author_model,
        tag_name=tag_name,
        tag_query_dir=tag_query_dir,
        code_original_name=code_original_name,
        code_repair_name=code_repair_name,
        iscaltoken=True,
    )
    print(f"Code repair completed. Repaired code saved to {repair_result.get('code_file')}.")

    # FormatCheck step disabled
    # agent_CodeGenerator = QMBagents.CodeGenerator(
    #     topic=topic,
    #     current_log_file=current_repair_file,
    #     current_code_dir=current_repair_dir,
    # )
    # _, flag_obey, _, format_user_token, format_judge_token = await agent_CodeGenerator.FormatCheck_code_by_rules(
    #     user_model=author_model,
    #     judge_model=judge_model,
    #     code_file=repair_result["code_file"],
    #     max_iter=max_iter_refinecode_rules,
    #     code_backup_name=f"code_LLM_{tag_query_dir}_origin",
    #     code_save_name=code_repair_name,
    #     tag_name=tag_query_dir,
    #     iscaltoken=True,
    # )
    # if not flag_obey:
    #     raise ValueError("Code does not obey the pre-defined rules.")
    # print("Code obeys the pre-defined rules.")
    format_user_token = None
    format_judge_token = None

    current_repo_dir = DFT_agents.create_run_log(
        pdf_name=pdf_name,
        subplotname=subplot_name,
        output_dir=output_repo_dir,
        topic=topic_for_dir,
        issubfile=True,
        iscreatlog=False,
        headname_dir="repo_repair",
        idname_dir=tag_name,
    )
    agent_RepoGenerator = DFT_agents.RepoGenerator(topic=topic, current_log_file=current_repair_file)
    Repo_info = await agent_RepoGenerator.Generate_repo_by_codefile_hardcode(
        user_model=author_model,
        code_file=repair_result["code_file"],
        fs_target_dir=current_repo_dir,
        iscaltoken=True,
    )
    print(f"Repository of repaired code generated at {Repo_info['repo_dir']}.")

    author_token_stats = _agg_token([
        retrieve_result.get("token_stats"),
        repair_result.get("token_stats"),
        format_user_token,
        Repo_info.get("token_stats"),
    ])
    suggest_token_stats = _agg_token([suggest_out.get("token_stats"), refinesuggest_out.get("token_stats")])
    judge_token_stats = _agg_token([format_judge_token])
    repair_tool_stats = retrieve_result.get("tool_stats")

    return {
        "code_file": repair_result["code_file"],
        "repo_dir": Repo_info["repo_dir"],
        "tag_name": tag_name,
        "author_token_stats": author_token_stats,
        "suggest_token_stats": suggest_token_stats,
        "judge_token_stats": judge_token_stats,
        "repair_tool_stats": repair_tool_stats,
    }


async def task_coderepair(args):
    task_data = OtherTools.load_task_parameters(args.topic, args.task_name)
    return await repair_code(
        topic=args.topic,
        pdf_name=task_data.get("pdf_name"),
        subplot_name=task_data.get("subplot_name"),
        idname_proj=args.idname_proj,
        idname_repair=args.idname_repair,
        suggest_model=args.suggest_model,
        author_model=args.author_model,
        judge_model=args.judge_model,
        tag_name_for_report=args.tag_name_for_report,
        tag_name=args.tag_name,
        tag_query_dir=args.tag_query_dir,
        output_dir=args.output_dir,
        output_repo_dir=args.output_repo_dir,
        concurr_num_retrieve=args.concurr_num_retrieve,
        code_original_name=args.code_original_name,
        code_repair_name=args.code_repair_name,
        max_iter_refinecode_rules=args.max_iter_refinecode_rules,
        topic_for_dir=None,
        code_file=getattr(args, "code_file", None),
        repo_dir=getattr(args, "repo_dir", None),
        isonlyexe=getattr(args, "isonlyexe", False),
    )


async def main():
    args = parse_args()
    task_data = OtherTools.load_task_parameters(args.topic, args.task_name)
    await repair_code(
        topic=args.topic,
        pdf_name=task_data.get("pdf_name"),
        subplot_name=task_data.get("subplot_name"),
        idname_proj=args.idname_proj,
        idname_repair=args.idname_repair,
        suggest_model=args.suggest_model,
        author_model=args.author_model,
        judge_model=args.judge_model,
        tag_name_for_report=args.tag_name_for_report,
        tag_name=args.tag_name,
        tag_query_dir=args.tag_query_dir,
        output_dir=args.output_dir,
        output_repo_dir=args.output_repo_dir,
        concurr_num_retrieve=args.concurr_num_retrieve,
        code_original_name=args.code_original_name,
        code_repair_name=args.code_repair_name,
        max_iter_refinecode_rules=args.max_iter_refinecode_rules,
        topic_for_dir=None,
    )


if __name__ == "__main__":
    asyncio.run(main())
