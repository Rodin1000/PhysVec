# author: from task to repo
# In this version, LLM will utilize RAG to generate better codes. 
# This program only cover the 'author' step. Added the token/tool call statistics. 
import os
import sys
import asyncio
import json
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Import from this project
from src import QMBagents, OtherTools


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
        description="Run author program to generate code and repository from task descriptions."
    )
    parser.add_argument(
        "--topic",
        default="dmrg",
        help="Topic of the paper to process (e.g., dmrg, nnwf, etc.).",
    )
    parser.add_argument(
        "--task_name",
        default="task_1_1.json",
        help="Name of the task to process (a figure or table in a specific paper).",
    )
    parser.add_argument(
        "--idname_proj",
        default="untagged",
        help="Identifier for the project.",
    )
    parser.add_argument(
        "--idname_code",
        default="untagged",
        help="Identifier for the code.",
    )
    parser.add_argument(
        "--idname_repo",
        default="untagged",
        help="Identifier for the repository.",
    )
    parser.add_argument(
        "--user_model_planner",
        default="deepseek/deepseek-v3.1-terminus",
        help="Model identifier for the planner agent.",
    )
    parser.add_argument(
        "--user_model_coder",
        default="deepseek/deepseek-chat-v3.1",
        help="Model identifier for the coder agent.",
    )
    parser.add_argument(
        "--user_model_repogenerator",
        default="deepseek/deepseek-v3.1-terminus",
        help="Model identifier for the repo generator agent.",
    )
    parser.add_argument(
        "--iscode",
        type=str2bool,
        default=True,
        help="Flag to indicate if code generation is required.",
    )
    parser.add_argument(
        "--isrepo",
        type=str2bool,
        default=True,
        help="Flag to indicate if repository generation is required.",
    )
    parser.add_argument(
        "--code_file",
        default="",
        help="Path to the code file to use for repository generation.",
    )
    parser.add_argument(
        "--user_model_coderefiner",
        default="deepseek/deepseek-v3.1-terminus",
        help="Model identifier for the code refiner agent.",
    )
    parser.add_argument(
        "--user_model_codejudge",
        default="deepseek/deepseek-v3.1-terminus",
        help="Model identifier for the code judge agent.",
    )
    parser.add_argument(
        "--isrefinecode_rules",
        type=str2bool,
        default=True,
        help="Flag to indicate if code refinement is required.",
    )
    parser.add_argument(
        "--max_iter_refinecode_rules",
        type=int,
        default=2,
        help="Maximum iterations for refining code to obey the pre-defined rules.",
    )
    parser.add_argument(
        "--output_dir",
        default="../logs",
        help="Output directory for logs and reports.",
    )
    parser.set_defaults(include_unit=True)
    return parser.parse_args()


def _save_author_report(current_code_dir: str, report_dict: dict) -> str:
    """Save author report to JSONL file."""
    # Use current_code_dir directly
    out_dir = Path(current_code_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report file name
    filename = "report_author.jsonl"
    report_path = out_dir / filename
    
    # Write report file
    pretty_json = json.dumps(report_dict, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(pretty_json + "\n\n")
    
    return str(report_path)


async def author(topic: str, pdf_path: str, pdf_name: str, subplot_name: str, User_requests: str, idname_proj: str, idname_code: str, idname_repo: str, user_model_planner: str, user_model_coder: str, user_model_repogenerator: str, user_model_coderefiner: str, user_model_codejudge: str, max_iter_refinecode_rules: int = 2, output_dir: str = "../logs", output_repo_dir: str = "../Output_repo", topic_for_dir: str = None, issmallscale: bool = False):

    # Generate codefile--------------------------------------------------------------------
    current_code_file, current_code_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir=output_dir, topic=topic_for_dir, issubfile=True, iscreatlog=True, headname_dir="", headname_log="log_code", idname_dir=idname_proj, idname_log=idname_code)
    agent_PaperSummerizer = QMBagents.PaperSummerizer(topic=topic, current_log_file=current_code_file)
    agent_CodeGenerator = QMBagents.CodeGenerator(topic=topic, current_log_file=current_code_file, current_code_dir=current_code_dir)

    # Extract from Tex
    PDF_info = agent_PaperSummerizer.Extract_tex_text(tex_path=pdf_path, tex_name=pdf_name)
    print('TEX text extracted.')

    # Process: Make a plan for a specific subplot
    Plan_text_info = agent_PaperSummerizer.Plan_task_by_text(
        user_model=user_model_planner,
        subplot_name=subplot_name,
        PDF_info=PDF_info,
        User_requests=User_requests,
        iscaltoken=True,
        # Paper_basic_info=Paper_basic_info,
        # Figure_info=Figure_captions_info,
    )
    print(f'Plan by text on {subplot_name} generated.')

    # Process: Retrieve knowledge by plan
    Query_info = await agent_CodeGenerator.Retrieve_knowledge_by_plan_mcp_use(
        user_model=user_model_coder,
        User_requests=User_requests,
        Plan_info=Plan_text_info,
        iscaltoken=True,
        iscaltool=True,
    )
    query_dir = Query_info.get("query_dir", str(Path(current_code_dir) / "query"))
    summary_path = Path(query_dir) / "query_summary.jsonl"
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        print("Summary not saved; skipping code generation.")
        return {"flag_obey": False, "current_code_dir": None}
    print(f'Knowledge retrieved to {query_dir}')

    # Process: Generate code (by MCP or non-MCP workflow)
    Code_info = agent_CodeGenerator.Generate_code_by_plan(
        user_model=user_model_coder,
        Plan_info=Plan_text_info,
        iscaltoken=True,
    )
    print(f'Code of {Code_info["subplot_name"]} saved to {Code_info["code_file"]}')

    # Refine code to obey rules----------------------------------------------------------------
    _, flag_obey, count_iter, FormatCheck_user_model_token_stats, FormatCheck_judge_model_token_stats = await agent_CodeGenerator.FormatCheck_code_by_rules(
        user_model=user_model_coderefiner,
        judge_model=user_model_codejudge,
        code_file=Code_info["code_file"],
        max_iter=max_iter_refinecode_rules,
        iscaltoken=True,
        isjudgeinfo=True,
    )
    
    # Collect statistics (before repo generation, so available even if flag_obey==False)
    plan_token = Plan_text_info.get("token_stats") or {}
    query_token = Query_info.get("token_stats") or {}
    code_token = Code_info.get("token_stats") or {}
    formatcheck_user_token = FormatCheck_user_model_token_stats or {}
    
    # Aggregate author_token_stats (User Tokens) - without repo_token for now
    author_total_input = (
        plan_token.get("input_tokens", 0) +
        query_token.get("input_tokens", 0) +
        code_token.get("input_tokens", 0) +
        formatcheck_user_token.get("input_tokens", 0)
    )
    author_total_output = (
        plan_token.get("output_tokens", 0) +
        query_token.get("output_tokens", 0) +
        code_token.get("output_tokens", 0) +
        formatcheck_user_token.get("output_tokens", 0)
    )
    author_total = author_total_input + author_total_output
    
    author_token_stats = {
        "input_tokens": author_total_input,
        "output_tokens": author_total_output,
        "total": author_total
    } if author_total > 0 else None
    
    # Collect judge_token_stats
    judge_token_stats = FormatCheck_judge_model_token_stats
    
    # Collect author_tool_stats
    author_tool_stats = Query_info.get("tool_stats")
    
    if flag_obey==False:
        # Save report with flag_obey and statistics (without repo)
        report_dict = {
            "flag_obey": False,
            "count_iter": count_iter,
            "author_token_stats": author_token_stats,
            "judge_token_stats": judge_token_stats,
            "author_tool_stats": author_tool_stats
        }
        report_path = _save_author_report(current_code_dir, report_dict)
        print('Code does not obey the pre-defined rules. Report saved.')
        return {"flag_obey": False, "current_code_dir": current_code_dir}
    else:
        print('Code obeys the pre-defined rules.')

    repo_code_file = Code_info["code_file"]
    if issmallscale:
        code_path = Path(Code_info["code_file"])
        ext = code_path.suffix
        fullparas_path = code_path.parent / f"code_LLM_fullparas{ext}"
        code_path.rename(fullparas_path)
        small_scale_info = agent_CodeGenerator.Generate_code_small_scale(
            user_model=user_model_coder,
            code_file=str(fullparas_path),
            save_name="code_LLM_loop1",
            iscaltoken=True,
        )
        small_code_token = small_scale_info.get("token_stats") or {}
        author_total_input += small_code_token.get("input_tokens", 0)
        author_total_output += small_code_token.get("output_tokens", 0)
        author_total = author_total_input + author_total_output
        author_token_stats = {"input_tokens": author_total_input, "output_tokens": author_total_output, "total": author_total} if author_total > 0 else author_token_stats
        Code_info["code_file"] = small_scale_info.get("code_file")
        repo_code_file = small_scale_info.get("code_file") or str(fullparas_path)

    # Generate repository-------------------------------------------------------------------
    current_repo_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir=output_repo_dir, topic=topic_for_dir, issubfile=True, iscreatlog=False, headname_dir="repo", idname_dir=idname_repo)
    current_repo_file, _ = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir=output_dir, topic=topic_for_dir, issubfile=True, iscreatlog=True, headname_dir="", headname_log="log_repo", idname_dir=idname_proj, idname_log=idname_repo)
    agent_RepoGenerator = QMBagents.RepoGenerator(topic=topic, current_log_file=current_repo_file)

    Repo_info = await agent_RepoGenerator.Generate_repo_by_codefile_hardcode(
        user_model=user_model_repogenerator,
        code_file=repo_code_file,
        fs_target_dir=current_repo_dir,
        iscaltoken=True,
    )
    print(f'Repository of {Code_info["subplot_name"]} generated at {Repo_info["repo_dir"]}')

    # Update author_token_stats to include repo_token
    repo_token = Repo_info.get("token_stats") or {}
    author_total_input += repo_token.get("input_tokens", 0)
    author_total_output += repo_token.get("output_tokens", 0)
    author_total = author_total_input + author_total_output
    
    author_token_stats = {
        "input_tokens": author_total_input,
        "output_tokens": author_total_output,
        "total": author_total
    } if author_total > 0 else None

    # Generate report-------------------------------------------------------------------
    report_dict = {
        "code_file": Code_info["code_file"],
        "repo_dir": Repo_info["repo_dir"],
        "flag_obey": flag_obey,
        "count_iter": count_iter,
        "author_token_stats": author_token_stats,
        "judge_token_stats": judge_token_stats,
        "author_tool_stats": author_tool_stats
    }
    
    report_path = _save_author_report(current_code_dir, report_dict)
    print(f'Author report saved at {report_path}')
    
    # Return flag_obey and current_code_dir
    return {"flag_obey": flag_obey, "current_code_dir": current_code_dir}


async def task_author(args):
    task_data = OtherTools.load_task_parameters(args.topic, args.task_name)
    
    result = await author(
        topic=args.topic,
        pdf_path=task_data.get('pdf_path'),
        pdf_name=task_data.get('pdf_name'),
        subplot_name=task_data.get('subplot_name'),
        User_requests=task_data.get('User_requests'),
        idname_proj=args.idname_proj,
        idname_code=args.idname_code,
        idname_repo=args.idname_repo,
        user_model_planner=args.user_model_planner,
        user_model_coder=args.user_model_coder,
        user_model_repogenerator=args.user_model_repogenerator,
        user_model_coderefiner=args.user_model_coderefiner,
        user_model_codejudge=args.user_model_codejudge,
        max_iter_refinecode_rules=args.max_iter_refinecode_rules,
        output_dir=args.output_dir,
        output_repo_dir=args.output_repo_dir,
        topic_for_dir=None,
        issmallscale=getattr(args, "issmallscale", False),
    )
    
    return result


async def main():
    args = parse_args()
    await task_author(args)


if __name__ == "__main__":
    asyncio.run(main())