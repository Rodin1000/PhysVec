# coderepair_v4: generate repair suggestions by verification results
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
from src import QMBagents, repair_agents, OtherTools

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
        description="Run repair program to generate repair suggestions by verification results."
    )
    parser.add_argument(
        "--topic",
        default="dmrg",
        help="Name of the topic to process (e.g., dmrg, nnwf, qcmb).",
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
        "--idname_repair",
        default="untagged",
        help="Identifier for the repair process.",
    )
    parser.add_argument(
        "--suggest_model",
        default="deepseek/deepseek-v3.1-terminus",
        help="LLM model for generating repair suggestions.",
    )
    parser.add_argument(
        "--author_model",
        default="deepseek/deepseek-v3.1-terminus",
        help="LLM model for knowledge retrieval.",
    )
    parser.add_argument(
        "--tag_name",
        default="untagged",
        help="Tag name to identify report files (e.g., 'test2').",
    )
    parser.add_argument(
        "--tag_query_dir",
        default="untagged",
        help="Tag name for query directory (e.g., 'test2').",
    )
    parser.add_argument(
        "--concurr_num_retrieve",
        type=int,
        default=5,
        help="Number of concurrent retrieval tasks.",
    )
    parser.add_argument(
        "--issuggest",
        type=str2bool,
        default=True,
        help="Whether to generate repair suggestions (Step 2).",
    )
    parser.add_argument(
        "--isretrieve",
        type=str2bool,
        default=True,
        help="Whether to retrieve knowledge for repair suggestions (Step 3).",
    )
    parser.add_argument(
        "--isrepair",
        type=str2bool,
        default=False,
        help="Whether to repair code by suggestions (Step 4).",
    )
    parser.add_argument(
        "--code_original_name",
        default="code_LLM",
        help="Name of the original code file (without extension).",
    )
    parser.add_argument(
        "--code_repair_name",
        default="code_repair",
        help="Name for the repaired code file (without extension).",
    )
    parser.add_argument(
        "--judge_model",
        default="deepseek/deepseek-v3.1-terminus",
        help="LLM model for judging code format rules.",
    )
    parser.add_argument(
        "--max_iter_refinecode_rules",
        type=int,
        default=5,
        help="Maximum number of iterations for code format refinement.",
    )
    parser.set_defaults(include_unit=True)
    return parser.parse_args()


async def repair_code(topic: str, pdf_name: str, subplot_name: str, idname_proj: str, idname_repair: str, suggest_model: str, author_model: str, judge_model: str, tag_name: str, tag_query_dir: str, concurr_num_retrieve: int = 5, issuggest: bool = True, isretrieve: bool = True, isrepair: bool = False, code_original_name: str = "code_LLM", code_repair_name: str = "code_LLM_repair", max_iter_refinecode_rules: int = 5):

    # Check: at least one of issuggest or isretrieve must be True
    if not issuggest and not isretrieve and not isrepair:
        raise ValueError("At least one of --issuggest or --isretrieve or --isrepair must be True.")

    # Step 1: Create a repair_agents instance
    current_repair_file, current_repair_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../logs", topic=topic, issubfile=True, iscreatlog=True, headname_dir = "", headname_log="log_repair", idname_dir=idname_proj, idname_log=idname_repair)

    repair_agent = repair_agents.Code_repairer(
        topic=topic,
        current_log_file=current_repair_file,
        output_dir=current_repair_dir
    )

    # Step 2: Generate repair suggestions by verification results
    if issuggest:
        repair_suggestions = repair_agent.Generate_repair_suggest_by_verification_results(
            user_model=suggest_model,
            tag_name=tag_name,
        )
        print(f'Repair suggestions generated at {current_repair_dir}. ')

    # Step 3: Retrieve knowledge for repair suggestions
    if isretrieve:
        retrieve_result = await repair_agent.Retrieve_knowledge_for_repair_mcp_use(
            user_model=author_model,
            tag_query_dir=tag_query_dir,
            tag_name=tag_name,
            concurr_num=concurr_num_retrieve
        )
        print(f'Knowledge retrieval completed at {retrieve_result.get("query_base_dir")}. ')

    # Step 4: Repair code by suggestions
    if isrepair:
        repair_result = repair_agent.Repair_code_by_suggestions(
            user_model=author_model,
            tag_name=tag_name,
            tag_query_dir=tag_query_dir,
            code_original_name=code_original_name,
            code_repair_name=code_repair_name
        )
        print(f'Code repair completed. Repaired code saved to {repair_result.get("code_file")}. ')

        # format check
        agent_CodeGenerator = QMBagents.CodeGenerator(topic=topic, current_log_file=current_repair_file, current_code_dir=current_repair_dir)
        _, flag_obey, _ = await agent_CodeGenerator.FormatCheck_code_by_rules(
                user_model=author_model,
                judge_model=judge_model,
                code_file=repair_result["code_file"],
                max_iter=max_iter_refinecode_rules,
                code_backup_name="code_LLM_" + tag_query_dir + "_origin",
                code_save_name=code_repair_name,
                tag_name=tag_query_dir
            )
        if flag_obey==False:
            raise ValueError("Code does not obey the pre-defined rules.")
        else:
            print('Code obeys the pre-defined rules.')

        # repo generation
        current_repo_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../Output_repo", topic=topic, issubfile=True, iscreatlog=False, headname_dir="repo_repair", idname_dir=tag_name)

        agent_RepoGenerator = QMBagents.RepoGenerator(topic=topic, current_log_file=current_repair_file)
        Repo_info = await agent_RepoGenerator.Generate_repo_by_codefile_hardcode(
                user_model=author_model,
                code_file=repair_result["code_file"],
                fs_target_dir=current_repo_dir,
            )
        print(f'Repository of repaired code generated at {Repo_info["repo_dir"]}')


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
        tag_name=args.tag_name,
        tag_query_dir=args.tag_query_dir,
        concurr_num_retrieve=args.concurr_num_retrieve,
        issuggest=args.issuggest,
        isretrieve=args.isretrieve,
        isrepair=args.isrepair,
        code_original_name=args.code_original_name,
        code_repair_name=args.code_repair_name,
        max_iter_refinecode_rules=args.max_iter_refinecode_rules,
    )

if __name__ == "__main__":
    asyncio.run(main())