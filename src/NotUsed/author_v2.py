# author: from task to repo
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
    parser.set_defaults(include_unit=True)
    return parser.parse_args()



async def author(pdf_path: str, pdf_name: str, subplot_name: str, User_requests: str, idname_proj: str, idname_code: str, idname_repo: str, user_model_planner: str, user_model_coder: str, user_model_repogenerator: str, iscode: bool = True, isrepo: bool = True, code_file: str = "", user_model_coderefiner: str = "", isrefinecode_rules: bool = True, max_iter_refinecode_rules: int = 2):

    # Input check--------------------------------------------------------------------------
    if not iscode and not isrepo and not isrefinecode_rules:
        raise ValueError("At least one of iscode or isrepo or isrefinecode_rules must be True.")
    
    # Generate codefile--------------------------------------------------------------------
    if iscode:
        current_code_file, current_code_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../logs", issubfile=True, iscreatlog=True, headname_dir="", headname_log="log_code", idname_dir=idname_proj, idname_log=idname_code)
        agent_PaperSummerizer = QMBagents.PaperSummerizer(current_log_file=current_code_file)
        agent_CodeGenerator = QMBagents.CodeGenerator(current_log_file=current_code_file, current_code_dir=current_code_dir)

        # Extract from Tex
        PDF_info = agent_PaperSummerizer.Extract_tex_text(tex_path=pdf_path, tex_name=pdf_name)
        print('TEX text extracted.')

        # Process: Make a plan for a specific subplot
        Plan_text_info = agent_PaperSummerizer.Plan_task_by_text(
            user_model=user_model_planner,
            subplot_name=subplot_name,
            PDF_info=PDF_info,
            User_requests=User_requests,
            # Paper_basic_info=Paper_basic_info,
            # Figure_info=Figure_captions_info,
        )
        print(f'Plan by text on {subplot_name} generated.')

        # Process: Generate code based on the plan
        # Code_info = agent_CodeGenerator.Generate_code_by_plan(
        #     user_model=common_name,
        #     Plan_info=Plan_text_info,
        #     PDF_info=PDF_info,
        # )
        # print(f'Code of {Code_info["subplot_name"]} saved to {Code_info["code_file"]}')

        # Process: Generate code by MCP
        Code_info = await agent_CodeGenerator.Generate_code_by_plan_mcp_use(
            user_model=user_model_coder,
            Plan_info=Plan_text_info,
            PDF_info=PDF_info,
        )
        print(f'Code of {Code_info["subplot_name"]} saved to {Code_info["code_file"]}')

    # Refine code to obey rules----------------------------------------------------------------
    if isrefinecode_rules:
        if iscode:
            await agent_CodeGenerator.Refine_code_obey_rules(
                user_model=user_model_coderefiner,
                code_file=Code_info["code_file"],
                max_iter=max_iter_refinecode_rules,
            )
            print('code refined to obey the pre-defined rules.')

        elif not iscode:
            current_code_file, current_code_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../logs", issubfile=True, iscreatlog=True, headname_dir="", headname_log="log_code", idname_dir=idname_proj, idname_log=idname_code)
            agent_CodeGenerator = QMBagents.CodeGenerator(current_log_file=current_code_file, current_code_dir=current_code_dir)
            await agent_CodeGenerator.Refine_code_obey_rules(
                user_model=user_model_coderefiner,
                code_file=code_file,
                max_iter=max_iter_refinecode_rules,
            )
            print('code refined to obey the pre-defined rules.')


    # Generate repository-------------------------------------------------------------------
    if isrepo:
        current_repo_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../Output_repo", issubfile=True, iscreatlog=False, headname_dir="repo", idname_dir=idname_repo)
        current_repo_file, _ = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../logs", issubfile=True, iscreatlog=True, headname_dir="", headname_log="log_repo", idname_dir=idname_proj, idname_log=idname_repo)
        agent_RepoGenerator = QMBagents.RepoGenerator(current_log_file=current_repo_file)

        if iscode:
            # Repo_info = await agent_RepoGenerator.Generate_repo_by_codefile(
            #     user_model=user_model_repogenerator,
            #     code_file=Code_info["code_file"],
            #     fs_target_dir=current_repo_dir,
            # )
            Repo_info = await agent_RepoGenerator.Generate_repo_by_codefile_hardcode(
                user_model=user_model_repogenerator,
                code_file=Code_info["code_file"],
                fs_target_dir=current_repo_dir,
            )
            print(f'Repository of {Code_info["subplot_name"]} generated at {Repo_info["repo_dir"]}')

        elif not iscode:
            if not code_file:
                raise ValueError("code_file must be provided when iscode is False.")
            # Repo_info = await agent_RepoGenerator.Generate_repo_by_codefile(
            #     user_model=user_model_repogenerator,
            #     code_file=code_file, 
            #     fs_target_dir=current_repo_dir,
            # )
            Repo_info = await agent_RepoGenerator.Generate_repo_by_codefile_hardcode(
                user_model=user_model_repogenerator,
                code_file=code_file, 
                fs_target_dir=current_repo_dir,
            )
            print(f'Repository of {subplot_name} generated at {Repo_info["repo_dir"]}')





async def main():
    args = parse_args()
    task_data = OtherTools.load_task_parameters(args.task_name)
    # print('args loaded:', args) # debug
    await author(
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
        iscode=args.iscode,
        isrepo=args.isrepo,
        code_file=args.code_file,
        user_model_coderefiner=args.user_model_coderefiner,
        isrefinecode_rules=args.isrefinecode_rules,
        max_iter_refinecode_rules=args.max_iter_refinecode_rules,
    )


if __name__ == "__main__":
    asyncio.run(main())