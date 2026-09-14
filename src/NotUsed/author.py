# author: from task to repo
import os
import sys
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Import from this project
from src import QMBagents



async def author(pdf_path: str, pdf_name: str, subplot_name: str, User_requests: str, idname: str, user_model_planner: str, user_model_coder: str, user_model_repogenerator: str, iscode: bool = True, isrepo: bool = True, code_file: str = ""):

    # Input check--------------------------------------------------------------------------
    if not iscode and not isrepo:
        raise ValueError("At least one of iscode or isrepo must be True.")
    
    # Create log and repo directories-------------------------------------------------
    current_log_file, current_log_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../logs", issubfile=True, iscreatlog=True, headname="total", isidname=True, idname=idname)
    if isrepo:
        current_repo_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../Output_repo", issubfile=True, iscreatlog=False, headname="repo", isidname=True, idname=idname)
   
    # Define agents-----------------------------------------------------------------------
    agent_PaperSummerizer = QMBagents.PaperSummerizer(current_log_file=current_log_file)
    agent_CodeGenerator = QMBagents.CodeGenerator(current_log_file=current_log_file, current_code_dir=current_log_dir)
    if isrepo:
        agent_RepoGenerator = QMBagents.RepoGenerator(current_log_file=current_log_file)
    
    if iscode:
    # Extract from Tex--------------------------------------------------------------------
        PDF_info = agent_PaperSummerizer.Extract_tex_text(tex_path=pdf_path, tex_name=pdf_name)
        print('TEX text extracted.')

    # Process: Make a plan for a specific subplot------------------------------------------------
        Plan_text_info = agent_PaperSummerizer.Plan_task_by_text(
            user_model=user_model_planner,
            subplot_name=subplot_name,
            PDF_info=PDF_info,
            User_requests=User_requests,
            # Paper_basic_info=Paper_basic_info,
            # Figure_info=Figure_captions_info,
        )
        print(f'Plan by text on {subplot_name} generated.')

    # Process: Generate code based on the plan----------------------------------------------------
    # Code_info = agent_CodeGenerator.Generate_code_by_plan(
    #     user_model=common_name,
    #     Plan_info=Plan_text_info,
    #     PDF_info=PDF_info,
    # )
    # print(f'Code of {Code_info["subplot_name"]} saved to {Code_info["code_file"]}')

    # Process: Generate code by MCP---------------------------------------------------------------
        Code_info = await agent_CodeGenerator.Generate_code_by_plan_mcp_use(
            user_model=user_model_coder,
            Plan_info=Plan_text_info,
            PDF_info=PDF_info,
        )
        print(f'Code of {Code_info["subplot_name"]} saved to {Code_info["code_file"]}')


    # Process: Construct repo from code-----------------------------------------------------------
    if isrepo and iscode:
        Repo_info = await agent_RepoGenerator.Generate_repo_by_codefile(
            user_model=user_model_repogenerator,
            code_file=Code_info["code_file"],
            fs_target_dir=current_repo_dir,
        )
        print(f'Repository of {Code_info["subplot_name"]} generated at {Repo_info["repo_dir"]}')
    elif isrepo and not iscode:
        Repo_info = await agent_RepoGenerator.Generate_repo_by_codefile(
            user_model=user_model_repogenerator,
            code_file=code_file, 
            fs_target_dir=current_repo_dir,
        )
        print(f'Repository of {subplot_name} generated at {Repo_info["repo_dir"]}')
