# judge_code: verify and grade code
import os
import sys
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Import from this project
from src import QMBagents


async def judge_integration(pdf_name: str, subplot_name: str, repo_dir: str, idname: str, upper_num: int, level_num: int, user_model_codeintegrator: str, log_dir: str, isguidelines: bool = True, isintecode: bool = True, isexecode: bool = True):

    # Create log and sandbox directories-------------------------------------------------
    current_log_file, current_log_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir=log_dir, issubfile=True, iscreatlog=True, headname="judge_integration", isidname=True, idname=idname)
    current_sandbox_integration_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../CodeIntegrator_sandbox", issubfile=True, iscreatlog=False, headname="integrate_run_functions", isidname=True, idname=idname)

    # Define agents-----------------------------------------------------------------------
    agent_CodeIntegrator = QMBagents.CodeIntegrator(current_log_file=current_log_file)


    # Generate integration guidelines-----------------------------------------------------
    if isguidelines:
        guideline_info = await agent_CodeIntegrator.Generate_integration_guidelines(
            user_model=user_model_codeintegrator,
            current_sandbox_dir=current_sandbox_integration_dir,
            repo_dir=repo_dir,
            upper_num=upper_num,
        )
        print(f'integration guidelines generated at {current_sandbox_integration_dir}. ')

    # Generate integration code----------------------------------------------------------
    if isintecode:
        integration_code_info = await agent_CodeIntegrator.Generate_integration_code(
            user_model=user_model_codeintegrator,
            current_sandbox_dir=current_sandbox_integration_dir,
            repo_dir=repo_dir,
            upper_num=level_num,
        )
        print(f'integration code generated at {current_sandbox_integration_dir}. ')


    # Execute integration code and evaluate----------------------------------------------
    if isexecode:
        integration_execution_output = agent_CodeIntegrator.Integrate_code_single_run_execution(
            user_model=user_model_codeintegrator,
            current_sandbox_dir=current_sandbox_integration_dir,
            upper_num=upper_num,
        )
        print(f'integration code executed and evaluated at {current_sandbox_integration_dir}. ')