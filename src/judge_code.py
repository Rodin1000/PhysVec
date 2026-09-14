# judge_code: verify and grade code
import os
import sys
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Import from this project
from src import QMBagents


async def judge_code(pdf_name: str, subplot_name: str, repo_dir: str, CodeVerifier_library_dir: str, idname: str, upper_num: int, ismcp: bool, isexec: bool, user_model_codeverifier: str, log_dir: str):

    # Create log and sandbox directories-------------------------------------------------
    current_log_file, current_log_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir=log_dir, issubfile=True, iscreatlog=True, headname="judge", isidname=True, idname=idname)
    current_sandbox_single_repocode_dir = QMBagents.create_run_log(pdf_name=pdf_name, subplotname=subplot_name, output_dir="../CodeVerifier_sandbox", issubfile=True, iscreatlog=False, headname="verify_single_repocode", isidname=True, idname=idname)

    # Define agents-----------------------------------------------------------------------
    agent_CodeVerifier = QMBagents.CodeVerifier(current_log_file=current_log_file)


    # Process: Verify code----------------------------------------------------------------
    Verify_single_repocode_info = await agent_CodeVerifier.Verify_code_single_repocode_generation(
        user_model=user_model_codeverifier,
        CodeVerifier_library_dir=CodeVerifier_library_dir,
        current_sandbox_dir=current_sandbox_single_repocode_dir,
        repo_dir=repo_dir,
        upper_num=upper_num,
        ismcp=ismcp,
    )
    print(f'single repocode verify code generated at {current_sandbox_single_repocode_dir}. ')

    if isexec:
        Single_repocode_judge_info = agent_CodeVerifier.Verify_code_single_repocode_execution(
            user_model=user_model_codeverifier,
            current_sandbox_dir=current_sandbox_single_repocode_dir,
            upper_num=upper_num,
        )
        print(f'single repocode verify code executed, results are stored in {agent_CodeVerifier.current_log_file}. ')





