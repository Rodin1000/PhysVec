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
from src import QMBagents, OtherTools, author_v3, unittest_v3, integtest_v3, rubrics_v3


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

    # for author_v3
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

    # for unittest_v3
    parser.add_argument(
        "--CodeVerifier_library_dir",
        default="../CodeVerifier_library_compact",
        help="Path to the CodeVerifier library directory.",
    )
    parser.add_argument(
        "--idname_unittest",
        default="untagged",
        help="Identifier for the unittest.",
    )
    parser.add_argument(
        "--upper_num_gencode_unittest",
        type=int,
        default=2,
        help="Number of verify code generations in this unittest.",
    )
    parser.add_argument(
        "--upper_num_exec_unittest",
        type=int,
        default=2,
        help="Number of code executions in this unittest.",
    )
    parser.add_argument(
        "--user_model_codeverifier",
        default="deepseek/deepseek-v3.1-terminus",
        help="LLM for this unittest workflow.",
    )
    parser.add_argument(
        "--concurr_num_unittest",
        type=int,
        default=10,
        help="Number of concurrent code generations and executions in this unittest.",
    )
    
    # for integtest_v3
    parser.add_argument(
        "--idname_integtest",
        default="untagged",
        help="Identifier for the integration test.",
    )
    parser.add_argument(
        "--upper_num_integtest",
        type=int,
        default=2,
        help="Number of run functions to be tested in this integtest.",
    )
    parser.add_argument(
        "--level_num_integtest",
        type=int,
        default=2,
        help="Number of levels to be re-constructed for each run function.",
    )
    parser.add_argument(
        "--upper_num_execode_integtest",
        type=int,
        default=2,
        help="Number of code executions for each run function in this integtest.",
    )
    parser.add_argument(
        "--user_model_codeintegrator",
        default="deepseek/deepseek-v3.1-terminus",
        help="User model for the code integrator.",
    )
    parser.add_argument(
        "--concurr_num_integtest",
        type=int,
        default=10,
        help="Number of concurrent code generations and executions in this integtest.",
    )

    # for rubrics_v3
    parser.add_argument(
        "--idname_rubrics",
        default="untagged",
        help="Identifier for the rubrics.",
    )
    parser.add_argument(
        "--user_model_rubricsgrader",
        default="deepseek/deepseek-v3.1-terminus",
        help="User model for the rubrics grader.",
    )

    parser.set_defaults(include_unit=True)
    return parser.parse_args()



async def main():
    args = parse_args()

    # workflow: arthor, generating code and repo
    Code_info, Repo_info = await author_v3.task_author(args)

    # workflow: unittest, verifying the generated code
    await unittest_v3.task_unittest(args, Repo_info.get("repo_dir"))

    # workflow: integtest, integration testing the generated repo
    await integtest_v3.task_integtest(args, Repo_info.get("repo_dir"))

    # workflow: rubrics, grading the generated code by rubrics
    await rubrics_v3.task_rubrics(args, Code_info.get("code_file"))

    print(f"task {args.task_name} workflow completed.")


if __name__ == "__main__":
    asyncio.run(main())