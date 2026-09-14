# author: from task to repo
# In this version, LLM will utilize RAG to generate better codes. 
# This program only cover the 'author' step.
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
from src import QMBagents, OtherTools, preverify_agents


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
        description="Run codeverifier program to generate and refine verifier code for QMB systems."
    )
    parser.add_argument(
        "--topic",
        default="nnwf",
        help="Topic of the system to process (e.g., nnwf, dmrg, etc.).",
    )
    parser.add_argument(
        "--type_name",
        default="spin0.5-RBM",
        help="Type name of the system (e.g., spin0.5-RBM, boson, etc.).",
    )
    parser.add_argument(
        "--idname_proj",
        default="PROJ",
        help="Identifier for the project.",
    )
    parser.add_argument(
        "--idname_codeverifier",
        default="IDNAME",
        help="Identifier for the codeverifier logs.",
    )
    parser.add_argument(
        "--user_model",
        default="qwen/qwen3-235b-a22b-2507",
        help="LLM model identifier for all steps.",
    )
    parser.add_argument(
        "--concurr_num",
        type=int,
        default=10,
        help="Number of concurrent tasks for Prepare and Generate steps.",
    )
    parser.add_argument(
        "--query_num",
        type=int,
        default=5,
        help="Number of queries for Prepare step.",
    )
    parser.add_argument(
        "--num_check",
        type=int,
        default=3,
        help="Number of check iterations for Generate step.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10,
        help="Maximum iterations for Refine step.",
    )
    parser.add_argument(
        "--program_timeout",
        type=int,
        default=180,
        help="Timeout in seconds for program execution in Refine step.",
    )
    parser.add_argument(
        "--refine_query_num",
        type=int,
        default=5,
        help="Number of queries for Refine2 step (RAG-based refinement).",
    )
    parser.add_argument(
        "--do_cold_init",
        type=str2bool,
        default=True,
        help="Whether to execute Cold_init step.",
    )
    parser.add_argument(
        "--do_prepare",
        type=str2bool,
        default=True,
        help="Whether to execute Prepare step.",
    )
    parser.add_argument(
        "--do_generate",
        type=str2bool,
        default=True,
        help="Whether to execute Generate step.",
    )
    parser.add_argument(
        "--do_combine",
        type=str2bool,
        default=True,
        help="Whether to execute Combine step.",
    )
    parser.add_argument(
        "--do_refine",
        type=str2bool,
        default=True,
        help="Whether to execute Refine step.",
    )
    return parser.parse_args()


async def codeverifier(
    topic: str,
    type_name: str,
    idname_proj: str,
    idname_codeverifier: str,
    user_model: str,
    concurr_num: int = 10,
    query_num: int = 5,
    num_check: int = 3,
    max_iterations: int = 10,
    program_timeout: int = 180,
    refine_query_num: int = 5,
    do_cold_init: bool = True,
    do_prepare: bool = True,
    do_generate: bool = True,
    do_combine: bool = True,
    do_refine: bool = True,
):
    """
    Run the codeverifier workflow to generate and refine verifier code.
    
    Args:
        topic: Topic of the system (e.g., nnwf, dmrg)
        type_name: Type name of the system (e.g., spin0.5-RBM, boson)
        idname_proj: Identifier for the project
        idname_codeverifier: Identifier for the codeverifier logs
        user_model: LLM model identifier for all steps
        concurr_num: Number of concurrent tasks for Prepare and Generate steps
        query_num: Number of queries for Prepare step
        num_check: Number of check iterations for Generate step
        max_iterations: Maximum iterations for Refine step
        program_timeout: Timeout in seconds for program execution in Refine step
        refine_query_num: Number of queries for Refine2 step (RAG-based refinement)
        do_cold_init: Whether to execute Cold_init step
        do_prepare: Whether to execute Prepare step
        do_generate: Whether to execute Generate step
        do_combine: Whether to execute Combine step
        do_refine: Whether to execute Refine step
    """
    print("=" * 80)
    print("CodeVerifier_v1 started.")
    print("=" * 80)

    # Create log file
    current_code_file, current_code_dir = QMBagents.create_run_log(
        pdf_name="TEST",
        subplotname="TEST",
        output_dir="../logs",
        topic="codeverifier",
        issubfile=True,
        iscreatlog=True,
        headname_dir="",
        headname_log="log_codeverifier",
        idname_dir=idname_proj,
        idname_log=idname_codeverifier,
    )
    
    # Create agent
    agent_codeverifier_generator = preverify_agents.CodeVerifier_library_generator(
        topic=topic,
        type_name=type_name,
        current_log_file=current_code_file,
    )

    # Execute steps based on flags
    if do_cold_init:
        agent_codeverifier_generator.Cold_init_single_type()
        print("=" * 80)
        print("Cold_init step completed.")
        print("=" * 80)

    if do_prepare:
        await agent_codeverifier_generator.Prepare_QueryMaterial_mcp_use_concurr_single_type(
            user_model=user_model,
            concurr_num=concurr_num,
            query_num=query_num,
        )
        print("=" * 80)
        print("Prepare step completed.")
        print("=" * 80)

    if do_generate:
        await agent_codeverifier_generator.Generate_VerifierCode_mcp_use_concurr_single_type(
            user_model=user_model,
            concurr_num=concurr_num,
            num_check=num_check,
        )
        print("=" * 80)
        print("Generate step completed.")
        print("=" * 80)

    if do_combine:
        # await agent_codeverifier_generator.Combine_VerifierCode_mcp_use_single_type(
        #     user_model=user_model,
        # )
        await agent_codeverifier_generator.Combine_VerifierCode_mcp_use_robust_single_type(
            user_model=user_model,
        )
        print("=" * 80)
        print("Combine step completed.")
        print("=" * 80)

    if do_refine:
        # await agent_codeverifier_generator.Refine_VerifierCode_mcp_use_single_type(
        #     user_model=user_model,
        #     max_iterations=max_iterations,
        #     program_timeout=program_timeout,
        #     refine_query_num=refine_query_num,
        # )
        await agent_codeverifier_generator.Refine_VerifierCode_freqRAG_mcp_use_single_type(
            user_model=user_model,
            max_iterations=max_iterations,
            program_timeout=program_timeout,
            refine_query_num=refine_query_num,
        )   
        print("=" * 80)
        print("Refine step completed.")
        print("=" * 80)

    print("=" * 80)
    print("CodeVerifier_v1 completed.")
    print("=" * 80)


async def main():
    args = parse_args()
    await codeverifier(
        topic=args.topic,
        type_name=args.type_name,
        idname_proj=args.idname_proj,
        idname_codeverifier=args.idname_codeverifier,
        user_model=args.user_model,
        concurr_num=args.concurr_num,
        query_num=args.query_num,
        num_check=args.num_check,
        max_iterations=args.max_iterations,
        program_timeout=args.program_timeout,
        refine_query_num=args.refine_query_num,
        do_cold_init=args.do_cold_init,
        do_prepare=args.do_prepare,
        do_generate=args.do_generate,
        do_combine=args.do_combine,
        do_refine=args.do_refine,
    )


if __name__ == "__main__":
    asyncio.run(main())