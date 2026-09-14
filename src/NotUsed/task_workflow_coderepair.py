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
from src import QMBagents, OtherTools, unittest_v5, integtest_v5, coderepair_v5


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
        description="Run repair workflow to iteratively repair code until all tests pass."
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
        "--tag_name",
        default="untagged",
        help="Initial tag name to identify report files (e.g., 'test2').",
    )
    parser.add_argument(
        "--code_original_name",
        default="code_LLM",
        help="Name of the original code file (without extension).",
    )
    parser.add_argument(
        "--max_repair_iterations",
        type=int,
        default=10,
        help="Maximum number of repair iterations.",
    )
    # Parameters for coderepair_v5
    parser.add_argument(
        "--suggest_model",
        default="deepseek/deepseek-v3.1-terminus",
        help="LLM model for generating repair suggestions.",
    )
    parser.add_argument(
        "--author_model",
        default="deepseek/deepseek-v3.1-terminus",
        help="LLM model for knowledge retrieval and code repair.",
    )
    parser.add_argument(
        "--judge_model",
        default="deepseek/deepseek-v3.1-terminus",
        help="LLM model for judging code format rules.",
    )
    parser.add_argument(
        "--concurr_num_retrieve",
        type=int,
        default=5,
        help="Number of concurrent retrieval tasks.",
    )
    parser.add_argument(
        "--max_iter_refinecode_rules",
        type=int,
        default=5,
        help="Maximum number of iterations for code format refinement.",
    )
    # Parameters for unittest_v5
    parser.add_argument(
        "--user_model_codeverifier",
        default="deepseek/deepseek-v3.1-terminus",
        help="LLM for this unittest workflow.",
    )
    parser.add_argument(
        "--judge_model_codeverifier",
        default="deepseek/deepseek-v3.1-terminus",
        help="LLM for judging code execution results.",
    )
    parser.add_argument(
        "--sandbox_dir",
        default="../CodeVerifier_sandbox",
        help="Path to the sandbox directory.",
    )
    parser.add_argument(
        "--concurr_num_unittest",
        type=int,
        default=5,
        help="Number of concurrent code generation in this unittest.",
    )
    # Parameters for integtest_v5
    parser.add_argument(
        "--user_model_codeintegrator",
        default="deepseek/deepseek-v3.1-terminus",
        help="User model for the code integrator.",
    )
    parser.add_argument(
        "--judge_model_codeintegrator",
        default="deepseek/deepseek-v3.1-terminus",
        help="Judge model for evaluating integration code execution results.",
    )
    parser.add_argument(
        "--sandbox_dir_integtest",
        default="../CodeIntegrator_sandbox",
        help="Path to the sandbox directory for integration tests.",
    )
    parser.add_argument(
        "--concurr_num_integtest",
        type=int,
        default=5,
        help="Number of concurrent integration tests to run.",
    )
    parser.set_defaults(include_unit=True)
    return parser.parse_args()


def find_code_file(work_dir: Path, code_name: str) -> Path:
    """Find code file (supports .py and .jl)"""
    for ext in [".py", ".jl"]:
        code_file = work_dir / f"{code_name}{ext}"
        if code_file.exists():
            return code_file
    raise FileNotFoundError(f"Code file '{code_name}' not found in {work_dir}")


def find_report_files(work_dir: Path, tag_name: str) -> tuple[Path, Path]:
    """Find unittest and integtest report files"""
    unittest_report = next(work_dir.glob(f"report_unittest_*_{tag_name}.jsonl"), None)
    integtest_report = next(work_dir.glob(f"report_integtest_*_{tag_name}.jsonl"), None)
    return unittest_report, integtest_report


def create_repair_args(base_args, tag_name_repair, tag_name_for_report, code_original_name, code_repair_name):
    """Create arguments for repair step"""
    args = argparse.Namespace()
    # Copy base arguments
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    # Set repair-specific parameters
    args.tag_name_for_report = tag_name_for_report
    args.tag_name = tag_name_repair
    args.tag_query_dir = tag_name_repair
    args.idname_repair = tag_name_repair
    args.code_original_name = code_original_name
    args.code_repair_name = code_repair_name
    args.issuggest = True
    args.isretrieve = True
    args.isrepair = True
    return args


def create_unittest_args(base_args, tag_name_repair, code_file, repo_dir):
    """Create arguments for unittest step"""
    args = argparse.Namespace()
    # Copy base arguments
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    # Set unittest-specific parameters
    args.code_file = code_file
    args.repo_dir = repo_dir
    args.idname_unittest = tag_name_repair
    args.isreport = True
    # Set default values to enable all necessary functions
    args.upper_num_gencode = 0
    args.upper_num_exec = 0
    args.isgencode = True
    args.ismcp = False
    args.isexec = True
    return args


def create_integtest_args(base_args, tag_name_repair, repo_dir):
    """Create arguments for integtest step"""
    args = argparse.Namespace()
    # Copy base arguments
    for key, value in vars(base_args).items():
        setattr(args, key, value)
    # Set integtest-specific parameters
    args.repo_dir = repo_dir
    args.idname_integtest = tag_name_repair
    args.isreport = True
    # Set default values to enable all necessary functions
    args.upper_num_integcode = 0
    args.level_num = 0
    args.upper_num_execode = 0
    args.isintegcode = True
    args.isexecode = True
    return args


def check_reports_pass(work_dir: Path, tag_name: str) -> tuple[bool, dict]:
    """
    Check if unittest and integtest reports for the given tag_name both pass.
    
    Returns:
        (all_pass: bool, report_info: dict)
        all_pass: True if both reports have correct_ratio == 1.0
        report_info: Contains paths and correct_ratio information for both reports
    """
    unittest_report, integtest_report = find_report_files(work_dir, tag_name)
    
    report_info = {
        "unittest_report": str(unittest_report) if unittest_report else None,
        "integtest_report": str(integtest_report) if integtest_report else None,
        "unittest_correct_ratio": None,
        "integtest_ave_correct_ratio": None
    }
    
    all_pass = True
    
    # Check unittest report
    if unittest_report and unittest_report.exists():
        try:
            content = unittest_report.read_text(encoding="utf-8").strip()
            if content:
                data = json.loads(content, strict=False)
                correct_ratio = data.get("correct_ratio", 0.0)
                report_info["unittest_correct_ratio"] = correct_ratio
                if correct_ratio != 1.0:
                    all_pass = False
        except Exception as e:
            print(f"Warning: Failed to read unittest report {unittest_report}: {e}")
            all_pass = False
    else:
        print(f"Warning: Unittest report not found for tag_name: {tag_name}")
        all_pass = False
    
    # Check integtest report
    if integtest_report and integtest_report.exists():
        try:
            content = integtest_report.read_text(encoding="utf-8").strip()
            if content:
                data = json.loads(content, strict=False)
                ave_correct_ratio = data.get("ave_correct_ratio", 0.0)
                report_info["integtest_ave_correct_ratio"] = ave_correct_ratio
                if ave_correct_ratio != 1.0:
                    all_pass = False
        except Exception as e:
            print(f"Warning: Failed to read integtest report {integtest_report}: {e}")
            all_pass = False
    else:
        print(f"Warning: Integtest report not found for tag_name: {tag_name}")
        all_pass = False
    
    return all_pass, report_info




async def main():
    args = parse_args()
    task_data = OtherTools.load_task_parameters(args.topic, args.task_name)
    
    pdf_name = task_data.get("pdf_name")
    subplot_name = task_data.get("subplot_name")
    
    # Determine work directory
    work_dir = QMBagents.create_run_log(
        pdf_name=pdf_name,
        subplotname=subplot_name,
        output_dir="../logs",
        topic=args.topic,
        issubfile=True,
        iscreatlog=False,
        headname_dir="",
        idname_dir=args.idname_proj
    )
    work_dir = Path(work_dir)
    
    # Initialize loop variables
    current_code_name = args.code_original_name
    repair_iteration = 0
    current_repo_dir = None
    tag_name_for_report = args.tag_name  # First iteration uses initial tag_name
    
    # Check initial state (optional)
    all_pass, _ = check_reports_pass(work_dir, args.tag_name)
    if all_pass:
        print(f"Initial reports for tag_name '{args.tag_name}' already pass. Exiting.")
        return
    elif not all_pass:
        print(f"Initial reports for tag_name '{args.tag_name}' do not pass. Starting repair process.")
    
    # Main repair loop------------------------------------------------------------------
    while repair_iteration < args.max_repair_iterations:
        print(f"\n{'='*60}")
        print(f"Repair iteration {repair_iteration + 1}")
        print(f"{'='*60}")
        
        # Step 1: Repair the code
        tag_name_repair = f"{args.tag_name}-repair{repair_iteration + 1}"
        code_repair_name = f"code_LLM_repair{repair_iteration + 1}"
        
        print(f"Step 1: Repairing code (tag_name: {tag_name_repair})...")
        args_repair = create_repair_args(
            args, tag_name_repair, tag_name_for_report, current_code_name, code_repair_name
        )
        repair_result = await coderepair_v5.task_coderepair(args_repair)
        current_code_file = repair_result["code_file"]
        current_code_name = code_repair_name
        current_repo_dir = repair_result["repo_dir"]
        print(f"Repair completed. Code: {current_code_file}, Repo: {current_repo_dir}")
        
        # Step 2: Run unittest on the repaired code/repo
        print(f"Step 2: Running unittest (idname_unittest: {tag_name_repair})...")
        args_unittest = create_unittest_args(
            args, tag_name_repair, current_code_file, current_repo_dir
        )
        unittest_result = await unittest_v5.task_unittest(args_unittest)
        print(f"Unittest completed. Report: {unittest_result['report_file']}")
        
        # Step 3: Run integtest on the repaired code/repo
        print(f"Step 3: Running integtest (idname_integtest: {tag_name_repair})...")
        args_integtest = create_integtest_args(
            args, tag_name_repair, current_repo_dir
        )
        integtest_result = await integtest_v5.task_integtest(args_integtest)
        print(f"Integtest completed. Report: {integtest_result['report_file']}")
        
        # Step 4: Check reports and decide whether to continue
        print(f"Step 4: Checking reports for tag_name: {tag_name_repair}...")
        all_pass, report_info = check_reports_pass(work_dir, tag_name_repair)
        
        if all_pass:
            print(f"\n{'='*60}")
            print(f"SUCCESS: All tests pass after {repair_iteration + 1} iteration(s)!")
            print(f"Unittest correct_ratio: {report_info['unittest_correct_ratio']}")
            print(f"Integtest ave_correct_ratio: {report_info['integtest_ave_correct_ratio']}")
            print(f"{'='*60}")
            break
        else:
            print(f"Tests not fully passed:")
            print(f"  Unittest correct_ratio: {report_info['unittest_correct_ratio']}")
            print(f"  Integtest ave_correct_ratio: {report_info['integtest_ave_correct_ratio']}")
            # Update tag_name_for_report for next iteration
            tag_name_for_report = tag_name_repair
            repair_iteration += 1
            if repair_iteration >= args.max_repair_iterations:
                print(f"\nWarning: Reached maximum iterations ({args.max_repair_iterations}). Stopping.")
    
    print(f"\ntask {args.task_name} workflow completed.")


if __name__ == "__main__":
    asyncio.run(main())
