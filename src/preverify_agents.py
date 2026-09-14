from openai import OpenAI
import time
import random
from pathlib import Path
import json
import base64
from typing import Optional, Dict, List, Tuple
import fitz
import re
from langchain_openai import ChatOpenAI
import os
import sys
import asyncio
import ast
import shutil
from decimal import Decimal, InvalidOperation
from mcp_use import MCPAgent, MCPClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Import from this project
from config import OPENROUTER_API_KEY
from config import get_model_settings
from utils import MCP_toolbox, run_program, code_editor
from QMBagents import QMBagent, create_run_log

# Check the API key
if not OPENROUTER_API_KEY:
    raise ValueError("OpenRouter API key is not set. Please set the OPENROUTER_API_KEY environment variable.")


class preverify_agents(QMBagent):
    """
    Agents for generating preverify code/rubrics. 
    """
    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        prompt_dir: str = "../prompts",
        output_dir: str = "../CodeVerifier_library_auto",
        current_log_file: str = None,
        current_code_dir: str = "../Paper_docu/Codes",
        topic: str = None,
        type_name: Optional[str] = None,
    ):
        self.api_key = api_key
        self.prompt_dir = prompt_dir
        self.output_dir = output_dir
        self.current_log_file = current_log_file
        self.current_code_dir = current_code_dir
        self.topic = topic
        self.type_name = type_name

        # check
        if self.api_key is None:
            raise ValueError("API key must be provided for QMBagent.")
        if self.topic is None:
            raise ValueError("Topic must be specified for QMBagent.")


class CodeVerifier_library_generator(preverify_agents):
    """
    Agents for generating CodeVerifier_library. 
    """

    def _resolve_library_dir(self) -> Path:
        """
        Resolve library directory for current topic.
        """
        base = Path(self.output_dir)
        return base / ("CodeVerifier_library_" + self.topic)

    def _sanitize_type_tag(self, type_str: str) -> str:
        """
        Make a safe identifier/file tag from a type string (avoid '/', spaces, etc.).
        """
        if not type_str:
            return "type"
        tag = re.sub(r"[^A-Za-z0-9_]+", "_", type_str).strip("_")
        return tag or "type"

    def _collect_type_guides(self, library_dir: Path) -> List[Tuple[Path, dict]]:
        """
        Collect per-type guide.json files under *type*_elements/ directories.

        Returns:
            list of (type_elements_dir, guide_dict)
        """
        guides: List[Tuple[Path, dict]] = []
        if not library_dir.exists():
            return guides

        for p in library_dir.rglob("guide.json"):
            if p.parent.name.endswith("_elements"):
                try:
                    guides.append((p.parent, json.loads(self._load_file(p))))
                except Exception:
                    continue
        return guides

    def _get_type_context(self, type_name: Optional[str] = None) -> Tuple[Path, Path, dict, str, str]:
        """
        Resolve the single working type context.

        Returns:
            (library_dir, type_elements_dir, guide_data, type_display, type_tag)
        """
        effective_type = type_name or getattr(self, "type_name", None)
        if not effective_type:
            raise ValueError("type_name is required (pass at agent init or to *_single_type method).")

        library_dir = self._resolve_library_dir()
        type_elements_dir = library_dir / f"{effective_type}_elements"
        guide_path = type_elements_dir / "guide.json"
        if not guide_path.exists():
            raise FileNotFoundError(
                f"Per-type guide.json not found: '{guide_path}'. "
                f"Expected directory '{type_elements_dir}' for type_name='{effective_type}'."
            )

        guide_data = json.loads(self._load_file(guide_path))
        type_display = str(guide_data.get("type", effective_type))
        type_tag = self._sanitize_type_tag(type_display)
        return library_dir, type_elements_dir, guide_data, type_display, type_tag


    # Cold initiation for a single type
    def Cold_init_single_type(self, isoverwrite: bool = False, type_name: Optional[str] = None):
        """
        Prepare the file structure for a single (topic, type) CodeVerifier library.

        If type_name is not provided, defaults to self.type_name.
        """
        library_dir, type_elements_dir, guide_data, type_display, _type_tag = self._get_type_context(type_name=type_name)
        library_dir.mkdir(parents=True, exist_ok=True)
        type_elements_dir.mkdir(parents=True, exist_ok=True)

        elements = guide_data.get("elements", [])

        # Shared archive directory at library root
        (library_dir / "verifier_code").mkdir(parents=True, exist_ok=True)

        # Create prepare/refine subdirectories
        (type_elements_dir / "prepare").mkdir(parents=True, exist_ok=True)
        (type_elements_dir / "refining_verifier_code").mkdir(parents=True, exist_ok=True)

        for element in elements:
            (type_elements_dir / f"query_{element}").mkdir(parents=True, exist_ok=True)
                
        print(f"Cold initiation for CodeVerifier_library_{self.topic} ({type_display}) completed.")

        return {"library_dir": library_dir, "type": type_display}


    async def Prepare_QueryMaterial_mcp_use_concurr_single_type(
        self,
        user_model: str,
        concurr_num: int = 5,
        query_num: int = 5,
        type_name: Optional[str] = None,
    ):
        """
        Single-type version of Prepare_QueryMaterial_mcp_use_concurr.

        If type_name is not provided, defaults to self.type_name.
        """
        library_dir, type_elements_dir, guide_data, type_display, type_tag = self._get_type_context(type_name=type_name)
        
        # Load base prompt once before loops (save computation)
        base_prompt = self._load_prompt(
            topic="general",
            prompt_file="PrepareQueryMaterial_mcp_use_stand1"
        )
        
        elements = guide_data.get("elements", [])
        element_notes = guide_data.get("element_notes", [])
        library_names = guide_data.get("library_name", [])
        library_names_str = json.dumps(library_names, ensure_ascii=False) if library_names else "[]"

        element_to_note = {}
        if element_notes and len(element_notes) == len(elements):
            element_to_note = dict(zip(elements, element_notes))

        print(f"Processing type: {type_display} ...")

        query_element_dirs: list[tuple[str, Path]] = []
        for element in elements:
            query_element_dir = type_elements_dir / f"query_{element}"
            if query_element_dir.exists() and query_element_dir.is_dir():
                query_element_dirs.append((element, query_element_dir))

        if query_element_dirs:
            sem = asyncio.Semaphore(concurr_num)

            async def _process_query_element(element: str, query_element_dir: Path):
                async with sem:
                    print(f"Processing {type_display}/{element}...")

                    element_note = element_to_note.get(element, "")

                    user_prompt = base_prompt.format(
                        type=type_display,
                        element=element,
                        element_note=element_note,
                        topic=self.topic,
                        query_num=query_num,
                        library_names=library_names_str,
                        query_element_dir=str(query_element_dir),
                        output_summary_file=f"{type_tag}_{element}_summary.json",
                    )

                    _ = await self._send_chat_through_mcp_dynamic(
                        user_model=user_model,
                        user_prompt=user_prompt,
                        fs_target_dir=str(query_element_dir),
                        retrieve_rt_target_dir=str(query_element_dir),
                        init_servers=["retrieve-mcp", "filesystem-mcp"],
                    )

                    print(f"Query material preparation for {type_display}/{element} completed.")
                    return element

            tasks = [
                asyncio.create_task(_process_query_element(element, query_element_dir))
                for element, query_element_dir in query_element_dirs
            ]
            await asyncio.gather(*tasks)
        
        output = {
            "Content": "Query material preparation for CodeVerifier_library",
            "LLM": user_model,
            "library_dir": str(library_dir),
            "topic": self.topic,
            "query_num": query_num
        }
        self._log_event(output)
        return output


    async def Generate_VerifierCode_mcp_use_concurr_single_type(
        self,
        user_model: str,
        concurr_num: int = 5,
        num_check: int = 2,
        type_name: Optional[str] = None,
    ):
        """
        Single-type version of Generate_VerifierCode_mcp_use_concurr.

        If type_name is not provided, defaults to self.type_name.
        """
        library_dir, type_elements_dir, guide_data, type_display, type_tag = self._get_type_context(type_name=type_name)
        
        # Load both prompts once before loops (save computation)
        general_prompt = self._load_prompt(
            topic="general",
            prompt_file="GenerateVerifierCode_initial_mcp_use_stand1"
        )
        format_prompt = self._load_prompt(
            topic=self.topic,
            prompt_file="VerifierCode_initial_generation_rules"
        )
        
        elements = guide_data.get("elements", [])
        element_notes = guide_data.get("element_notes", [])

        element_to_note = {}
        if element_notes and len(element_notes) == len(elements):
            element_to_note = dict(zip(elements, element_notes))

        print(f"Processing type: {type_display} ...")

        prepare_dir = type_elements_dir / "prepare"
        prepare_dir.mkdir(parents=True, exist_ok=True)

        element_summaries: list[tuple[str, Path, Path]] = []
        for element in elements:
            query_element_dir = type_elements_dir / f"query_{element}"
            summary_file_path = query_element_dir / f"{type_tag}_{element}_summary.json"
            element_summaries.append((element, summary_file_path, query_element_dir))

        sem = asyncio.Semaphore(concurr_num)

        async def _process_element(element: str, summary_file_path: Path, query_element_dir: Path):
            async with sem:
                print(f"Generating verifier code for {type_display}/{element}...")

                element_note = element_to_note.get(element, "")

                formatted_general = general_prompt.format(
                    type=type_tag,
                    element=element,
                    topic=self.topic,
                    summary_file_path=str(summary_file_path),
                    output_file_path=str(prepare_dir),
                    output_filename=f"{element}_verifier_{type_tag}",
                )
                formatted_format = format_prompt.format(
                    type=type_tag,
                    element=element,
                    element_note=element_note,
                    topic=self.topic,
                )

                combined_prompt = formatted_general + "\n\n" + formatted_format

                _ = await self._send_chat_through_mcp_dynamic(
                    user_model=user_model,
                    user_prompt=combined_prompt,
                    fs_target_dir=str(prepare_dir),
                    reference_dir_dict={"libs": str(query_element_dir)},
                    init_servers=["filesystem-mcp"],
                )

                print(f"Verifier code generation for {type_display}/{element} completed.")
                return element

        tasks = [
            asyncio.create_task(_process_element(element, summary_file_path, query_element_dir))
            for element, summary_file_path, query_element_dir in element_summaries
        ]
        await asyncio.gather(*tasks)

        for check_round in range(1, num_check + 1):
            print(f"Checking for missing verifier functions in {type_display} (round {check_round}/{num_check})...")
            existing_files = set()
            for file in prepare_dir.iterdir():
                if file.is_file():
                    filename = file.stem
                    if filename.endswith(f"_verifier_{type_tag}"):
                        element_from_file = filename[:-len(f"_verifier_{type_tag}")]
                        existing_files.add(element_from_file)

            missing_elements: list[tuple[str, Path, Path]] = []
            for element in elements:
                if element not in existing_files:
                    query_element_dir = type_elements_dir / f"query_{element}"
                    summary_file_path = query_element_dir / f"{type_tag}_{element}_summary.json"
                    missing_elements.append((element, summary_file_path, query_element_dir))

            if missing_elements:
                print(f"Found {len(missing_elements)} missing verifier function(s) for {type_display}: {[e[0] for e in missing_elements]}")
                print(f"Regenerating missing verifier functions (round {check_round}/{num_check})...")
                missing_tasks = [
                    asyncio.create_task(_process_element(element, summary_file_path, query_element_dir))
                    for element, summary_file_path, query_element_dir in missing_elements
                ]
                await asyncio.gather(*missing_tasks)
                print(f"Missing verifier functions regeneration for {type_display} (round {check_round}/{num_check}) completed.")
            else:
                print(f"All verifier functions for {type_display} are present.")
                break
        
        output = {
            "Content": "Verifier code generation for CodeVerifier_library",
            "LLM": user_model,
            "library_dir": str(library_dir),
            "topic": self.topic
        }
        self._log_event(output)
        return output


    async def Combine_VerifierCode_mcp_use_single_type(self, user_model: str, type_name: Optional[str] = None):
        """
        Combine verifier code by integrating multiple element functions into complete programs (single type).
        
        Workflow:
        1. Read all element functions from prepare directory (as reference)
        2. Group functions by element type (first word of function name)
        3. Generate integ_verifier_xxx programs combining all element types
        4. If multiple functions exist for same element type, create multiple integ_verifier programs
        
        Args:
            user_model: LLM model to use
            type_name: Type name (e.g., "boson", "fermion")
        """
        library_dir, type_elements_dir, _guide_data, type_display, type_tag = self._get_type_context(type_name=type_name)
        prepare_dir = type_elements_dir / "prepare"
        refining_dir = type_elements_dir / "refining_verifier_code"
        refining_dir.mkdir(parents=True, exist_ok=True)
        
        # Load prompt
        combine_prompt = self._load_prompt(
            topic="general",
            prompt_file="CombineVerifierCode_mcp_use_stand1"
        )
        
        # Format prompt with variables
        formatted_prompt = combine_prompt.format(
            type=type_tag,
            topic=self.topic,
            prepare_dir=str(prepare_dir),
            refining_dir=str(refining_dir)
        )
        
        # Single MCP call with filesystem-mcp
        # prepare_dir as reference, refining_dir as working directory
        response = await self._send_chat_through_mcp_dynamic(
            user_model=user_model,
            user_prompt=formatted_prompt,
            fs_target_dir=str(refining_dir),
            reference_dir_dict={"libs": str(prepare_dir)},
            init_servers=["filesystem-mcp"]
        )
        
        output = {
            "Content": "Verifier code combination for CodeVerifier_library",
            "LLM": user_model,
            "type": type_display,
            "library_dir": str(library_dir),
            "topic": self.topic
        }
        self._log_event(output)
        return output


    async def Combine_VerifierCode_mcp_use_robust_single_type(self, user_model: str, type_name: Optional[str] = None):
        """
        Robust version: Hard code merge all prepare files, then use LLM to organize.
        
        Workflow:
        1. Hard code: Copy and merge all .py/.jl files from prepare into integ_verifier_basic.<ext>
        2. LLM: Organize imports and create main() function
        
        Args:
            user_model: LLM model to use
            type_name: Type name (e.g., "boson", "fermion")
        """
        library_dir, type_elements_dir, _guide_data, type_display, type_tag = self._get_type_context(type_name=type_name)
        prepare_dir = type_elements_dir / "prepare"
        refining_dir = type_elements_dir / "refining_verifier_code"
        refining_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Hard code merge all prepare files
        prepare_files = []
        for ext in [".py", ".jl"]:
            prepare_files.extend(prepare_dir.glob(f"*{ext}"))
        
        if not prepare_files:
            print(f"No program files found in {prepare_dir}, skipping {type_display}.")
            return {"type": type_tag, "skipped": True, "reason": "no_prepare_files"}
        
        # Determine output extension (use first file's extension)
        output_ext = prepare_files[0].suffix
        prepare_files.sort(key=lambda p: p.name)  # Sort for consistency
        
        # Merge all files
        merged_lines = []
        for file_path in prepare_files:
            content = file_path.read_text(encoding="utf-8")
            merged_lines.append(f"# === {file_path.name} ===\n")
            merged_lines.append(content)
            merged_lines.append("\n\n")
        
        # Save merged file
        output_file = refining_dir / f"integ_verifier_basic{output_ext}"
        output_file.write_text("".join(merged_lines), encoding="utf-8")
        print(f"[robust combine] Merged {len(prepare_files)} files into {output_file.name}")
        
        # Step 2: LLM organization
        robust_prompt = self._load_prompt(
            topic="general",
            prompt_file="CombineVerifierCode_mcp_use_robust_stand1"
        )
        
        formatted_prompt = robust_prompt.format(
            type=type_tag,
            topic=self.topic,
            program_file=f"integ_verifier_basic{output_ext}",
            file_ext=output_ext[1:] if output_ext else "unknown",
            refining_dir=str(refining_dir),
        )
        
        await self._send_chat_through_mcp_dynamic(
            user_model=user_model,
            user_prompt=formatted_prompt,
            fs_target_dir=str(refining_dir),
            init_servers=["filesystem-mcp"]
        )
        
        output = {
            "Content": "Robust verifier code combination for CodeVerifier_library",
            "LLM": user_model,
            "type": type_display,
            "library_dir": str(library_dir),
            "topic": self.topic
        }
        self._log_event(output)
        return output


    async def Refine_VerifierCode_freqRAG_mcp_use_single_type(
        self,
        user_model: str,
        type_name: Optional[str] = None,
        max_iterations: int = 10,
        program_timeout: int = 120,
        refine_query_num: int = 6,
    ):
        """
        Single-type version with 3-phase freqRAG workflow.

        Phase 1: Run + Modify (max 2 iterations)
        Phase 2: RAG Retrieval (if Phase 1 fails)
        Phase 3: Modify code with summary (if Phase 1 fails)
        
        Each iteration (1 to max_iterations) is a "section" containing all 3 phases.
        """
        # Load files
        library_dir, chosen_dir, _guide_data, type_display, type_tag = self._get_type_context(type_name=type_name)
        phase1_prompt = self._load_prompt(
            topic="general",
            prompt_file="RefineVerifierCode_freqRAG_phase1_mcp_use_stand1",
        )
        phase2_prompt = self._load_prompt(
            topic="general",
            prompt_file="RefineVerifierCode_freqRAG_phase2_mcp_use_stand1"
        )
        phase3_prompt = self._load_prompt(
            topic="general",
            prompt_file="RefineVerifierCode_freqRAG_phase3_mcp_use_stand1"
        )

        refining_dir = chosen_dir / "refining_verifier_code"
        type_elements_dir = chosen_dir
        if not refining_dir.exists():
            print(f"Refining directory '{refining_dir}' not found, skipping {type_display}.")
            return {"type": type_tag, "skipped": True, "reason": "refining_dir_missing"}

        program_files: list[Path] = []
        for ext in ["*.py", "*.jl"]:
            program_files.extend(refining_dir.glob(ext))
        if not program_files:
            print(f"No program files found in {refining_dir}, skipping {type_display}.")
            return {"type": type_tag, "skipped": True, "reason": "no_program_files"}
        print(f"Found {len(program_files)} program file(s) in {type_display}.")

        # Begin the main workflow
        for program_file in program_files:
            print(f"Refining program: {program_file.name} ...")

            file_ext = program_file.suffix
            is_python = file_ext == ".py"
            is_julia = file_ext == ".jl"
            if not (is_python or is_julia):
                print(f"Skipping {program_file.name} (not a Python or Julia file).")
                continue
            ext_no_dot = file_ext[1:] if file_ext else "unknown"
            
            # Iterate through sections (each section = Phase 1 + Phase 2 + Phase 3)
            for iteration in range(1, max_iterations + 1):
                print(f"[Section {iteration}/{max_iterations}] Starting Phase 1 for {program_file.name}...")
                
                # Phase 1: Run + Modify (max 2 iterations)---------------------------------------------
                formatted_phase1_prompt = phase1_prompt.format(
                    type=type_tag,
                    topic=self.topic,
                    program_file=str(program_file.name),
                    program_path=str(program_file),
                    num_iterations=2, 
                    file_ext=ext_no_dot,
                )
                await self._send_chat_through_mcp_dynamic(
                    user_model=user_model,
                    user_prompt=formatted_phase1_prompt,
                    fs_target_dir=str(refining_dir),  # filesystem-mcp working directory
                    init_servers=["program-mcp", "filesystem-mcp"],
                    # init_servers=["program-mcp"],
                    program_path=str(program_file),
                    program_timeout=program_timeout,
                    isskipclose=True,  # Skip close to avoid cancel scope errors with dual MCP servers
                    # max_steps=20
                )
                await asyncio.to_thread(time.sleep, 1)

                # Check if Phase 1 succeeded
                run_log_path = refining_dir / f"run_log_{program_file.stem}_{ext_no_dot}.json"
                exitcode = None
                if run_log_path.exists():
                    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
                    exitcode = run_log.get("exitcode")
                # if phase 1 succeeded
                if exitcode == 0:
                    print(f"[Section {iteration}] Phase 1 succeeded, archiving code...")
                    self._archive_verifier_code(
                        program_file=program_file,
                        type_name=type_tag,
                        library_dir=library_dir,
                    )
                    break  # Exit iteration loop
                # if phase 1 failed
                print(f"[Section {iteration}] Phase 1 failed (exitcode={exitcode}), starting Phase 2...")
                
                # Phase 2: RAG Retrieval---------------------------------------------
                query_dir = type_elements_dir / f"query_refine_{iteration}"
                if query_dir.exists():
                    shutil.rmtree(query_dir)
                query_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy report and run_log to query directory (if exist)
                report_file = refining_dir / f"report_{program_file.stem}_{ext_no_dot}.json"
                if report_file.exists():
                    shutil.copy2(report_file, query_dir / report_file.name)
                if run_log_path.exists():
                    shutil.copy2(run_log_path, query_dir / run_log_path.name)
                
                # Read stderr from run_log and pass directly to prompt
                stderr_info = ""
                if run_log_path.exists():
                    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
                    stderr_info = run_log.get("stderr", "")
                
                _, _, guide_data, _, _ = self._get_type_context(type_name=self.type_name)
                library_names = guide_data.get("library_name", [])
                library_names_str = json.dumps(library_names, ensure_ascii=False) if library_names else "[]"
                
                formatted_phase2_prompt = phase2_prompt.format(
                    type=type_display,
                    topic=self.topic,
                    query_num=refine_query_num,
                    library_names=library_names_str,
                    query_dir=str(query_dir),
                    output_summary_file=f"refine_summary.json",
                    stderr_info=stderr_info,  # Pass stderr directly to prompt
                )
                await self._send_chat_through_mcp_dynamic(
                    user_model=user_model,
                    user_prompt=formatted_phase2_prompt,
                    fs_target_dir=str(query_dir),  # filesystem-mcp working directory
                    retrieve_rt_target_dir=str(query_dir),  # retrieve-mcp working directory
                    init_servers=["retrieve-mcp", "filesystem-mcp"],
                    isskipclose=True,  # Skip close to avoid cancel scope errors with dual MCP servers
                )
                print(f"[Section {iteration}] Phase 2 completed, starting Phase 3...")
                
                # Phase 3: Modify code with summary---------------------------------------------
                formatted_phase3_prompt = phase3_prompt.format(
                    type=type_tag,
                    topic=self.topic,
                    program_file=str(program_file.name),
                    file_ext=ext_no_dot,
                    query_dir=str(query_dir),  # Reference directory name
                    refining_dir=str(refining_dir),  # Working directory
                )
                await self._send_chat_through_mcp_dynamic(
                    user_model=user_model,
                    user_prompt=formatted_phase3_prompt,
                    fs_target_dir=str(refining_dir),  # filesystem-mcp working directory
                    reference_dir_dict={"libs": str(query_dir)},  # Reference to query_refine_{iteration}
                    init_servers=["filesystem-mcp"],  # No program-mcp
                )
                print(f"[Section {iteration}] Phase 3 completed.")
            
            # After max_iterations loop, check final status
            run_log_path = refining_dir / f"run_log_{program_file.stem}_{ext_no_dot}.json"
            if run_log_path.exists():
                run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
                exitcode = run_log.get("exitcode")
                if exitcode != 0:
                    snippet = (run_log.get("stderr") or "").strip().replace("\r\n", "\n")[:600]
                    print(f"[freqRAG refine not complete] {program_file.name}: exitcode={exitcode}, skip archive.")
                    print(f"[stderr snippet]\n{snippet}\n")
            print(f"Refinement for {program_file.name} completed.")

        output = {
            "Content": "Verifier code refinement (freqRAG) for CodeVerifier_library (single type)",
            "LLM": user_model,
            "library_dir": str(library_dir),
            "topic": self.topic,
            "type": type_tag,
            "max_iterations": max_iterations,
        }
        self._log_event(output)
        return output


    async def Refine_VerifierCode_mcp_use_single_type(
        self,
        user_model: str,
        type_name: Optional[str] = None,
        max_iterations: int = 10,
        program_timeout: int = 120,
        refine_query_num: int = 5,
    ):
        """
        Single-type version of Refine_VerifierCode_mcp_use_concurr.

        Runs refine workflow for ONE specified type only (no cross-type concurrency).
        """
        library_dir, chosen_dir, _guide_data, type_display, type_tag = self._get_type_context(type_name=type_name)

        refine_prompt = self._load_prompt(
            topic="general",
            prompt_file="RefineVerifierCode_mcp_use_stand1",
        )

        refining_dir = chosen_dir / "refining_verifier_code"
        type_elements_dir = chosen_dir
        if not refining_dir.exists():
            print(f"Refining directory '{refining_dir}' not found, skipping {type_display}.")
            return {"type": type_tag, "skipped": True, "reason": "refining_dir_missing"}

        program_files: list[Path] = []
        for ext in ["*.py", "*.jl"]:
            program_files.extend(refining_dir.glob(ext))

        if not program_files:
            print(f"No program files found in {refining_dir}, skipping {type_display}.")
            return {"type": type_tag, "skipped": True, "reason": "no_program_files"}

        print(f"Found {len(program_files)} program file(s) in {type_display}.")

        for program_file in program_files:
            print(f"Refining program: {program_file.name} ...")

            file_ext = program_file.suffix
            is_python = file_ext == ".py"
            is_julia = file_ext == ".jl"
            if not (is_python or is_julia):
                print(f"Skipping {program_file.name} (not a Python or Julia file).")
                continue

            formatted_prompt = refine_prompt.format(
                type=type_tag,
                topic=self.topic,
                program_file=str(program_file.name),
                program_path=str(program_file),
                max_iterations=max_iterations,
                file_ext=file_ext[1:] if file_ext else "unknown",
            )

            _ = await self._send_chat_through_mcp_dynamic(
                user_model=user_model,
                user_prompt=formatted_prompt,
                fs_target_dir=str(refining_dir),
                init_servers=["program-mcp", "filesystem-mcp"],
                program_path=str(program_file),
                program_timeout=program_timeout,
            )

            # Archive refined program into verifier_code/ if exitcode is 0 in run_log
            try:
                ext_no_dot = file_ext[1:] if file_ext else "unknown"
                run_log_path = refining_dir / f"run_log_{program_file.stem}_{ext_no_dot}.json"
                if not run_log_path.exists():
                    print(f"[warn] run_log not found, skip archive: {run_log_path}")
                else:
                    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
                    exitcode = run_log.get("exitcode")
                    if exitcode == 0:
                        self._archive_verifier_code(
                            program_file=program_file,
                            type_name=type_tag,
                            library_dir=library_dir,
                        )
                    else:
                        snippet = (run_log.get("stderr") or "").strip().replace("\r\n", "\n")[:600]
                        print(f"[refine not complete] {program_file.name}: exitcode={exitcode}, starting Refine2...")
                        if snippet:
                            print(f"[stderr snippet]\n{snippet}\n")
                        
                        # Call second-stage refinement
                        await self.Refine_VerifierCode2_mcp_use_single_type(
                            user_model=user_model,
                            program_file=program_file,
                            refining_dir=refining_dir,
                            type_elements_dir=type_elements_dir,
                            library_dir=library_dir,
                            type_tag=type_tag,
                            type_display=type_display,
                            max_iterations=max_iterations,
                            program_timeout=program_timeout,
                            refine_query_num=refine_query_num,
                        )
            except Exception as e:
                print(f"[warn] archive step failed for {program_file.name}: {e}")

            print(f"Refinement for {program_file.name} completed.")

        output = {
            "Content": "Verifier code refinement for CodeVerifier_library (single type)",
            "LLM": user_model,
            "library_dir": str(library_dir),
            "topic": self.topic,
            "type": type_tag,
            "max_iterations": max_iterations,
        }
        self._log_event(output)
        return output


    async def Refine_VerifierCode2_mcp_use_single_type(
        self,
        user_model: str,
        program_file: Path,
        refining_dir: Path,
        type_elements_dir: Path,
        library_dir: Path,
        type_tag: str,
        type_display: str,
        max_iterations: int,
        program_timeout: int,
        refine_query_num: int,
    ) -> dict:
        """
        Second-stage refinement using RAG when first stage fails (exitcode != 0).
        
        Workflow:
        1. Create query directory and copy report/run_log files
        2. Retrieve queries based on errors and generate summary
        3. Refine code using RAG summary and existing verifier code as references
        """
        # Step 1: Create query directory and copy files
        # query_dir is at the same level as refining_verifier_code (not inside it)
        query_dir = type_elements_dir / f"query_{program_file.stem}"
        # Remove existing directory if it exists, then create fresh
        if query_dir.exists():
            shutil.rmtree(query_dir)
        query_dir.mkdir(parents=True, exist_ok=True)
        
        file_ext = program_file.suffix
        ext_no_dot = file_ext[1:] if file_ext else "unknown"
        report_file = refining_dir / f"report_{program_file.stem}_{ext_no_dot}.json"
        run_log_file = refining_dir / f"run_log_{program_file.stem}_{ext_no_dot}.json"
        
        if report_file.exists():
            shutil.copy2(report_file, query_dir / report_file.name)
        if run_log_file.exists():
            shutil.copy2(run_log_file, query_dir / run_log_file.name)
        
        # Step 2: Retrieve queries and generate summary
        refine_query_prompt = self._load_prompt(
            topic="general",
            prompt_file="RefineQueryMaterial_mcp_use_stand1"
        )
        
        # Get library_names from guide_data
        _, _, guide_data, _, _ = self._get_type_context(type_name=self.type_name)
        library_names = guide_data.get("library_name", [])
        library_names_str = json.dumps(library_names, ensure_ascii=False) if library_names else "[]"
        
        formatted_query_prompt = refine_query_prompt.format(
            type=type_display,
            topic=self.topic,
            query_num=refine_query_num,
            library_names=library_names_str,
            query_dir=str(query_dir),
            output_summary_file="refine_summary.json",
        )
        
        await self._send_chat_through_mcp_dynamic(
            user_model=user_model,
            user_prompt=formatted_query_prompt,
            fs_target_dir=str(query_dir),
            retrieve_rt_target_dir=str(query_dir),
            init_servers=["retrieve-mcp", "filesystem-mcp"],
        )
        
        print(f"[Refine2] Step 2 completed, starting Step 3 for {program_file.name}...")
        
        # Step 3: Refine with RAG results and verifier_code reference
        verifier_code_dir = library_dir / "verifier_code"
        
        refine2_prompt = self._load_prompt(
            topic="general",
            prompt_file="RefineVerifierCode2_mcp_use_stand1"
        )
        
        formatted_refine2_prompt = refine2_prompt.format(
            type=type_tag,
            topic=self.topic,
            program_file=str(program_file.name),
            program_path=str(program_file),
            program_stem=program_file.stem,
            max_iterations=max_iterations,
            file_ext=ext_no_dot,
        )
        
        reference_dir_dict = {
            "query": str(query_dir),
            "verifier_code": str(verifier_code_dir),
        }
        
        await self._send_chat_through_mcp_dynamic(
            user_model=user_model,
            user_prompt=formatted_refine2_prompt,
            fs_target_dir=str(refining_dir),
            reference_dir_dict=reference_dir_dict,
            init_servers=["program-mcp", "filesystem-mcp"],
            program_path=str(program_file),
            program_timeout=program_timeout,
        )
        
        # Check if refinement succeeded and archive if exitcode == 0
        try:
            run_log_path = refining_dir / f"run_log_{program_file.stem}_{ext_no_dot}.json"
            if run_log_path.exists():
                run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
                exitcode = run_log.get("exitcode")
                if exitcode == 0:
                    self._archive_verifier_code(
                        program_file=program_file,
                        type_name=type_tag,
                        library_dir=library_dir,
                    )
                else:
                    snippet = (run_log.get("stderr") or "").strip().replace("\r\n", "\n")[:600]
                    print(f"[refine2 not complete] {program_file.name}: exitcode={exitcode}, skip archive.")
                    if snippet:
                        print(f"[stderr snippet]\n{snippet}\n")
        except Exception as e:
            print(f"[warn] archive step failed for {program_file.name}: {e}")
        
        return {"type": type_tag, "program_file": str(program_file.name)}


    def _archive_verifier_code(
        self, 
        program_file: Path, 
        type_name: str, 
        library_dir: Path,
        keep_entrypoint: bool = True
    ) -> Optional[Path]:
        """
        If a program is refined successfully, copy it into verifier_code/.
        
        Args:
            keep_entrypoint: If True (default), save complete code including main() and entrypoint.
                           If False, remove main() and entrypoint before saving.
        
        Output filename (overwrite if exists):
            verifier_{topic}_{type}_{suffix}{ext}
        where suffix is the part after "verifier_" in the program stem (fallback: full stem).
        """
        # verifier_code directory is at the library root (sibling of *type*_elements dirs)
        verifier_code_dir = library_dir / "verifier_code"
        verifier_code_dir.mkdir(parents=True, exist_ok=True)

        def _variant_from_stem(stem: str) -> str:
            if "verifier_" in stem:
                return stem.split("verifier_", 1)[1] or stem
            return stem

        src = program_file.read_text(encoding="utf-8", errors="replace")
        ext = program_file.suffix.lower()
        variant = _variant_from_stem(program_file.stem)
        out_name = f"verifier_{self.topic}_{type_name}_{variant}{program_file.suffix}"
        out_path = verifier_code_dir / out_name

        if keep_entrypoint:
            # Save complete code including main() and entrypoint
            cleaned = src
        else:
            # Remove main() and entrypoint as before
            if ext == ".py":
                cleaned = self._strip_python_main_and_entrypoint(src)
            elif ext == ".jl":
                cleaned = self._strip_julia_main_and_entrypoint(src)
            else:
                cleaned = src

        out_path.write_text(cleaned, encoding="utf-8")
        print(f"[archive] saved: {out_path}")
        return out_path

    def _strip_python_main_and_entrypoint(self, src: str) -> str:
        """
        Remove Python `def main(...)` and `if __name__ == "__main__": ...` blocks using AST line ranges.
        Also removes a standalone top-level `main()` call line if present.
        """
        import ast
        import re

        lines = src.splitlines(True)  # keep line endings
        try:
            tree = ast.parse(src)
        except Exception:
            return self._strip_by_regex_fallback(src, language="py")

        def _is_dunder_main_test(test: ast.AST) -> bool:
            if not isinstance(test, ast.Compare):
                return False
            if not (isinstance(test.left, ast.Name) and test.left.id == "__name__"):
                return False
            if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
                return False
            if len(test.comparators) != 1:
                return False
            comp = test.comparators[0]
            return isinstance(comp, ast.Constant) and comp.value == "__main__"

        ranges: list[tuple[int, int]] = []
        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main":
                if getattr(node, "end_lineno", None):
                    ranges.append((node.lineno, node.end_lineno))
            elif isinstance(node, ast.If) and _is_dunder_main_test(node.test):
                if getattr(node, "end_lineno", None):
                    ranges.append((node.lineno, node.end_lineno))

        # standalone main() call line
        for idx, line in enumerate(lines, start=1):
            if re.match(r"^\s*main\(\)\s*;?\s*$", line):
                ranges.append((idx, idx))

        if not ranges:
            return src

        ranges.sort()
        merged: list[list[int]] = []
        for a, b in ranges:
            if not merged or a > merged[-1][1] + 1:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)

        drop: set[int] = set()
        for a, b in merged:
            drop.update(range(a, b + 1))

        kept = [line for i, line in enumerate(lines, start=1) if i not in drop]
        return "".join(kept).rstrip() + "\n"

    def _strip_julia_main_and_entrypoint(self, src: str) -> str:
        """
        Remove Julia `function main(...) ... end` block (best-effort) and any top-level `main()` calls.
        """
        import re

        lines = src.splitlines(True)  # keep line endings

        def _strip_strings_and_comments(s: str) -> str:
            # best-effort: remove double-quoted strings then comments
            s2 = re.sub(r"\"([^\"\\\\]|\\\\.)*\"", "\"\"", s)
            return s2.split("#", 1)[0]

        def _is_main_call(line: str) -> bool:
            core = _strip_strings_and_comments(line)
            return re.match(r"^\s*main\(\)\s*;?\s*$", core) is not None

        start = None
        for i, line in enumerate(lines):
            core = _strip_strings_and_comments(line)
            if re.search(r"^\s*function\s+main\s*(\(|$)", core):
                start = i
                break

        if start is None:
            kept = [ln for ln in lines if not _is_main_call(ln)]
            return "".join(kept).rstrip() + "\n"

        # block matching: count openers + end, ignoring strings/comments
        openers = ("function", "if", "for", "while", "let", "begin", "try", "struct", "macro", "quote")
        depth = 0
        end_idx = None
        for j in range(start, len(lines)):
            core = _strip_strings_and_comments(lines[j])
            tokens = re.findall(r"\b[A-Za-z_]+\b", core)
            if j == start:
                depth = 1
            else:
                for kw in tokens:
                    if kw in openers:
                        depth += 1
            depth -= sum(1 for kw in tokens if kw == "end")
            if depth == 0 and j > start:
                end_idx = j
                break

        kept = lines[:start] if end_idx is None else (lines[:start] + lines[end_idx + 1 :])
        kept2 = [ln for ln in kept if not _is_main_call(ln)]
        return "".join(kept2).rstrip() + "\n"

    def _strip_by_regex_fallback(self, src: str, language: str) -> str:
        import re
        if language == "py":
            m = re.search(r"^\s*def\s+main\s*\(", src, flags=re.MULTILINE)
            if m:
                return (src[: m.start()]).rstrip() + "\n"
            m = re.search(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:\s*$", src, flags=re.MULTILINE)
            if m:
                return (src[: m.start()]).rstrip() + "\n"
        return src