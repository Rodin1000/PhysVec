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
from DFT_agents import DFT_agent, create_run_log

# Check the API key
if not OPENROUTER_API_KEY:
    raise ValueError("OpenRouter API key is not set. Please set the OPENROUTER_API_KEY environment variable.")


class repair_agents(DFT_agent):
    """
    Agents for repairing code according to the verification results. 
    """
    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        prompt_dir: str = "../prompts",
        output_dir: str = "../logs",
        current_log_file: str = None,
        current_code_dir: str = "../Paper_docu/Codes",
        topic: str = None,
    ):
        self.api_key = api_key
        self.prompt_dir = prompt_dir
        self.output_dir = output_dir
        self.current_log_file = current_log_file
        self.current_code_dir = current_code_dir
        self.topic = topic

        # check
        if self.api_key is None:
            raise ValueError("API key must be provided for repair_agents.")
        if self.topic is None:
            raise ValueError("Topic must be specified for repair_agents.")


class Code_repairer(repair_agents):
    """
    Agent for repairing code according to the verification results. 
    """

    def Execute_fullcode(self, code_file: str, timeout: int = 300) -> dict:
        """Execute code file (.py / .jl / .inp) and return {exitcode, stdout, stderr, timeout?, running_successfully?}.
        For .inp (ORCA) uses timeout=600 to match unittest; run_orca_file may return timeout and running_successfully
        (e.g. when ORCA runs for some time without error then times out)."""
        path = Path(code_file)
        project_root = str(path.parent)
        ext = path.suffix.lower()
        if ext == ".py":
            result = run_program.run_python_file(str(path), project_root, timeout)
            out = {
                "exitcode": result.get("exitcode"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "timeout": False,
                "running_successfully": None,
            }
        elif ext == ".jl":
            result = run_program.run_julia_file(str(path), project_root, timeout)
            out = {
                "exitcode": result.get("exitcode"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "timeout": False,
                "running_successfully": None,
            }
        elif ext == ".inp":
            orca_timeout = 600  # Match unittest for ORCA
            result = run_program.run_orca_file(str(path), project_root, orca_timeout)
            out = {
                "exitcode": result.get("exitcode"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "timeout": result.get("timeout", False),
                "running_successfully": result.get("running_successfully"),
            }
        else:
            raise ValueError(f"Unsupported extension: {ext}")
        return out

    def Generate_repair_suggest_by_verification_results(self, user_model: str, tag_name: str, tag_name_for_save: str = None, iscaltoken: bool = False, code_file_stem: str = "code_LLM"):
        """
        Generate repair suggestions based on verification reports (unit test, integration test, and optional fullcode).
        For DFT/ORCA: code file can be .inp; fullcode report may include timeout and running_successfully.
        Loads report_unittest_*_{tag}.jsonl, report_integtest_*_{tag}.jsonl, optional report_fullcode_{tag}.jsonl,
        and the code file (.py, .jl, or .inp), then calls LLM to produce one suggestion per distinct location.
        Output schema is identical to dengken: repair_suggestions list with location, location_description,
        repair_suggestion per item (repair_id added before saving).
        
        Args:
            user_model: Model name to use for LLM
            tag_name: Tag name to identify report files (e.g., "loop1")
            tag_name_for_save: Tag name for saving repair suggestions file (default: same as tag_name)
            iscaltoken: If True, return token_stats in the result dict.
            code_file_stem: Stem of the code file to analyze (default: "code_LLM"). Used to load {stem}.py, .jl or .inp.
        
        Returns:
            Dictionary containing repair_suggestions list and optionally "token_stats" if iscaltoken=True.
        """
        if tag_name_for_save is None:
            tag_name_for_save = tag_name
        # Step 1: Load the verification reports and code file
        work_dir = Path(self.output_dir)
        
        # Find and load report files
        unittest_report_path = next(work_dir.glob(f"report_unittest_*_{tag_name}.jsonl"), None)
        integtest_report_path = next(work_dir.glob(f"report_integtest_*_{tag_name}.jsonl"), None)
        if not unittest_report_path:
            raise FileNotFoundError(f"No report_unittest file found with tag_name: {tag_name}")
        if not integtest_report_path:
            raise FileNotFoundError(f"No report_integtest file found with tag_name: {tag_name}")
        unittest_report = json.loads(unittest_report_path.read_text(encoding="utf-8").strip(), strict=False)
        integtest_report = json.loads(integtest_report_path.read_text(encoding="utf-8").strip(), strict=False)
        
        # Load code file (stem from parameter, e.g. code_LLM_loop1; support .py, .jl, .inp)
        code_llm_path = None
        for ext in (".py", ".jl", ".inp"):
            p = work_dir / f"{code_file_stem}{ext}"
            if p.exists():
                code_llm_path = p
                break
        if code_llm_path is None:
            raise FileNotFoundError(f"{code_file_stem}.py, .jl or .inp not found in {work_dir}")
        code_llm_content = code_llm_path.read_text(encoding="utf-8")
        
        _MAX_FULLCODE_FIELD_CHARS = 5000
        report_fullcode_path = work_dir / f"report_fullcode_{tag_name}.jsonl"
        report_fullcode_data = None
        if report_fullcode_path.exists():
            try:
                raw = report_fullcode_path.read_text(encoding="utf-8").strip()
                if raw:
                    report_fullcode_data = json.loads(raw, strict=False)
                    if isinstance(report_fullcode_data, dict):
                        report_fullcode_data = dict(report_fullcode_data)
                        for key in ("stdout", "stderr"):
                            if key in report_fullcode_data and isinstance(report_fullcode_data[key], str):
                                s = report_fullcode_data[key]
                                if len(s) > _MAX_FULLCODE_FIELD_CHARS:
                                    removed = len(s) - _MAX_FULLCODE_FIELD_CHARS
                                    report_fullcode_data[key] = f"... (truncated: {removed} chars from beginning) ...\n\n" + s[-_MAX_FULLCODE_FIELD_CHARS:]
            except Exception:
                pass
        
        # Step 2: Generate the repair suggestions by LLM
        role_prompt = self._load_prompt(topic="general_dft", prompt_file="RepairCode_suggest_by_verification_results_stand1")
        # Build user prompt by appending inputs
        user_prompt = role_prompt + "\n\nUNIT_TEST_REPORT:\n" + json.dumps(unittest_report, indent=2, ensure_ascii=False)
        user_prompt += "\n\nINTEGRATION_TEST_REPORT:\n" + json.dumps(integtest_report, indent=2, ensure_ascii=False)
        if report_fullcode_data:
            user_prompt += "\n\nFULLCODE_EXECUTION_REPORT:\n" + json.dumps(report_fullcode_data, indent=2, ensure_ascii=False)
        if code_llm_path.suffix.lower() == ".inp":
            user_prompt += "\n\nThe following CODE_LLM is from an ORCA input file (" + code_llm_path.name + ").\n\nCODE_LLM:\n"
        else:
            user_prompt += "\n\nCODE_LLM:\n"
        user_prompt += code_llm_content
        # Call LLM
        if iscaltoken:
            response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
        else:
            response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
            token_dict = None
        parsed_response = self._parse_json(response)
        repair_suggestions = parsed_response.get("repair_suggestions", [])
        
        # Step 3: Attach runtime errors from unit test report to each suggestion
        # Build a lookup: target_file_name -> errors text from judge_result
        _ut_errors_by_target = {}
        for jr in unittest_report.get("judge_result", []):
            tfn = jr.get("target_file_name", "")
            fj = jr.get("final_judge", jr.get("judge", ""))
            if fj in ("wrong", "undetermined") and jr.get("errors"):
                _ut_errors_by_target[tfn] = jr["errors"]
        # Also extract key error lines from fullcode execution report
        _fullcode_error_snippet = ""
        if report_fullcode_data and isinstance(report_fullcode_data, dict):
            for field in ("stderr", "stdout"):
                text = report_fullcode_data.get(field, "")
                if not text:
                    continue
                for line in text.split("\n"):
                    ll = line.strip().lower()
                    if any(kw in ll for kw in ("error", "unknown identifier", "unrecognized", "aborting")):
                        _fullcode_error_snippet += line.strip() + "\n"
            _fullcode_error_snippet = _fullcode_error_snippet.strip()[:1000]

        for i, repair_suggestion in enumerate(repair_suggestions):
            repair_suggestion["repair_id"] = f"rep{i+1}"
            # Attach matching runtime error from unit test or fullcode
            tfn = repair_suggestion.get("target_file_name", "")
            runtime_err = _ut_errors_by_target.get(tfn, "")
            if not runtime_err and _fullcode_error_snippet:
                runtime_err = _fullcode_error_snippet
            if runtime_err:
                repair_suggestion["runtime_error"] = runtime_err[:1000]

        output_dict = {
            "repair_suggestions": repair_suggestions
        }
        if iscaltoken and token_dict:
            output_dict["token_stats"] = token_dict
        output_file = work_dir / f"repair_suggest_{tag_name_for_save}.jsonl"
        pretty_json = json.dumps(output_dict, ensure_ascii=False, indent=2)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(pretty_json + "\n\n")
        
        # Log event
        self._log_event({
            "Content": "Generate repair suggestions by verification reports",
            "LLM": user_model,
            "tag_name": tag_name,
            "output_file": str(output_file),
            "unittest_report_file": str(unittest_report_path),
            "integtest_report_file": str(integtest_report_path),
            "num_suggestions": len(repair_suggestions)
        })
        
        return output_dict


    async def Retrieve_knowledge_for_repair_mcp_use(self, user_model: str, tag_query_dir: str, tag_name: str, concurr_num: int = 5, iscaltoken: bool = False, iscaltool: bool = False) -> dict:
        """
        Retrieve knowledge for repair suggestions using MCP retrieve-mcp server.
        Workflow and I/O match dengken: load repair_suggest_{tag_name}.jsonl, run retrieval per suggestion
        into query_{tag_query_dir}_{repair_id}, summarize to query_summary.jsonl, archive to query_summary_{repair_id}.jsonl.
        For DFT topic (e.g. dft_qc), the prompt restricts retrieval to DFT manuals (orca-manual, etc.). The summary
        step uses a DFT-adapted prompt (RetrieveKnowledgeByPlan_summary_stand1) so summaries suit both API docs and
        software-manual/input-file docs (e.g. ORCA).
        
        Args:
            user_model: Model name to use for LLM
            tag_query_dir: Tag name for query directory (e.g., "test2")
            tag_name: Tag name to identify repair suggestion file (e.g., "test2")
            concurr_num: Number of concurrent retrieval tasks (default: 5)
            iscaltoken: If True, aggregate and return token_stats.
            iscaltool: If True, aggregate and return tool_stats (MCP tool calls).
        
        Returns:
            Dictionary containing retrieval results and metadata; optionally "token_stats" and "tool_stats" when requested.
        """
        # Step 1: Load the repair suggestions
        repair_suggest_path = Path(self.output_dir) / f"repair_suggest_{tag_name}.jsonl"
        if not repair_suggest_path.exists():
            raise FileNotFoundError(f"Repair suggestion file not found: {repair_suggest_path}")
        repair_suggest_data = json.loads(repair_suggest_path.read_text(encoding="utf-8").strip(), strict=False)
        repair_suggestions = repair_suggest_data.get("repair_suggestions", [])
        if not repair_suggestions:
            raise ValueError(f"No repair suggestions found in {repair_suggest_path}")

        # Step 2: Prepare the query directories
        query_base_dir = Path(self.output_dir) / f"query_{tag_query_dir}"
        if query_base_dir.exists():
            shutil.rmtree(query_base_dir)
        query_base_dir.mkdir(parents=True, exist_ok=True)

        repair_subdirs = []
        for repair_suggestion in repair_suggestions:
            repair_id = repair_suggestion.get("repair_id")
            subdir_name = f"query_{tag_query_dir}_{repair_id}"
            subdir_path = query_base_dir / subdir_name
            subdir_path.mkdir(parents=True, exist_ok=True)
            repair_subdirs.append((repair_id, subdir_path, repair_suggestion))

        # Step 3: Implement the retrieval process (including summary, concurrency)
        sem = asyncio.Semaphore(concurr_num)

        async def _process_repair_suggestion(repair_id: str, subdir_path: Path, repair_suggestion: dict):
            task_token_dict = None
            task_tool_stats = None
            async with sem:
                # Load prompt and build user prompt
                role_prompt = self._load_prompt(topic="general_dft", prompt_file="RetrieveKnowledgeForRepair_mcp_use_stand1")
                parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                parts.append("TOPIC:\n")
                parts.append(self.topic)
                parts.append("\n\n")
                parts.append("REPAIR_SUGGESTION:\n")
                parts.append(json.dumps(repair_suggestion, ensure_ascii=False, indent=2))
                parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)
                
                # Call MCP retrieve
                try:
                    if iscaltool:
                        response, task_tool_stats = await self._send_chat_through_mcp_dynamic(
                            user_model=user_model,
                            user_prompt=user_prompt,
                            fs_target_dir=None,
                            retrieve_rt_target_dir=str(subdir_path),
                            init_servers=["retrieve-mcp"],
                            iscaltool=True
                        )
                    else:
                        response = await self._send_chat_through_mcp_dynamic(
                            user_model=user_model,
                            user_prompt=user_prompt,
                            fs_target_dir=None,
                            retrieve_rt_target_dir=str(subdir_path),
                            init_servers=["retrieve-mcp"]
                        )
                except Exception as e:
                    self._log_event({
                        "Content": "Retrieve knowledge for repair error",
                        "repair_id": repair_id,
                        "error": str(e)
                    })
                    return (None, None)
                
                # Load retrieval results and generate summary
                json_files = sorted(subdir_path.glob("*.json"))
                if json_files:
                    all_queries = []
                    for json_file in json_files:
                        try:
                            query_data = json.loads(self._load_file(str(json_file)))
                            all_queries.append(query_data)
                        except Exception as e:
                            self._log_event({
                                "Content": "Failed to load query file",
                                "file": str(json_file),
                                "error": str(e)
                            })
                            continue
                    
                    if all_queries:
                        # Generate summary
                        summary_prompt = self._load_prompt(topic="general_dft", prompt_file="RetrieveKnowledgeByPlan_summary_stand1")
                        parts = [summary_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                        parts.append("ALL_QUERY:\n")
                        parts.append(json.dumps(all_queries, ensure_ascii=False, indent=2))
                        parts.append("\n\n")
                        parts.append("-- END AGGREGATED INPUT --\n")
                        user_prompt = "".join(parts)
                        if iscaltoken:
                            summary_response, task_token_dict = await asyncio.to_thread(self._send_chat_robust, user_model, user_prompt, max_iter=3, iscaltoken=True)
                        else:
                            summary_response = await asyncio.to_thread(self._send_chat_robust, user_model, user_prompt, max_iter=3)
                        await asyncio.to_thread(self._save_jsonl_file, summary_response, anci="query_summary", jsonl_dir=str(subdir_path), isarbi=True)

            return (task_token_dict, task_tool_stats)

        # Execute tasks
        tasks = [asyncio.create_task(_process_repair_suggestion(repair_id, subdir, repair)) for repair_id, subdir, repair in repair_subdirs]
        results = []
        if tasks:
            results = await asyncio.gather(*tasks)

        # Step 4: Archive the retrieval summaries
        for repair_id, subdir_path, _ in repair_subdirs:
            summary_file = subdir_path / "query_summary.jsonl"
            if summary_file.exists():
                dest_file = query_base_dir / f"query_summary_{repair_id}.jsonl"
                shutil.copy2(summary_file, dest_file)

        # Log event and return
        output = {
            "Content": "Retrieve knowledge for repair suggestions",
            "LLM": user_model,
            "tag_query_dir": tag_query_dir,
            "tag_name": tag_name,
            "num_repairs": len(repair_suggestions),
            "query_base_dir": str(query_base_dir)
        }
        if iscaltoken and results:
            token_dicts = [r[0] for r in results if r and r[0]]
            if token_dicts:
                total_input = sum(d.get("input_tokens", 0) for d in token_dicts if d)
                total_output = sum(d.get("output_tokens", 0) for d in token_dicts if d)
                if total_input + total_output > 0:
                    output["token_stats"] = {"input_tokens": total_input, "output_tokens": total_output, "total": total_input + total_output}
        if iscaltool and results:
            tool_stats_list = [r[1] for r in results if r and r[1]]
            if tool_stats_list:
                total_tool_calls = sum(s.get("total_tool_calls", 0) for s in tool_stats_list if s)
                per_tool_calls = {}
                for s in tool_stats_list:
                    if s and s.get("per_tool_calls"):
                        for k, v in s["per_tool_calls"].items():
                            per_tool_calls[k] = per_tool_calls.get(k, 0) + v
                output["tool_stats"] = {"total_tool_calls": total_tool_calls, "per_tool_calls": per_tool_calls}
        self._log_event(output)
        return output


    def Repair_code_by_suggestions(self, user_model: str, tag_name: str, tag_query_dir: str, code_original_name: str, code_repair_name: str, iscaltoken: bool = False) -> dict:
        """
        Repair code by applying repair suggestions sequentially.
        
        Args:
            user_model: Model name to use for LLM
            tag_name: Tag name to identify repair suggestion file (e.g., "test2")
            tag_query_dir: Tag name for query directory (e.g., "test2")
            code_original_name: Name of the original code file (without extension)
            code_repair_name: Name for the repaired code file (without extension)
            iscaltoken: If True, aggregate and return token_stats.
        
        Returns:
            Dictionary containing repair results and metadata; optionally "token_stats" when iscaltoken=True.
        """
        # Step 1: Load the repair suggestions and original code
        work_dir = Path(self.output_dir)
        
        # Load repair suggestions
        repair_suggest_path = work_dir / f"repair_suggest_{tag_name}.jsonl"
        if not repair_suggest_path.exists():
            raise FileNotFoundError(f"Repair suggestion file not found: {repair_suggest_path}")
        repair_suggest_data = json.loads(repair_suggest_path.read_text(encoding="utf-8").strip(), strict=False)
        repair_suggestions = repair_suggest_data.get("repair_suggestions", [])
        if not repair_suggestions:
            raise ValueError(f"No repair suggestions found in {repair_suggest_path}")
        
        # Find and load original code file
        code_original_path = None
        code_ext = None
        for file in work_dir.iterdir():
            if file.is_file() and file.stem == code_original_name:
                code_original_path = file
                code_ext = file.suffix  # .py or .jl
                break
        if code_original_path is None:
            raise FileNotFoundError(f"Original code file '{code_original_name}' not found in {work_dir}")
        code_original_content = code_original_path.read_text(encoding="utf-8", errors="replace")
        
        # Determine code_type from extension (.inp = ORCA input for DFT)
        if code_ext == ".py":
            code_type = "python"
        elif code_ext == ".jl":
            code_type = "julia"
        elif code_ext == ".inp":
            code_type = "orca"
        else:
            code_type = "python"

        # Step 2: Loop of repairing code by suggestions and retrieve summaries
        # Load prompt
        role_prompt = self._load_prompt(topic="general_dft", prompt_file="RepairCodeBySuggestions_Query_instruct_stand1")
        # Initialize current code content
        current_code_content = code_original_content
        total_input = 0
        total_output = 0
        
        # Loop through each repair suggestion
        for repair_suggestion in repair_suggestions:
            repair_id = repair_suggestion.get("repair_id")
            # Load query summary
            query_summary_path = work_dir / f"query_{tag_query_dir}" / f"query_summary_{repair_id}.jsonl"
            query_summary_data = None
            if query_summary_path.exists():
                query_summary_data = json.loads(query_summary_path.read_text(encoding="utf-8").strip(), strict=False)
            # Extract repair suggestion fields
            repair_info = {
                "location": repair_suggestion.get("location", ""),
                "location_description": repair_suggestion.get("location_description", ""),
                "repair_suggestion": repair_suggestion.get("repair_suggestion", ""),
                "repair_id": repair_id
            }
            
            # Build user prompt
            parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            constraints = self._load_topic_language_packages()
            if constraints:
                parts.append(self._format_topic_constraints_input(constraints))
            parts.append("ORIGINAL_CODE:\n")
            parts.append(current_code_content)
            parts.append("\n\n")
            parts.append("QUERY_SUMMARY:\n")
            parts.append(json.dumps(query_summary_data, ensure_ascii=False, indent=2))
            parts.append("\n\n")
            parts.append("REPAIR_SUGGESTION:\n")
            parts.append(json.dumps(repair_info, ensure_ascii=False, indent=2))
            parts.append("\n\n")                
            parts.append("-- END AGGREGATED INPUT --\n")
            user_prompt = "".join(parts)
            if iscaltoken:
                response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True, require_code_block=True, code_tags=["CODE"])
                if token_dict:
                    total_input += token_dict.get("input_tokens", 0)
                    total_output += token_dict.get("output_tokens", 0)
            else:
                response = self._send_chat_robust(user_model, user_prompt, max_iter=3, require_code_block=True, code_tags=["CODE"])
            save_output = self._save_code_file(
                response=response,
                anci=code_repair_name,
                code_dir=str(self.output_dir),
                isarbi=True,
                file_ext=code_ext,
                content_tag="CODE",
            )
            
            if save_output:
                current_code_content = save_output.get("code_content", current_code_content)
                print(f"Repaired code for {repair_id} saved to {save_output.get('code_file')}")
            else:
                self._log_event({"Content": "Failed to save repaired code", "repair_id": repair_id})

        # Step 3: Log event and return
        # Determine final code file path
        final_code_file = work_dir / f"{code_repair_name}{code_ext}"
        
        output = {
            "Content": "Repair code by suggestions",
            "LLM": user_model,
            "tag_name": tag_name,
            "tag_query_dir": tag_query_dir,
            "code_original_name": code_original_name,
            "code_repair_name": code_repair_name,
            "num_repairs": len(repair_suggestions),
            "code_file": str(final_code_file) if final_code_file.exists() else None
        }
        if iscaltoken and (total_input > 0 or total_output > 0):
            output["token_stats"] = {"input_tokens": total_input, "output_tokens": total_output, "total": total_input + total_output}
        self._log_event(output)
        return output

    def Refine_repair_suggestions_by_query(self, user_model: str, tag_name: str, tag_query_dir: str, iscaltoken: bool = False) -> dict:
        """
        Refine each repair_suggestion using QUERY_SUMMARY; overwrite repair_suggest_{tag_name}.jsonl once at the end.
        For DFT topic, QUERY_SUMMARY may come from software manuals (e.g. ORCA); the refine prompt is adapted so
        revised suggestions align with manual-based input syntax/keywords as well as API usage.
        Returns token_stats for merging into suggest token stats in coderepair_v7.
        """
        work_dir = Path(self.output_dir)
        repair_suggest_path = work_dir / f"repair_suggest_{tag_name}.jsonl"
        repair_suggest_data = json.loads(repair_suggest_path.read_text(encoding="utf-8").strip(), strict=False)
        repair_suggestions = repair_suggest_data.get("repair_suggestions", [])
        if not repair_suggestions:
            return {"token_stats": {"input_tokens": 0, "output_tokens": 0, "total": 0}} if iscaltoken else {}

        role_prompt = self._load_prompt(topic="general_dft", prompt_file="RefineRepairSuggestionByQuery_stand1")
        query_base = work_dir / f"query_{tag_query_dir}"
        total_input, total_output = 0, 0

        for item in repair_suggestions:
            repair_id = item.get("repair_id")
            query_summary_path = query_base / f"query_summary_{repair_id}.jsonl"
            if not query_summary_path.exists():
                continue
            print(f"Refine suggest: {repair_id} ...")
            query_summary_data = json.loads(query_summary_path.read_text(encoding="utf-8").strip(), strict=False)
            current_entry = {"location": item.get("location", ""), "location_description": item.get("location_description", ""), "repair_suggestion": item.get("repair_suggestion", "")}
            if item.get("runtime_error"):
                current_entry["runtime_error"] = item["runtime_error"]
            user_prompt = role_prompt + "\n\nCURRENT_ENTRY:\n" + json.dumps(current_entry, ensure_ascii=False, indent=2) + "\n\nQUERY_SUMMARY:\n" + json.dumps(query_summary_data, ensure_ascii=False, indent=2)
            if iscaltoken:
                response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
                if token_dict:
                    total_input += token_dict.get("input_tokens", 0)
                    total_output += token_dict.get("output_tokens", 0)
            else:
                response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
            parsed = self._parse_json(response)
            if parsed is not None:
                first = (parsed.get("repair_suggestions") or [{}])[0]
                if first.get("repair_suggestion"):
                    # Safety net: if a runtime_error exists and the original
                    # suggestion says remove/delete but the refinement reverses
                    # it to keep/do-not-remove, reject the refinement.
                    has_runtime_err = bool(item.get("runtime_error"))
                    if has_runtime_err:
                        orig_low = item.get("repair_suggestion", "").lower()
                        refined_low = first["repair_suggestion"].lower()
                        orig_says_remove = any(w in orig_low for w in ("remove", "delete"))
                        refined_says_keep = any(w in refined_low for w in (
                            "keep", "do not remove", "don't remove", "do not delete",
                            "don't delete", "retain", "restore",
                        ))
                        if orig_says_remove and refined_says_keep:
                            print(f"  WARNING: Refinement reversed runtime-error removal "
                                  f"for {repair_id}. Keeping original suggestion.")
                            continue
                    item["repair_suggestion"] = first["repair_suggestion"]

        # Overwrite repair_suggest file once: same list (items updated in place), preserve original token_stats if present
        out_dict = {"repair_suggestions": repair_suggestions}
        if "token_stats" in repair_suggest_data:
            out_dict["token_stats"] = repair_suggest_data["token_stats"]
        repair_suggest_path.write_text(json.dumps(out_dict, ensure_ascii=False, indent=2) + "\n\n", encoding="utf-8")

        out = {"token_stats": {"input_tokens": total_input, "output_tokens": total_output, "total": total_input + total_output}} if iscaltoken else {}
        self._log_event({"Content": "Refine_repair_suggestions_by_query", "tag_name": tag_name, **out})
        return out



