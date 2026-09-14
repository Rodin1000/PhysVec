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


class repair_agents(QMBagent):
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
        """Execute code file (py/jl) and return {exitcode, stdout, stderr}."""
        path = Path(code_file)
        project_root = str(path.parent)
        ext = path.suffix.lower()
        if ext == ".py":
            result = run_program.run_python_file(str(path), project_root, timeout)
        elif ext == ".jl":
            result = run_program.run_julia_file(str(path), project_root, timeout)
        else:
            if self.current_log_file:
                self._log_event({"Content": "Execute_fullcode failed", "error_type": "ValueError", "error": f"Unsupported extension: {ext}"})
            raise ValueError(f"Unsupported extension: {ext}")
        return {
            "exitcode": result.get("exitcode"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }

    def Merge_fullparas_into_smallscale(
        self,
        user_model: str,
        iscaltoken: bool = False,
        smallscale_path: str | Path = None,
        fullparas_path: str | Path = None,
        output_stem: str = None,
    ) -> dict:
        """
        Merge fullparas main parameters into smallscale structure.
        If smallscale_path, fullparas_path, output_stem are all provided, use them; else use fixed code_LLM_smallscale, code_LLM_fullparas, code_LLM_loop1.
        Returns dict with output path and optionally token_stats.
        """
        work_dir = Path(self.output_dir)
        if smallscale_path is not None and fullparas_path is not None and output_stem is not None:
            smallscale_p = Path(smallscale_path)
            fullparas_p = Path(fullparas_path)
            if not smallscale_p.exists():
                self._log_event({"Content": "Merge_fullparas_into_smallscale failed", "error_type": "FileNotFoundError", "error": f"smallscale not found: {smallscale_path}"})
                raise FileNotFoundError(f"smallscale not found: {smallscale_path}")
            if not fullparas_p.exists():
                self._log_event({"Content": "Merge_fullparas_into_smallscale failed", "error_type": "FileNotFoundError", "error": f"fullparas not found: {fullparas_path}"})
                raise FileNotFoundError(f"fullparas not found: {fullparas_path}")
            code_smallscale = smallscale_p.read_text(encoding="utf-8")
            code_fullparas = fullparas_p.read_text(encoding="utf-8")
            ext = smallscale_p.suffix
            out_dir = str(smallscale_p.parent)
        else:
            smallscale_py = work_dir / "code_LLM_smallscale.py"
            smallscale_jl = work_dir / "code_LLM_smallscale.jl"
            if smallscale_py.exists():
                smallscale_p = smallscale_py
                fullparas_p = work_dir / "code_LLM_fullparas.py"
            elif smallscale_jl.exists():
                smallscale_p = smallscale_jl
                fullparas_p = work_dir / "code_LLM_fullparas.jl"
            else:
                self._log_event({"Content": "Merge_fullparas_into_smallscale failed", "error_type": "FileNotFoundError", "error": "code_LLM_smallscale.py or .jl not found"})
                raise FileNotFoundError("code_LLM_smallscale.py or .jl not found in output_dir")
            if not fullparas_p.exists():
                self._log_event({"Content": "Merge_fullparas_into_smallscale failed", "error_type": "FileNotFoundError", "error": f"{fullparas_p.name} not found"})
                raise FileNotFoundError(f"{fullparas_p.name} not found in output_dir")
            code_smallscale = smallscale_p.read_text(encoding="utf-8")
            code_fullparas = fullparas_p.read_text(encoding="utf-8")
            ext = smallscale_p.suffix
            output_stem = "code_LLM_loop1"
            out_dir = str(work_dir)

        role_prompt = self._load_prompt(topic="general", prompt_file="MergeFullparasIntoSmallscale_stand1")
        user_prompt = role_prompt + "\n\nCODE_SMALLSCALE:\n" + code_smallscale + "\n\nCODE_FULLPARAS:\n" + code_fullparas
        try:
            if iscaltoken:
                response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True, require_code_block=True, code_tags=["CODE"])
            else:
                response = self._send_chat_robust(user_model, user_prompt, max_iter=3, require_code_block=True, code_tags=["CODE"])
                token_dict = None
            save_output = self._save_code_file(
                response=response,
                anci=output_stem,
                code_dir=out_dir,
                isarbi=True,
                content_tag="CODE",
                file_ext=ext.lstrip("."),
            )
        except Exception as e:
            self._log_event({"Content": "Merge_fullparas_into_smallscale failed", "error_type": type(e).__name__, "error": str(e)})
            raise
        out_path = Path(out_dir) / f"{output_stem}{ext}" if save_output else None
        output = {"code_file": str(out_path) if out_path and out_path.exists() else None}
        if iscaltoken and token_dict:
            output["token_stats"] = token_dict
        self._log_event({"Content": "Merge_fullparas_into_smallscale", "output_file": output.get("code_file"), "LLM": user_model})
        return output

    def Generate_repair_suggest_by_rubrics_report(self, user_model: str, tag_name: str, tag_name_for_save: str = None, code_file_stem: str = "code_LLM_loop1", iscaltoken: bool = False) -> dict:
        """
        Generate repair suggestions from report_rubrics_*_{tag_name}.jsonl (Scorecard.error_issues and optionally LimitingCases/ScientificResults).
        Writes repair_suggest_{tag_name_for_save}.jsonl with same structure as Generate_repair_suggest_by_verification_results.
        """
        if tag_name_for_save is None:
            tag_name_for_save = tag_name
        work_dir = Path(self.output_dir)
        report_path = next(work_dir.glob(f"report_rubrics_*_{tag_name}.jsonl"), None)
        if not report_path or not report_path.exists():
            self._log_event({"Content": "Generate_repair_suggest_by_rubrics_report failed", "error_type": "FileNotFoundError", "error": f"No report_rubrics_*_{tag_name}.jsonl in {work_dir}"})
            raise FileNotFoundError(f"No report_rubrics_*_{tag_name}.jsonl in output_dir")
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8").strip(), strict=False)
        except Exception as e:
            self._log_event({"Content": "Generate_repair_suggest_by_rubrics_report failed", "error_type": type(e).__name__, "error": str(e)})
            raise
        role_prompt = self._load_prompt(topic="general", prompt_file="GenerateRepairSuggestByRubricsReport_stand1")
        user_prompt = role_prompt + "\n\nREPORT_RUBRICS:\n" + json.dumps(report_data, ensure_ascii=False, indent=2) + "\n\nCODE_FILE_STEM:\n" + code_file_stem
        try:
            if iscaltoken:
                response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
            else:
                response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
                token_dict = None
            parsed = self._parse_json(response)
            repair_suggestions = parsed.get("repair_suggestions", [])
        except Exception as e:
            self._log_event({"Content": "Generate_repair_suggest_by_rubrics_report failed", "error_type": type(e).__name__, "error": str(e)})
            raise
        for i, item in enumerate(repair_suggestions):
            item["repair_id"] = item.get("repair_id") or f"rep{i+1}"
        output_dict = {"repair_suggestions": repair_suggestions}
        if iscaltoken and token_dict:
            output_dict["token_stats"] = token_dict
        output_file = work_dir / f"repair_suggest_{tag_name_for_save}.jsonl"
        output_file.write_text(json.dumps(output_dict, ensure_ascii=False, indent=2) + "\n\n", encoding="utf-8")
        self._log_event({
            "Content": "Generate_repair_suggest_by_rubrics_report",
            "LLM": user_model,
            "tag_name": tag_name,
            "output_file": str(output_file),
            "report_file": str(report_path),
            "num_suggestions": len(repair_suggestions),
        })
        return output_dict

    def Generate_repair_suggest_by_rubrics_report_seperate(self, user_model: str, tag_name: str, tag_name_for_save: str = None, code_file_stem: str = "code_LLM_loop1", iscaltoken: bool = False) -> dict:
        """
        Generate repair suggestions from rubrics report in two steps:
        Step 1: Generate suggestions for main code (Scorecard.error_issues)
        Step 2: Analyze each LimitingCase to determine if problem is in test code

        Returns:
            {
                "original_file": str,
                "original_count": int,
                "limitingcase_files": list[str],
                "limitingcase_count": int,
                "token_stats": dict or None
            }
        """
        if tag_name_for_save is None:
            tag_name_for_save = tag_name
        work_dir = Path(self.output_dir)

        # Load report
        report_path = next(work_dir.glob(f"report_rubrics_*_{tag_name}.jsonl"), None)
        if not report_path or not report_path.exists():
            self._log_event({"Content": "Generate_repair_suggest_by_rubrics_report_separate failed", "error_type": "FileNotFoundError", "error": f"No report_rubrics_*_{tag_name}.jsonl in {work_dir}"})
            raise FileNotFoundError(f"No report_rubrics_*_{tag_name}.jsonl in output_dir")
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8").strip(), strict=False)
        except Exception as e:
            self._log_event({"Content": "Generate_repair_suggest_by_rubrics_report_separate failed", "error_type": type(e).__name__, "error": str(e)})
            raise

        # ========== STEP 1: Generate Original Suggestions ==========
        role_prompt_original = self._load_prompt(topic="general", prompt_file="GenerateRepairSuggestByRubricsReport_original_stand1")
        user_prompt_original = role_prompt_original + "\n\nREPORT_RUBRICS:\n" + json.dumps(report_data, ensure_ascii=False, indent=2) + "\n\nCODE_FILE_STEM:\n" + code_file_stem

        try:
            if iscaltoken:
                response_original, token_dict_original = self._send_chat_robust(user_model, user_prompt_original, max_iter=3, iscaltoken=True)
            else:
                response_original = self._send_chat_robust(user_model, user_prompt_original, max_iter=3)
                token_dict_original = None
            parsed_original = self._parse_json(response_original)
            repair_suggestions_original = parsed_original.get("repair_suggestions", [])
        except Exception as e:
            self._log_event({"Content": "Generate original suggestions failed", "error_type": type(e).__name__, "error": str(e)})
            raise

        # Assign repair_id
        for i, item in enumerate(repair_suggestions_original):
            item["repair_id"] = item.get("repair_id") or f"rep{i+1}"

        # Save to file (keep original filename format)
        output_dict_original = {"repair_suggestions": repair_suggestions_original}
        if iscaltoken and token_dict_original:
            output_dict_original["token_stats"] = token_dict_original
        output_file_original = work_dir / f"repair_suggest_{tag_name_for_save}.jsonl"
        output_file_original.write_text(json.dumps(output_dict_original, ensure_ascii=False, indent=2) + "\n\n", encoding="utf-8")

        # ========== STEP 2: Analyze LimitingCases ==========
        limitingcase_files = []
        total_input_tokens = token_dict_original.get("input_tokens", 0) if token_dict_original else 0
        total_output_tokens = token_dict_original.get("output_tokens", 0) if token_dict_original else 0

        limiting_cases = report_data.get("LimitingCases", {}).get("results", [])
        role_prompt_limitingcase = self._load_prompt(topic="general", prompt_file="GenerateRepairSuggestByRubricsReport_limitingcase_stand1")

        for idx, case in enumerate(limiting_cases):
            if case.get("judge") == "correct":
                continue

            case_data = {
                "content": case.get("content"),
                "judge": case.get("judge"),
                "final_exitcode": case.get("final_exitcode"),
                "reason": case.get("reason"),
            }

            user_prompt_limitingcase = role_prompt_limitingcase + "\n\nLIMITING_CASE_DATA:\n" + json.dumps(case_data, ensure_ascii=False, indent=2) + "\n\nORIGINAL_CODE_STEM:\n" + code_file_stem

            try:
                if iscaltoken:
                    response_limitingcase, token_dict_limitingcase = self._send_chat_robust(user_model, user_prompt_limitingcase, max_iter=3, iscaltoken=True)
                else:
                    response_limitingcase = self._send_chat_robust(user_model, user_prompt_limitingcase, max_iter=3)
                    token_dict_limitingcase = None

                parsed_limitingcase = self._parse_json(response_limitingcase)
                is_limiting = parsed_limitingcase.get("islimiting") == "True"
                repair_suggestions_text = parsed_limitingcase.get("repair_suggestions")

                if iscaltoken and token_dict_limitingcase:
                    total_input_tokens += token_dict_limitingcase.get("input_tokens", 0)
                    total_output_tokens += token_dict_limitingcase.get("output_tokens", 0)

                if is_limiting and repair_suggestions_text:
                    output_dict_limiting = {
                        "case_index": idx,
                        "case_content": case.get("content"),
                        "repair_suggestions": repair_suggestions_text
                    }
                    if iscaltoken and token_dict_limitingcase:
                        output_dict_limiting["token_stats"] = token_dict_limitingcase

                    output_file_limiting = work_dir / f"limitingcase_{idx}_suggest_{tag_name_for_save}.jsonl"
                    output_file_limiting.write_text(json.dumps(output_dict_limiting, ensure_ascii=False, indent=2) + "\n\n", encoding="utf-8")
                    limitingcase_files.append(str(output_file_limiting))

            except Exception as e:
                self._log_event({"Content": "Analyze limiting case failed", "case_index": idx, "error": str(e)})
                continue

        # ========== Return Results ==========
        token_stats_total = None
        if iscaltoken:
            token_stats_total = {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens, "total": total_input_tokens + total_output_tokens}

        output = {
            "original_file": str(output_file_original),
            "original_count": len(repair_suggestions_original),
            "limitingcase_files": limitingcase_files,
            "limitingcase_count": len(limitingcase_files),
            "token_stats": token_stats_total
        }

        self._log_event({
            "Content": "Generate_repair_suggest_by_rubrics_report_separate",
            "LLM": user_model,
            "tag_name": tag_name,
            "original_file": str(output_file_original),
            "limitingcase_files": limitingcase_files,
            "report_file": str(report_path),
            "original_count": len(repair_suggestions_original),
            "limitingcase_count": len(limitingcase_files),
        })
        return output

    def Generate_repair_suggest_by_verification_results(self, user_model: str, tag_name: str, tag_name_for_save: str = None, iscaltoken: bool = False, code_file_stem: str = "code_LLM"):
        """
        Generate repair suggestions based on verification reports (unit test and integration test).
        
        Args:
            user_model: Model name to use for LLM
            tag_name: Tag name to identify report files (e.g., "test2")
            tag_name_for_save: Tag name for saving repair suggestions file (default: same as tag_name)
            iscaltoken: If True, return token_stats in the result dict.
            code_file_stem: Stem of the code file to analyze (default: "code_LLM"). Used to load {stem}.py or {stem}.jl.
        
        Returns:
            Dictionary containing repair suggestions and metadata; optionally "token_stats" if iscaltoken=True.
        """
        if tag_name_for_save is None:
            tag_name_for_save = tag_name
        # Step 1: Load the verification reports and code file
        work_dir = Path(self.output_dir)
        
        # Find and load report files
        unittest_report_path = next(work_dir.glob(f"report_unittest_*_{tag_name}.jsonl"), None)
        integtest_report_path = next(work_dir.glob(f"report_integtest_*_{tag_name}.jsonl"), None)
        unittest_report = None
        if unittest_report_path:
            unittest_report = json.loads(unittest_report_path.read_text(encoding="utf-8").strip(), strict=False)
        integtest_report = None
        if integtest_report_path:
            integtest_report = json.loads(integtest_report_path.read_text(encoding="utf-8").strip(), strict=False)
        
        # Load code file (stem from parameter, e.g. code_LLM_loop1)
        code_llm_path = work_dir / f"{code_file_stem}.py" if (work_dir / f"{code_file_stem}.py").exists() else work_dir / f"{code_file_stem}.jl"
        if not code_llm_path.exists():
            self._log_event({"Content": "Generate repair suggest failed", "error_type": "FileNotFoundError", "error": f"{code_file_stem}.py or {code_file_stem}.jl not found in {work_dir}"})
            raise FileNotFoundError(f"{code_file_stem}.py or {code_file_stem}.jl not found in {work_dir}")
        code_llm_content = code_llm_path.read_text(encoding="utf-8")
        
        # Optional: load report_fullcode if exists
        report_fullcode_path = work_dir / f"report_fullcode_{tag_name}.jsonl"
        report_fullcode_data = None
        if report_fullcode_path.exists():
            try:
                raw = report_fullcode_path.read_text(encoding="utf-8").strip()
                if raw:
                    report_fullcode_data = json.loads(raw, strict=False)
            except Exception:
                pass
        
        # Step 2: Generate the repair suggestions by LLM
        role_prompt = self._load_prompt(topic="general", prompt_file="RepairCode_suggest_by_verification_results_stand1")
        # Build user prompt by appending inputs
        if unittest_report is not None:
            user_prompt = role_prompt + "\n\nUNIT_TEST_REPORT:\n" + json.dumps(unittest_report, indent=2, ensure_ascii=False)
        else:
            user_prompt = role_prompt + "\n\nUNIT_TEST_REPORT: not available (unittest skipped or failed)."
        if integtest_report is not None:
            user_prompt += "\n\nINTEGRATION_TEST_REPORT:\n" + json.dumps(integtest_report, indent=2, ensure_ascii=False)
        else:
            user_prompt += "\n\nINTEGRATION_TEST_REPORT: not available (integtest skipped or failed)."
        if report_fullcode_data:
            user_prompt += "\n\nFULLCODE_EXECUTION_REPORT:\n" + json.dumps(report_fullcode_data, indent=2, ensure_ascii=False)
        user_prompt += "\n\nCODE_LLM:\n" + code_llm_content
        # Call LLM and save
        try:
            if iscaltoken:
                response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
            else:
                response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
                token_dict = None
            parsed_response = self._parse_json(response)
            repair_suggestions = parsed_response.get("repair_suggestions", [])
        except Exception as e:
            self._log_event({"Content": "Generate repair suggest failed", "error_type": type(e).__name__, "error": str(e)})
            raise
        # Step 3: Save the repair suggestions
        for i, repair_suggestion in enumerate(repair_suggestions):
            repair_suggestion["repair_id"] = f"rep{i+1}"
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
            "unittest_report_file": str(unittest_report_path) if unittest_report_path else "not available",
            "integtest_report_file": str(integtest_report_path) if integtest_report_path else "not available",
            "num_suggestions": len(repair_suggestions)
        })
        
        return output_dict


    async def Retrieve_knowledge_for_repair_mcp_use(self, user_model: str, tag_query_dir: str, tag_name: str, concurr_num: int = 5, iscaltoken: bool = False, iscaltool: bool = False) -> dict:
        """
        Retrieve knowledge for repair suggestions using MCP retrieve-mcp server.
        
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
            self._log_event({"Content": "Retrieve knowledge failed", "error_type": "FileNotFoundError", "error": f"Repair suggestion file not found: {repair_suggest_path}"})
            raise FileNotFoundError(f"Repair suggestion file not found: {repair_suggest_path}")
        repair_suggest_data = json.loads(repair_suggest_path.read_text(encoding="utf-8").strip(), strict=False)
        repair_suggestions = repair_suggest_data.get("repair_suggestions", [])
        if not repair_suggestions:
            self._log_event({"Content": "Retrieve knowledge failed", "error_type": "ValueError", "error": f"No repair suggestions found in {repair_suggest_path}"})
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
                role_prompt = self._load_prompt(topic="general", prompt_file="RetrieveKnowledgeForRepair_mcp_use_stand1")
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
                        summary_prompt = self._load_prompt(topic="general", prompt_file="RetrieveKnowledgeByPlan_summary_stand1")
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
        try:
            for repair_id, subdir_path, _ in repair_subdirs:
                summary_file = subdir_path / "query_summary.jsonl"
                if summary_file.exists():
                    dest_file = query_base_dir / f"query_summary_{repair_id}.jsonl"
                    shutil.copy2(summary_file, dest_file)
        except Exception as e:
            self._log_event({"Content": "Retrieve knowledge failed (archive summaries)", "error_type": type(e).__name__, "error": str(e)})
            raise

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
            self._log_event({"Content": "Repair code by suggestions failed", "error_type": "FileNotFoundError", "error": f"Repair suggestion file not found: {repair_suggest_path}"})
            raise FileNotFoundError(f"Repair suggestion file not found: {repair_suggest_path}")
        repair_suggest_data = json.loads(repair_suggest_path.read_text(encoding="utf-8").strip(), strict=False)
        repair_suggestions = repair_suggest_data.get("repair_suggestions", [])
        if not repair_suggestions:
            self._log_event({"Content": "Repair code by suggestions failed", "error_type": "ValueError", "error": f"No repair suggestions found in {repair_suggest_path}"})
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
            self._log_event({"Content": "Repair code by suggestions failed", "error_type": "FileNotFoundError", "error": f"Original code file '{code_original_name}' not found in {work_dir}"})
            raise FileNotFoundError(f"Original code file '{code_original_name}' not found in {work_dir}")
        code_original_content = code_original_path.read_text(encoding="utf-8")
        
        # Determine code_type from extension
        code_type = "python" if code_ext == ".py" else "julia"

        # Step 2: Loop of repairing code by suggestions and retrieve summaries
        # Load prompt
        role_prompt = self._load_prompt(topic="general", prompt_file="RepairCodeBySuggestions_Query_instruct_stand1")
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
            try:
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
                    content_tag="CODE",
                )
            except Exception as e:
                self._log_event({"Content": "Repair code by suggestions failed", "repair_id": repair_id, "error_type": type(e).__name__, "error": str(e)})
                raise
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

    def Execute_and_refine_code(
        self,
        code_file: str,
        user_model: str,
        max_refine_iter: int = 3,
        timeout: int = 600,
        iscaltoken: bool = False,
    ) -> dict:
        """
        Execute code; if exitcode!=0, refine via LLM (minimal edits) and repeat until success or max_refine_iter.
        Returns {exitcode, stdout, stderr, code_file, refine_iterations, token_stats}.
        """
        path = Path(code_file)
        if not path.exists():
            if self.current_log_file:
                self._log_event({"Content": "Execute_and_refine_code failed", "reason": "code_file not found", "code_file": code_file})
            return {"exitcode": None, "stdout": "", "stderr": "", "code_file": None, "refine_iterations": 0, "token_stats": None}
        ext = path.suffix.lower()
        if ext not in (".py", ".jl"):
            if self.current_log_file:
                self._log_event({"Content": "Execute_and_refine_code failed", "reason": f"unsupported extension {ext}"})
            return {"exitcode": None, "stdout": "", "stderr": "", "code_file": None, "refine_iterations": 0, "token_stats": None}
        code_dir = str(path.parent)
        file_ext = ext.lstrip(".")

        def _run(code_path: Path) -> dict:
            proot = str(code_path.parent)
            if ext == ".py":
                return run_program.run_python_file(str(code_path), proot, timeout)
            return run_program.run_julia_file(str(code_path), proot, timeout)

        current_code_file = str(path)
        exitcode = None
        stdout = ""
        stderr = ""
        refine_count = 0
        token_list = []

        while True:
            result = _run(Path(current_code_file))
            exitcode = result.get("exitcode")
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")

            if self.current_log_file:
                run_iter = refine_count
                self._log_event({
                    "Content": "Execute_and_refine_code run",
                    "code_file": current_code_file,
                    "exitcode": exitcode,
                    "stdout": (stdout[:2000] + "...") if len(stdout) > 2000 else stdout,
                    "stderr": (stderr[:2000] + "...") if len(stderr) > 2000 else stderr,
                    "run_iter": run_iter,
                })

            if exitcode == 0:
                break
            if refine_count >= max_refine_iter:
                break

            try:
                current_code_content = self._load_file(current_code_file)
                role_prompt = self._load_prompt(topic="general", prompt_file="RefineExecuteCode_stand1")
                parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                parts.append("CODE_CONTENT:\n")
                parts.append(current_code_content)
                parts.append("\n\nSTDOUT:\n")
                parts.append(json.dumps(stdout, ensure_ascii=False))
                parts.append("\n\nSTDERR:\n")
                parts.append(json.dumps(stderr, ensure_ascii=False))
                parts.append("\n\nERROR_ANALYSIS:\n")
                parts.append(json.dumps(f"Exit code: {exitcode}.", ensure_ascii=False))
                parts.append("\n\nTOPIC:\n")
                parts.append(json.dumps(self.topic or "", ensure_ascii=False))
                parts.append("\n-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)

                if iscaltoken:
                    response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True, require_code_block=True, code_tags=["CODE"])
                    if token_dict:
                        token_list.append(token_dict)
                else:
                    response = self._send_chat_robust(user_model, user_prompt, max_iter=3, require_code_block=True, code_tags=["CODE"])

                save_result = self._save_code_file(
                    response,
                    anci=path.stem,
                    code_dir=code_dir,
                    isarbi=True,
                    content_tag="CODE",
                    file_ext=file_ext,
                )
                if save_result is None:
                    if self.current_log_file:
                        self._log_event({"Content": "Execute_and_refine_code save failed", "refine_count": refine_count})
                    break
                current_code_file = save_result.get("code_file")
                refine_count += 1
                if self.current_log_file and current_code_file:
                    self._log_event({"Content": "Execute_and_refine_code refined", "code_file": current_code_file, "refine_count": refine_count})
            except Exception as e:
                if self.current_log_file:
                    self._log_event({"Content": "Execute_and_refine_code refine exception", "error": str(e), "refine_count": refine_count})
                break

        token_stats = None
        if iscaltoken and token_list:
            ti = sum(t.get("input_tokens", 0) for t in token_list)
            to = sum(t.get("output_tokens", 0) for t in token_list)
            if ti + to > 0:
                token_stats = {"input_tokens": ti, "output_tokens": to, "total": ti + to}

        return {
            "exitcode": exitcode,
            "stdout": stdout,
            "stderr": stderr,
            "code_file": current_code_file,
            "refine_iterations": refine_count,
            "token_stats": token_stats,
        }

    def Refine_repair_suggestions_by_query(self, user_model: str, tag_name: str, tag_query_dir: str, iscaltoken: bool = False) -> dict:
        """
        Refine each repair_suggestion using QUERY_SUMMARY; overwrite repair_suggest_{tag_name}.jsonl once at the end.
        Returns token_stats for merging into suggest token stats in coderepair_v7.
        """
        try:
            work_dir = Path(self.output_dir)
            repair_suggest_path = work_dir / f"repair_suggest_{tag_name}.jsonl"
            repair_suggest_data = json.loads(repair_suggest_path.read_text(encoding="utf-8").strip(), strict=False)
            repair_suggestions = repair_suggest_data.get("repair_suggestions", [])
            if not repair_suggestions:
                return {"token_stats": {"input_tokens": 0, "output_tokens": 0, "total": 0}} if iscaltoken else {}

            role_prompt = self._load_prompt(topic="general", prompt_file="RefineRepairSuggestionByQuery_stand1")
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
                user_prompt = role_prompt + "\n\nCURRENT_ENTRY:\n" + json.dumps(current_entry, ensure_ascii=False, indent=2) + "\n\nQUERY_SUMMARY:\n" + json.dumps(query_summary_data, ensure_ascii=False, indent=2)
                if iscaltoken:
                    response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
                    if token_dict:
                        total_input += token_dict.get("input_tokens", 0)
                        total_output += token_dict.get("output_tokens", 0)
                else:
                    response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
                parsed = self._parse_json(response)
                first = (parsed.get("repair_suggestions") or [{}])[0]
                if first.get("repair_suggestion"):
                    item["repair_suggestion"] = first["repair_suggestion"]

            # Overwrite repair_suggest file once: same list (items updated in place), preserve original token_stats if present
            out_dict = {"repair_suggestions": repair_suggestions}
            if "token_stats" in repair_suggest_data:
                out_dict["token_stats"] = repair_suggest_data["token_stats"]
            repair_suggest_path.write_text(json.dumps(out_dict, ensure_ascii=False, indent=2) + "\n\n", encoding="utf-8")

            out = {"token_stats": {"input_tokens": total_input, "output_tokens": total_output, "total": total_input + total_output}} if iscaltoken else {}
            self._log_event({"Content": "Refine_repair_suggestions_by_query", "tag_name": tag_name, **out})
            return out
        except Exception as e:
            self._log_event({"Content": "Refine_repair_suggestions_by_query failed", "error_type": type(e).__name__, "error": str(e)})
            raise



