# ReAct agents: baseline agent with Generate / Execute / Suggest / Repair
import json
from pathlib import Path

from config import OPENROUTER_API_KEY
from src import DFT_agents
from utils import run_program


class ReAct_dft_agent(DFT_agents.DFT_agent):
    """Agent for ReAct baseline workflow: generate code, execute, suggest, repair."""

    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        prompt_dir: str = "../prompts",
        output_dir: str = "../logs",
        current_log_file: str = None,
        current_code_dir: str = None,
        topic: str = None,
    ):
        super().__init__(
            api_key=api_key,
            prompt_dir=prompt_dir,
            output_dir=output_dir,
            current_log_file=current_log_file,
            current_code_dir=current_code_dir,
            topic=topic or "dmrg",
        )

    def Generate_code_ReAct_by_plan(self, user_model: str, Plan_info: dict, iscaltoken: bool = False) -> dict:
        """Generate code from plan. Optionally uses query_summary if available. Saves as code_ReAct_fullparas."""
        role_prompt = self._load_prompt(topic="ReAct_dft", prompt_file="GenerateCode_ReAct_stand1")
        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        constraints = self._load_topic_language_packages()
        if constraints:
            parts.append(self._format_topic_constraints_input(constraints))
        if Plan_info:
            parts.append("PLAN_INFO:\n")
            parts.append(json.dumps(Plan_info, ensure_ascii=False))
            parts.append("\n\n")
        work_dir = Path(self.current_code_dir) if self.current_code_dir else None
        query_summary_data = None
        if work_dir:
            query_summary_file = work_dir / "query" / "query_summary.jsonl"
            if query_summary_file.exists():
                try:
                    raw = query_summary_file.read_text(encoding="utf-8").strip()
                    if raw:
                        query_summary_data = json.loads(raw, strict=False)
                except Exception:
                    pass
        if query_summary_data:
            parts.append("QUERY_SUMMARY:\n")
            parts.append(json.dumps(query_summary_data, ensure_ascii=False))
            parts.append("\n\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)
        if iscaltoken:
            response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True, require_code_block=True, code_tags=["CODE"])
        else:
            response = self._send_chat_robust(user_model, user_prompt, max_iter=3, require_code_block=True, code_tags=["CODE"])
            token_dict = None
        save_output = self._save_code_file(response, anci="code_ReAct_fullparas", isarbi=True, content_tag="CODE")
        subplot_name = save_output.get("subplot_name") if save_output else None
        code_file = save_output.get("code_file") if save_output else None
        code_content = save_output.get("code_content") if save_output else None
        output = {
            "Content": "Generated code (ReAct)",
            "LLM": user_model,
            "subplot_name": subplot_name,
            "code_file": code_file,
            "code_content": code_content,
            "token_stats": token_dict,
        }
        self._log_event(output)
        return output

    def Execute_code_ReAct(self, code_file: str, timeout: int = 300) -> dict:
        """Execute code file via run_program. Returns {exitcode, stdout, stderr}; for .inp also timeout, running_successfully.
        For .inp (ORCA) uses timeout=60 (aligns with mcp_orca_timeout in DFT verification)."""
        path = Path(code_file)
        project_root = str(path.parent)
        ext = path.suffix.lower()
        if ext == ".py":
            result = run_program.run_python_file(str(path), project_root, timeout)
            return {
                "exitcode": result.get("exitcode"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
            }
        elif ext == ".jl":
            result = run_program.run_julia_file(str(path), project_root, timeout)
            return {
                "exitcode": result.get("exitcode"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
            }
        elif ext == ".inp":
            orca_timeout = 60  # Aligns with mcp_orca_timeout in DFT verification
            result = run_program.run_orca_file(str(path), project_root, orca_timeout)
            return {
                "exitcode": result.get("exitcode"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "timeout": result.get("timeout", False),
                "running_successfully": result.get("running_successfully"),
            }
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    def Generate_repair_suggest_by_execution_result(
        self,
        user_model: str,
        execution_result: dict,
        code_content: str,
        tag_name_for_save: str,
        iscaltoken: bool = False,
    ) -> dict:
        """Generate repair suggestions from execution result. Output format same as coderepair."""
        role_prompt = self._load_prompt(topic="ReAct_dft", prompt_file="RepairCode_suggest_by_execution_result_stand1")
        parts = [role_prompt, "\n\nEXECUTION_RESULT:\n"]
        parts.append(json.dumps(execution_result, ensure_ascii=False, indent=2))
        parts.append("\n\nCODE:\n")
        parts.append(code_content)
        parts.append("\n\n-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)
        if iscaltoken:
            response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
        else:
            response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
            token_dict = None
        parsed = self._parse_json(response)
        repair_suggestions = parsed.get("repair_suggestions", [])
        for i, s in enumerate(repair_suggestions):
            s["repair_id"] = f"rep{i+1}"
        output_dict = {"repair_suggestions": repair_suggestions}
        if iscaltoken and token_dict:
            output_dict["token_stats"] = token_dict
        work_dir = Path(self.current_code_dir or self.output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        out_file = work_dir / f"repair_suggest_{tag_name_for_save}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(output_dict, ensure_ascii=False, indent=2) + "\n\n")
        self._log_event({
            "Content": "Generate repair suggestions by execution result",
            "LLM": user_model,
            "tag_name_for_save": tag_name_for_save,
            "output_file": str(out_file),
            "num_suggestions": len(repair_suggestions),
        })
        return output_dict

    def Repair_code_ReAct_by_suggestions(
        self,
        user_model: str,
        tag_name: str,
        code_original_name: str,
        code_repair_name: str,
        iscaltoken: bool = False,
        tag_query_dir: str | None = None,
    ) -> dict:
        """Repair code by suggestions. Optionally uses QUERY_SUMMARY when tag_query_dir is set."""
        work_dir = Path(self.current_code_dir or self.output_dir)
        suggest_path = work_dir / f"repair_suggest_{tag_name}.jsonl"
        if not suggest_path.exists():
            raise FileNotFoundError(f"Repair suggestion file not found: {suggest_path}")
        suggest_data = json.loads(suggest_path.read_text(encoding="utf-8").strip(), strict=False)
        repair_suggestions = suggest_data.get("repair_suggestions", [])
        if not repair_suggestions:
            raise ValueError(f"No repair suggestions in {suggest_path}")
        code_original_path = work_dir / f"{code_original_name}.py"
        if not code_original_path.exists():
            code_original_path = work_dir / f"{code_original_name}.jl"
        if not code_original_path.exists():
            code_original_path = work_dir / f"{code_original_name}.inp"
        if not code_original_path.exists():
            raise FileNotFoundError(f"Original code '{code_original_name}' not found in {work_dir}")
        code_ext = code_original_path.suffix
        current_code_content = code_original_path.read_text(encoding="utf-8")
        role_prompt = self._load_prompt(topic="ReAct_dft", prompt_file="RepairCode_ReAct_by_suggestions_stand1")
        total_input = total_output = 0
        for repair_suggestion in repair_suggestions:
            repair_info = {
                "location": repair_suggestion.get("location", ""),
                "location_description": repair_suggestion.get("location_description", ""),
                "repair_suggestion": repair_suggestion.get("repair_suggestion", ""),
                "repair_id": repair_suggestion.get("repair_id", ""),
            }
            parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            constraints = self._load_topic_language_packages()
            if constraints:
                parts.append(self._format_topic_constraints_input(constraints))
            parts.append("ORIGINAL_CODE:\n")
            parts.append(current_code_content)
            query_summary_data = None
            if tag_query_dir:
                qpath = work_dir / f"query_{tag_query_dir}" / f"query_summary_{repair_info.get('repair_id', '')}.jsonl"
                if qpath.exists():
                    try:
                        raw = qpath.read_text(encoding="utf-8").strip()
                        if raw:
                            query_summary_data = json.loads(raw, strict=False)
                    except Exception:
                        pass
            if query_summary_data:
                parts.append("\n\nQUERY_SUMMARY:\n")
                parts.append(json.dumps(query_summary_data, ensure_ascii=False, indent=2))
                parts.append("\n\n")
            parts.append("REPAIR_SUGGESTION:\n")
            parts.append(json.dumps(repair_info, ensure_ascii=False, indent=2))
            parts.append("\n\n-- END AGGREGATED INPUT --\n")
            user_prompt = "".join(parts)
            if iscaltoken:
                response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True, require_code_block=True, code_tags=["CODE"])
                if token_dict:
                    total_input += token_dict.get("input_tokens", 0)
                    total_output += token_dict.get("output_tokens", 0)
            else:
                response = self._send_chat_robust(user_model, user_prompt, max_iter=3, require_code_block=True, code_tags=["CODE"])
            save_output = self._save_code_file(response, anci=code_repair_name, code_dir=str(work_dir), isarbi=True, file_ext=code_ext.lstrip("."), content_tag="CODE")
            if save_output:
                current_code_content = save_output.get("code_content", current_code_content)
        final_path = work_dir / f"{code_repair_name}{code_ext}"
        output = {
            "Content": "Repair code by suggestions (ReAct)",
            "LLM": user_model,
            "tag_name": tag_name,
            "code_file": str(final_path) if final_path.exists() else None,
        }
        if iscaltoken and (total_input > 0 or total_output > 0):
            output["token_stats"] = {"input_tokens": total_input, "output_tokens": total_output, "total": total_input + total_output}
        self._log_event(output)
        return output
