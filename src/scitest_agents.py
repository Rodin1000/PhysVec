# scitest_agents: agents for convergence testing (convtest), mirroring repair_agents structure.
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import OPENROUTER_API_KEY
from QMBagents import QMBagent
from utils import run_program


def _truncate(s: str, max_len: int = 500) -> str:
    """Truncate string for log; avoid huge entries."""
    if not s or not isinstance(s, str):
        return ""
    return (s[:max_len] + "...") if len(s) > max_len else s


class scitest_agents(QMBagent):
    """Base class for convergence-test agents."""

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
        if self.api_key is None:
            raise ValueError("API key must be provided for scitest_agents.")
        if self.topic is None:
            raise ValueError("Topic must be specified for scitest_agents.")


class Convtest_runner(scitest_agents):
    """Runner for convtest: Step 1 Modify_code_by_convergence_cases; Step 2 methods added later."""

    def Modify_code_by_convergence_cases(
        self,
        user_model: str,
        code_file: str,
        convergence_case: dict,
        output_code_stem: str,
        iscaltoken: bool = False,
        case_index: int = None,
    ) -> dict:
        """
        Modify code according to a single convergence case. Only the case's 'content' is passed to the LLM.
        Returns dict with 'code_file' (saved path) and optionally 'token_stats'.
        """
        if self.current_log_file:
            self._log_event({
                "Content": "Modify_code_by_convergence_cases start",
                "case_index": case_index,
                "case_content": _truncate(convergence_case.get("content", "")),
                "case_parameter": convergence_case.get("parameter", ""),
                "case_criteria": _truncate(convergence_case.get("criteria", "")),
                "code_file": code_file,
                "output_code_stem": output_code_stem,
            })
        code_path = Path(code_file)
        if not code_path.exists():
            if self.current_log_file:
                self._log_event({
                    "Content": "Modify_code_by_convergence_cases failed",
                    "case_index": case_index,
                    "reason": "code_file not found",
                    "code_file": code_file,
                })
            return {"code_file": None, "token_stats": None}

        code_content = self._load_file(str(code_path))
        ext = code_path.suffix.lower()
        if ext not in (".py", ".jl"):
            if self.current_log_file:
                self._log_event({
                    "Content": "Modify_code_by_convergence_cases failed",
                    "case_index": case_index,
                    "reason": f"unsupported extension {ext}",
                })
            return {"code_file": None, "token_stats": None}

        # Pass only this case's content to the LLM (plan §9.3)
        case_for_prompt = {"content": convergence_case.get("content", "")}
        role_prompt = self._load_prompt(topic="general", prompt_file="ModifyCodeByConvergenceCases_stand1")
        user_prompt = (
            role_prompt
            + "\n\n-- BEGIN AGGREGATED INPUT --\nCODE_CONTENT:\n"
            + code_content
            + "\n\nCONVERGENCE_CASE:\n"
            + json.dumps(case_for_prompt, ensure_ascii=False)
            + "\n-- END AGGREGATED INPUT --\n"
        )

        try:
            if iscaltoken:
                response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True, require_code_block=True, code_tags=["CODE"])
            else:
                response = self._send_chat_robust(user_model, user_prompt, max_iter=3, require_code_block=True, code_tags=["CODE"])
                token_dict = None
        except Exception as e:
            if self.current_log_file:
                self._log_event({
                    "Content": "Modify_code_by_convergence_cases failed",
                    "case_index": case_index,
                    "reason": "LLM failed",
                    "error": str(e),
                })
            return {"code_file": None, "token_stats": None}

        code_dir = str(code_path.parent)
        file_ext = ext.lstrip(".")
        save_result = self._save_code_file(
            response,
            anci=output_code_stem,
            code_dir=code_dir,
            isarbi=True,
            content_tag="CODE",
            file_ext=file_ext,
        )
        if save_result is None:
            if self.current_log_file:
                self._log_event({
                    "Content": "Modify_code_by_convergence_cases failed",
                    "case_index": case_index,
                    "reason": "save failed",
                    "output_code_stem": output_code_stem,
                    "token_stats": token_dict if iscaltoken else None,
                })
            return {"code_file": None, "token_stats": token_dict if iscaltoken else None}

        out_path = save_result.get("code_file")
        if self.current_log_file and out_path:
            self._log_event({
                "Content": "Modify_code_by_convergence_cases done",
                "case_index": case_index,
                "code_file": out_path,
                "token_stats": token_dict if iscaltoken else None,
            })

        return {"code_file": out_path, "token_stats": token_dict if iscaltoken else None}

    def Modify_code_plotsave(
        self,
        code_file: str,
        user_model: str,
        output_code_stem: str,
        iscaltoken: bool = False,
    ) -> dict:
        """
        Read the program file and ask the model to check and add (if missing) plotting and figure-saving instructions.
        Do NOT modify any program implementation, parameter settings, function definitions, etc. outside of plotting.
        Save to the program's directory using savefig etc.; do not add grid lines.
        Returns dict with 'code_file' (saved path) and optionally 'token_stats'.
        """
        if self.current_log_file:
            self._log_event({
                "Content": "Modify_code_plotsave start",
                "code_file": code_file,
                "output_code_stem": output_code_stem,
            })
        code_path = Path(code_file)
        if not code_path.exists():
            if self.current_log_file:
                self._log_event({
                    "Content": "Modify_code_plotsave failed",
                    "reason": "code_file not found",
                    "code_file": code_file,
                })
            return {"code_file": None, "token_stats": None}

        code_content = self._load_file(str(code_path))
        ext = code_path.suffix.lower()
        if ext not in (".py", ".jl"):
            if self.current_log_file:
                self._log_event({
                    "Content": "Modify_code_plotsave failed",
                    "reason": f"unsupported extension {ext}",
                })
            return {"code_file": None, "token_stats": None}

        code_ext = ext.lstrip(".")  # "py" or "jl"
        # Read plot_requests.json from task dir (parent of work_dir)
        task_dir = code_path.parent.parent
        plot_requests_path = task_dir / "plot_requests.json"
        plot_requests_content = None
        if plot_requests_path.exists():
            try:
                with open(plot_requests_path, encoding="utf-8") as f:
                    data = json.load(f)
                req_list = data.get("plot_requests")
                if req_list and isinstance(req_list, list):
                    lines = []
                    for item in req_list:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if isinstance(v, str):
                                    lines.append(f"- {k}: {v}")
                    if lines:
                        plot_requests_content = "\n".join(lines)
            except (json.JSONDecodeError, OSError):
                pass

        role_prompt = self._load_prompt(topic="general", prompt_file="ModifyCodePlotsave_stand1")
        user_prompt = (
            role_prompt
            + "\n\n-- BEGIN AGGREGATED INPUT --\nCODE_EXT:\n"
            + code_ext
            + "\n\nCODE_CONTENT:\n"
            + code_content
        )
        if plot_requests_content:
            user_prompt += "\n\nPLOT_REQUESTS:\n" + plot_requests_content + "\n"
        user_prompt += "\n-- END AGGREGATED INPUT --\n"

        try:
            if iscaltoken:
                response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True, require_code_block=True, code_tags=["CODE"])
            else:
                response = self._send_chat_robust(user_model, user_prompt, max_iter=3, require_code_block=True, code_tags=["CODE"])
                token_dict = None
        except Exception as e:
            if self.current_log_file:
                self._log_event({
                    "Content": "Modify_code_plotsave failed",
                    "reason": "LLM failed",
                    "error": str(e),
                })
            return {"code_file": None, "token_stats": None}

        code_dir = str(code_path.parent)
        file_ext = ext.lstrip(".")
        save_result = self._save_code_file(
            response,
            anci=output_code_stem,
            code_dir=code_dir,
            isarbi=True,
            content_tag="CODE",
            file_ext=file_ext,
        )
        if save_result is None:
            if self.current_log_file:
                self._log_event({
                    "Content": "Modify_code_plotsave failed",
                    "reason": "save failed",
                    "output_code_stem": output_code_stem,
                })
            return {"code_file": None, "token_stats": token_dict if iscaltoken else None}

        out_path = save_result.get("code_file")
        if self.current_log_file and out_path:
            self._log_event({
                "Content": "Modify_code_plotsave done",
                "code_file": out_path,
                "token_stats": token_dict if iscaltoken else None,
            })

        return {"code_file": out_path, "token_stats": token_dict if iscaltoken else None}

    def Run_convtest_code_and_refine(
        self,
        code_file: str,
        user_model: str,
        output_code_stem: str,
        convergence_case: dict = None,
        max_refine_iter: int = 3,
        timeout: int = 10000,
        iscaltoken: bool = False,
        case_index: int = None,
        conv_loop: int = None,
    ) -> dict:
        """
        Inner loop: each iteration (1) execute current code, (2) if exitcode==0 return success,
        (3) if exitcode!=0 and refine_count < max_refine_iter then refine (overwrite file) and repeat;
        otherwise return. Returns exitcode, stdout, stderr, code_file, refine_iterations, token_stats (plan §10.3).
        """
        if self.current_log_file:
            self._log_event({
                "Content": "Run_convtest_code_and_refine start",
                "case_index": case_index,
                "conv_loop": conv_loop,
                "code_file_input": code_file,
                "output_stem": output_code_stem,
            })
        path = Path(code_file)
        if not path.exists():
            if self.current_log_file:
                self._log_event({"Content": "Run_convtest_code_and_refine failed", "reason": "code_file not found", "code_file": code_file})
            return {"exitcode": None, "stdout": "", "stderr": "", "code_file": None, "refine_iterations": 0, "token_stats": None}

        code_dir = str(path.parent)
        ext = path.suffix.lower()
        if ext not in (".py", ".jl"):
            if self.current_log_file:
                self._log_event({"Content": "Run_convtest_code_and_refine failed", "reason": f"unsupported extension {ext}"})
            return {"exitcode": None, "stdout": "", "stderr": "", "code_file": None, "refine_iterations": 0, "token_stats": None}
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
            # (1) Execute current code
            result = _run(Path(current_code_file))
            exitcode = result.get("exitcode")
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")

            if exitcode == 0:
                break

            if refine_count >= max_refine_iter:
                break

            # Log execution failure before refine
            if self.current_log_file:
                self._log_event({
                    "Content": "Run_convtest_code_and_refine execution failed, refining",
                    "case_index": case_index,
                    "conv_loop": conv_loop,
                    "exitcode": exitcode,
                    "refine_count_before": refine_count,
                    "stdout": stdout,
                    "stderr": stderr,
                })

            # (2) Refine: LLM modifies code, overwrite same file
            try:
                current_code_content = self._load_file(current_code_file)
                role_prompt = self._load_prompt(topic="general", prompt_file="RefineConvtestCode_stand1")
                parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                if convergence_case:
                    parts.append("CONVERGENCE_CASE:\n")
                    parts.append(json.dumps(convergence_case.get("content", ""), ensure_ascii=False))
                    parts.append("\n\n")
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
                    anci=output_code_stem,
                    code_dir=code_dir,
                    isarbi=True,
                    content_tag="CODE",
                    file_ext=file_ext,
                )
                if save_result is None:
                    if self.current_log_file:
                        self._log_event({"Content": "Run_convtest_code_and_refine save failed", "refine_count": refine_count})
                    break
                current_code_file = save_result.get("code_file")
                refine_count += 1
                if self.current_log_file and current_code_file:
                    self._log_event({"Content": "Run_convtest_code_and_refine refined", "code_file": current_code_file, "refine_count": refine_count})
            except Exception as e:
                if self.current_log_file:
                    self._log_event({"Content": "Run_convtest_code_and_refine refine exception", "error": str(e), "refine_count": refine_count})
                break

        token_stats = None
        if iscaltoken and token_list:
            ti = sum(t.get("input_tokens", 0) for t in token_list)
            to = sum(t.get("output_tokens", 0) for t in token_list)
            if ti + to > 0:
                token_stats = {"input_tokens": ti, "output_tokens": to, "total": ti + to}

        result = {
            "exitcode": exitcode,
            "stdout": stdout,
            "stderr": stderr,
            "code_file": current_code_file,
            "refine_iterations": refine_count,
            "token_stats": token_stats,
        }
        if self.current_log_file and exitcode != 0:
            self._log_event({
                "Content": "Run_convtest_code_and_refine exitcode non-zero",
                "case_index": case_index,
                "conv_loop": conv_loop,
                "exitcode": exitcode,
                "refine_iterations": refine_count,
                "stdout": stdout,
                "stderr": stderr,
            })
        elif self.current_log_file and exitcode == 0:
            self._log_event({
                "Content": "Run_convtest_code_and_refine done",
                "case_index": case_index,
                "conv_loop": conv_loop,
                "exitcode": 0,
                "code_file": current_code_file,
                "refine_iterations": refine_count,
                "token_stats": token_stats,
                "stdout": stdout,
                "stderr": stderr,
            })
        return result

    def Summarize_convtest_run(
        self,
        code_file: str,
        stdout: str,
        user_model: str = None,
        convergence_case: dict = None,
        iscaltoken: bool = False,
        report_path: str = None,
        conv_loop: int = None,
        case_index: int = None,
    ) -> dict:
        """
        Summarize a convtest run (code + stdout) into computation_params and convergence_cases_summary.
        If report_path and conv_loop are given, appends one JSON line (conv_loop, computation_params, convergence_cases_summary) to report_path.
        Returns dict with at least computation_params, convergence_cases_summary; plus token_stats if iscaltoken. Plan §10.5.
        """
        path = Path(code_file)
        if not path.exists():
            if self.current_log_file:
                self._log_event({"Content": "Summarize_convtest_run failed", "reason": "code_file not found", "code_file": code_file})
            return {"computation_params": "", "convergence_cases_summary": "", "token_stats": None}

        code_content = self._load_file(str(path))
        role_prompt = self._load_prompt(topic="general", prompt_file="SummarizeConvtestRun_stand1")
        parts = [
            role_prompt,
            "\n\n-- BEGIN AGGREGATED INPUT --\nCODE_CONTENT:\n",
            json.dumps(code_content, ensure_ascii=False),
            "\n\nSTDOUT:\n",
            json.dumps(stdout, ensure_ascii=False),
        ]
        if convergence_case:
            parts.append("\n\nCONVERGENCE_CASE:\n")
            parts.append(json.dumps(convergence_case.get("content", ""), ensure_ascii=False))
        parts.append("\n-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)

        try:
            if iscaltoken:
                response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
            else:
                response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
                token_dict = None
        except Exception as e:
            if self.current_log_file:
                self._log_event({"Content": "Summarize_convtest_run LLM failed", "error": str(e)})
            return {"computation_params": "", "convergence_cases_summary": "", "token_stats": None}

        parsed = self._parse_json(response)
        if not isinstance(parsed, dict):
            return {"computation_params": "", "convergence_cases_summary": "", "token_stats": token_dict if iscaltoken else None}
        out = {
            "computation_params": parsed.get("computation_params", ""),
            "convergence_cases_summary": parsed.get("convergence_cases_summary", ""),
        }
        if iscaltoken:
            out["token_stats"] = token_dict
        if report_path is not None and conv_loop is not None:
            report_file = Path(report_path)
            report_file.parent.mkdir(parents=True, exist_ok=True)
            line = {"conv_loop": conv_loop, "computation_params": out["computation_params"], "convergence_cases_summary": out["convergence_cases_summary"]}
            with open(report_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        if self.current_log_file:
            self._log_event({
                "Content": "Summarize_convtest_run done",
                "case_index": case_index,
                "conv_loop": conv_loop,
                "computation_params": _truncate(out["computation_params"], 400),
                "convergence_cases_summary": _truncate(out["convergence_cases_summary"], 400),
                "report_path": report_path,
                "token_stats": out.get("token_stats"),
            })
        return out

    def Judge_convtest_convergence(
        self,
        report_path: str,
        content: str,
        criteria: str,
        user_model: str,
        iscaltoken: bool = False,
        case_index: int = None,
        conv_loop: int = None,
    ) -> dict:
        """
        Judge whether the convergence test has converged. When report has <2 lines (conv_loop==1),
        skip LLM and set isconv=False, reason="First round...". Otherwise reads last two lines,
        calls LLM. Always updates the last line with isconv/reason. Plan §10.6.
        """
        path = Path(report_path)
        if not path.exists():
            if self.current_log_file:
                self._log_event({
                    "Content": "Judge_convtest_convergence failed",
                    "case_index": case_index,
                    "conv_loop": conv_loop,
                    "reason": "report_path not found",
                    "report_path": report_path,
                })
            return {"isconv": False, "reason": "Report file not found", "token_stats": None}
        lines = [l for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
        if not lines:
            if self.current_log_file:
                self._log_event({
                    "Content": "Judge_convtest_convergence failed",
                    "case_index": case_index,
                    "conv_loop": conv_loop,
                    "reason": "Report has no lines",
                    "report_path": report_path,
                })
            return {"isconv": False, "reason": "Report has no lines", "token_stats": None}
        if len(lines) < 2:
            isconv = False
            reason = "First round: no previous round result available for comparison; convergence judgment is skipped."
            token_dict = None
            if self.current_log_file:
                self._log_event({
                    "Content": "Judge_convtest_convergence done",
                    "case_index": case_index,
                    "conv_loop": conv_loop,
                    "isconv": isconv,
                    "reason": _truncate(reason, 400),
                    "token_stats": None,
                })
            last = json.loads(lines[-1]) if lines[-1] else {}
            last["isconv"] = isconv
            last["reason"] = reason
            lines[-1] = json.dumps(last, ensure_ascii=False)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return {"isconv": isconv, "reason": reason, "token_stats": None}
        else:
            summary_fields = ["conv_loop", "convergence_cases_summary"]
            last_two = []
            for line in lines[-2:]:
                try:
                    d = json.loads(line)
                    last_two.append({k: d.get(k, "") for k in summary_fields})
                except json.JSONDecodeError:
                    last_two.append({})
            report_str = "\n".join(json.dumps(r, ensure_ascii=False) for r in last_two)
            role_prompt = self._load_prompt(topic="general", prompt_file="JudgeConvtestConvergence_stand1")
            user_prompt = (
                role_prompt
                + "\n\n-- BEGIN AGGREGATED INPUT --\nREPORT_LAST_TWO_LINES:\n"
                + report_str
                + "\n\nCONTENT:\n"
                + json.dumps(content, ensure_ascii=False)
                + "\n\nCRITERIA:\n"
                + json.dumps(criteria, ensure_ascii=False)
                + "\n-- END AGGREGATED INPUT --\n"
            )
            try:
                if iscaltoken:
                    response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
                else:
                    response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
                    token_dict = None
            except Exception as e:
                if self.current_log_file:
                    self._log_event({"Content": "Judge_convtest_convergence LLM failed", "error": str(e)})
                return {"isconv": False, "reason": f"Judge LLM failed: {e}", "token_stats": None}
            parsed = self._parse_json(response)
            if not isinstance(parsed, dict):
                return {"isconv": False, "reason": "Judge response parse failed", "token_stats": token_dict if iscaltoken else None}
            isconv = parsed.get("isconv", False)
            if isinstance(isconv, str):
                isconv = isconv.lower() in ("true", "1", "yes")
            isconv = bool(isconv)
            reason = str(parsed.get("reason", ""))
        last = json.loads(lines[-1]) if lines[-1] else {}
        last["isconv"] = isconv
        last["reason"] = reason
        lines[-1] = json.dumps(last, ensure_ascii=False)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = {"isconv": isconv, "reason": reason}
        if iscaltoken:
            out["token_stats"] = token_dict
        if self.current_log_file:
            self._log_event({
                "Content": "Judge_convtest_convergence done",
                "case_index": case_index,
                "conv_loop": conv_loop,
                "isconv": isconv,
                "reason": _truncate(reason, 400),
                "token_stats": out.get("token_stats"),
            })
        return out

    def Gen_conv_repair_suggestion(
        self,
        code_file: str,
        report_path: str,
        case_content: str,
        case_parameter: str,
        user_model: str,
        output_path: str,
        iscaltoken: bool = False,
    ) -> dict:
        """
        Generate repair suggestion for a non-converged case. Reads isconv and reason from
        report file's last JSON line. Plan §10.7.
        """
        path = Path(report_path)
        if not path.exists():
            if self.current_log_file:
                self._log_event({"Content": "Gen_conv_repair_suggestion failed", "reason": "report_path not found", "report_path": report_path})
            return {}
        lines = [l for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
        if not lines:
            return {}
        try:
            last = json.loads(lines[-1])
            isconv = last.get("isconv", False)
            reason = last.get("reason", "")
        except json.JSONDecodeError:
            if self.current_log_file:
                self._log_event({"Content": "Gen_conv_repair_suggestion failed", "reason": "report last line not valid JSON"})
            return {}

        code_path = Path(code_file)
        if not code_path.exists():
            if self.current_log_file:
                self._log_event({"Content": "Gen_conv_repair_suggestion failed", "reason": "code_file not found", "code_file": code_file})
            return {}
        code_content = self._load_file(str(code_path))

        role_prompt = self._load_prompt(topic="general", prompt_file="GenConvtestRepairSuggestion_stand1")
        user_prompt = (
            role_prompt
            + "\n\n-- BEGIN AGGREGATED INPUT --\nCODE_CONTENT:\n"
            + json.dumps(code_content, ensure_ascii=False)
            + "\n\nISCONV:\n"
            + json.dumps(isconv, ensure_ascii=False)
            + "\n\nREASON:\n"
            + json.dumps(reason, ensure_ascii=False)
            + "\n\nCONVERGENCE_CASE_CONTENT:\n"
            + json.dumps(case_content, ensure_ascii=False)
            + "\n\nCONVERGENCE_CASE_PARAMETER:\n"
            + json.dumps(case_parameter, ensure_ascii=False)
            + "\n-- END AGGREGATED INPUT --\n"
        )

        try:
            if iscaltoken:
                response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
            else:
                response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
                token_dict = None
        except Exception as e:
            if self.current_log_file:
                self._log_event({"Content": "Gen_conv_repair_suggestion LLM failed", "error": str(e)})
            return {}

        parsed = self._parse_json(response)
        if not isinstance(parsed, dict) or "repair_suggestions" not in parsed:
            if self.current_log_file:
                self._log_event({"Content": "Gen_conv_repair_suggestion failed", "reason": "LLM response missing repair_suggestions"})
            return {}
        repair_suggestions = parsed.get("repair_suggestions", [])
        if not isinstance(repair_suggestions, list):
            repair_suggestions = []
        out_obj = {"repair_suggestions": repair_suggestions}
        if iscaltoken and token_dict:
            out_obj["token_stats"] = token_dict

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_obj, ensure_ascii=False) + "\n", encoding="utf-8")
        if self.current_log_file:
            self._log_event({
                "Content": "Gen_conv_repair_suggestion done",
                "output_path": str(out_path),
                "num_repair_suggestions": len(repair_suggestions),
                "reason": _truncate(reason, 200),
                "token_stats": token_dict if iscaltoken else None,
            })
        return {"token_stats": token_dict} if iscaltoken else {}


    def Apply_conv_repair_to_code(
        self,
        code_file: str,
        suggestion_file: str,
        output_code_stem: str,
        user_model: str,
        iscaltoken: bool = False,
        case_index: int = None,
        conv_loop: int = None,
    ) -> dict:
        """
        Apply repair suggestion to code and save as next-round code. Plan §10.7.
        """
        code_path = Path(code_file)
        sugg_path = Path(suggestion_file)
        if not code_path.exists():
            if self.current_log_file:
                self._log_event({
                    "Content": "Apply_conv_repair_to_code failed",
                    "case_index": case_index,
                    "conv_loop": conv_loop,
                    "reason": "code_file not found",
                    "code_file": code_file,
                })
            return {"code_file": None}
        if not sugg_path.exists():
            if self.current_log_file:
                self._log_event({
                    "Content": "Apply_conv_repair_to_code failed",
                    "case_index": case_index,
                    "conv_loop": conv_loop,
                    "reason": "suggestion_file not found",
                    "suggestion_file": suggestion_file,
                })
            return {"code_file": None}

        code_content = self._load_file(str(code_path))
        sugg_text = self._load_file(str(sugg_path))
        suggestion_content = sugg_text
        try:
            sugg_lines = [l for l in sugg_text.strip().split("\n") if l.strip()]
            if sugg_lines:
                first = json.loads(sugg_lines[0])
                if isinstance(first, dict) and "repair_suggestions" in first:
                    parts = [s.get("repair_suggestion", "") for s in first.get("repair_suggestions", []) if isinstance(s, dict)]
                    suggestion_content = "\n\n".join(p for p in parts if p) if parts else sugg_text
        except (json.JSONDecodeError, TypeError):
            pass

        ext = code_path.suffix.lower()
        if ext not in (".py", ".jl"):
            if self.current_log_file:
                self._log_event({
                    "Content": "Apply_conv_repair_to_code failed",
                    "case_index": case_index,
                    "conv_loop": conv_loop,
                    "reason": f"unsupported extension {ext}",
                })
            return {"code_file": None}
        file_ext = ext.lstrip(".")
        code_dir = str(code_path.parent)

        role_prompt = self._load_prompt(topic="general", prompt_file="ApplyConvtestRepairToCode_stand1")
        user_prompt = (
            role_prompt
            + "\n\n-- BEGIN AGGREGATED INPUT --\nCODE_CONTENT:\n"
            + code_content
            + "\n\nSUGGESTION_CONTENT:\n"
            + suggestion_content
            + "\n-- END AGGREGATED INPUT --\n"
        )

        try:
            if iscaltoken:
                response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True, require_code_block=True, code_tags=["CODE"])
            else:
                response = self._send_chat_robust(user_model, user_prompt, max_iter=3, require_code_block=True, code_tags=["CODE"])
                token_dict = None
        except Exception as e:
            if self.current_log_file:
                self._log_event({
                    "Content": "Apply_conv_repair_to_code failed",
                    "case_index": case_index,
                    "conv_loop": conv_loop,
                    "reason": "LLM failed",
                    "error": str(e),
                })
            return {"code_file": None}

        save_result = self._save_code_file(
            response,
            anci=output_code_stem,
            code_dir=code_dir,
            isarbi=True,
            content_tag="CODE",
            file_ext=file_ext,
        )
        if save_result is None:
            if self.current_log_file:
                self._log_event({
                    "Content": "Apply_conv_repair_to_code failed",
                    "case_index": case_index,
                    "conv_loop": conv_loop,
                    "reason": "save failed",
                    "suggest_path": suggestion_file,
                    "code_file_input": code_file,
                })
            return {"code_file": None}
        out_path = save_result.get("code_file")
        if self.current_log_file and out_path:
            self._log_event({
                "Content": "Apply_conv_repair_to_code done",
                "case_index": case_index,
                "conv_loop": conv_loop,
                "code_file_next": out_path,
                "token_stats": token_dict if iscaltoken else None,
            })
        result = {"code_file": out_path}
        if iscaltoken:
            result["token_stats"] = token_dict
        return result

    def log_converged(self, case_index: int = None, conv_loop: int = None, final_path: str = None, source_code_file: str = None):
        """Log convergence event (call from orchestration when isconv=True)."""
        if self.current_log_file:
            self._log_event({
                "Content": "convtest converged",
                "case_index": case_index,
                "conv_loop": conv_loop,
                "final_path": final_path,
                "source_code_file": source_code_file,
                "exitcode": 0,
            })

    def log_max_loop_reached(self, case_index: int = None, max_convtest_loop: int = None, last_run_result: dict = None):
        """Log when max_convtest_loop reached without convergence (call from orchestration)."""
        if self.current_log_file:
            self._log_event({
                "Content": "convtest max_convtest_loop reached",
                "case_index": case_index,
                "isconv": False,
                "max_convtest_loop": max_convtest_loop,
                "last_exitcode": last_run_result.get("exitcode") if last_run_result else None,
                "last_refine_iterations": last_run_result.get("refine_iterations", 0) if last_run_result else 0,
                "last_code_file": last_run_result.get("code_file") if last_run_result else None,
            })
