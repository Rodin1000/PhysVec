# main.py
from openai import OpenAI
import time
import random
from pathlib import Path
import json
import base64
from typing import Optional, Dict, List, Tuple, Union, Any
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
from config import OPENROUTER_API_KEY, YIDONG_API_KEY
from config import get_model_settings

from utils import MCP_toolbox, code_editor_dft, run_program, send_chat_yidong


# Check the API key
if not OPENROUTER_API_KEY:
    raise ValueError("OpenRouter API key is not set. Please set the OPENROUTER_API_KEY environment variable.")


# File management
def create_run_log(pdf_name: str, subplotname: str = None, output_dir: str = "../logs", topic: str = "Unnamed", issubfile: bool = False, iscreatlog: bool = True, headname_dir: str = None, headname_log: str = None, idname_dir: str = None, idname_log: str = None) -> str:
        """
        Create a new JSONL log file for this run. Filename: <timestamp>_<pdf_name>.jsonl
        Returns the Path to the new log file.
        """

        safe_name = Path(pdf_name).name
        if safe_name.lower().endswith('.pdf'):
            safe_name = safe_name[:-4]
        safe_name = safe_name.replace(' ', '_')

        # Create output directory-----------------------------------------------------------
        if not issubfile:
            if topic is None:
                out_dir = Path(output_dir)
            else:
                out_dir = Path(output_dir) / f"{topic}"
            safe_subplotname = None
        elif subplotname is None:
            if topic is None:
                out_dir = Path(output_dir) / f"{headname_dir}_{safe_name}_{idname_dir}"
            else:
                out_dir = Path(output_dir) / f"{topic}" / f"{headname_dir}_{safe_name}_{idname_dir}"
            safe_subplotname = None
        else:
            safe_subplotname = subplotname.replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_')
            if topic is None:
                out_dir = Path(output_dir) / f"{headname_dir}_{safe_name}_{safe_subplotname}_{idname_dir}"
            else:
                out_dir = Path(output_dir) / f"{topic}" / f"{headname_dir}_{safe_name}_{safe_subplotname}_{idname_dir}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create log file if needed--------------------------------------------------------
        if iscreatlog:
            if safe_subplotname:
                filename = f"{headname_log}_{safe_name}_{safe_subplotname}_{idname_log}.jsonl"
            else:
                filename = f"{headname_log}_{safe_name}_{idname_log}.jsonl"
            log_path = out_dir / filename
            if not log_path.exists():
                try:
                    log_path.write_text("", encoding="utf-8")
                except Exception:
                    pass
            return str(log_path), str(out_dir)
        else:
            return str(out_dir)



# Agent define
class DFT_agent:
    """
    Agents for quantum many-body physics research. 
    """
    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        dataset_dir: str = "../Paper_dataset",
        prompt_dir: str = "../prompts",
        output_dir: str = "../Paper_docu/Content_papers",
        current_log_file: str = None,
        current_code_dir: str = "../Paper_docu/Codes",
        topic: str = None,
    ):
        self.api_key = api_key
        self.dataset_dir = dataset_dir
        self.prompt_dir = prompt_dir
        self.output_dir = output_dir
        self.current_log_file = current_log_file
        self.current_code_dir = current_code_dir
        self.topic = topic

        # check
        if self.api_key is None:
            raise ValueError("API key must be provided for DFT_agent.")
        if self.topic is None:
            raise ValueError("Topic must be specified for DFT_agent.")


    # IO unit------------------------------------------------
    def _load_prompt(self, topic: str, prompt_file: str) -> str:
        """
        Load a prompt from a text file.
        """
        # Use pathlib for robust path handling
        prompt_path = Path(self.prompt_dir) / topic / (prompt_file + ".txt")
        try:
            return prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file '{prompt_path}' not found.")
        
        
    def _load_file(self, file_path: str) -> str:
        """
        Load content from a text file.
        """
        path = Path(file_path)
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{path}' not found.")
        

    def _log_event(self, event: dict):
        """Append a single event as JSONL to the current run log (or fallback file)."""
        if self.current_log_file is None:
            log_file = Path(self.output_dir) / "agent_log.jsonl"
            log_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            log_file = Path(self.current_log_file)
       
        pretty = json.dumps(event, ensure_ascii=False, indent=2)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(pretty + "\n\n")

    def _load_topic_language_packages(self) -> Optional[dict]:
        """Load allowed language and packages for self.topic from prompts/{topic}/language_packages.txt. Returns None if file missing. (From dengken QMBagents.)"""
        path = Path(self.prompt_dir) / self.topic / "language_packages.txt"
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        out = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("allowed_language:"):
                out["allowed_language"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("allowed_packages:"):
                out["allowed_packages"] = line.split(":", 1)[1].strip()
        return out if len(out) == 2 else None

    def _format_topic_constraints_input(self, constraints: dict) -> str:
        """Format TOPIC / ALLOWED_LANGUAGE / ALLOWED_PACKAGES block for prompt injection. (From dengken QMBagents.)"""
        if not constraints:
            return ""
        return f"TOPIC:\n{self.topic}\n\nALLOWED_LANGUAGE:\n{constraints.get('allowed_language', '')}\n\nALLOWED_PACKAGES:\n{constraints.get('allowed_packages', '')}\n\n"

    def _parse_json(self, text: str):
        """Try to parse text as JSON; return parsed object or original text on failure."""
        # if not isinstance(text, str):
        #     return text

        # match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        # if match:
        #     json_str = match.group(1)
        # else:
        #     json_str = text # If no wrapper, use the text as is

        # try:
            # return json.loads(text)
        # except Exception:
        #     return text
        return self._extract_json_from_response(text)

    def _normalize_string_response(self, s: str) -> str:
        """
        Normalize a string that may be the string representation of a Python list of message parts.
        When the MCP/API returns [{"text": "..."}] but it gets stringified before reaching us,
        we receive e.g. "[{'text': '```json\\n{...}'}]" (literal backslash-n, not newlines).
        Parse with ast.literal_eval and extract the "text" fields to recover the actual content.
        """
        if not isinstance(s, str) or not s or len(s) < 10:
            return s
        s = s.strip()
        # Quick heuristic: looks like stringified list of dicts with text
        if not (s.startswith("[") and ("'text'" in s or '"text"' in s)):
            return s
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                parts = []
                for item in parsed:
                    if isinstance(item, dict) and "text" in item:
                        t = item["text"]
                        parts.append(str(t) if t is not None else "")
                    elif isinstance(item, str):
                        parts.append(item)
                if parts:
                    out = "\n".join(parts)
                    self._log_event({
                        "Content": "Normalized string response",
                        "reason": "response was stringified list of message parts; extracted text via ast.literal_eval",
                        "num_parts": len(parts),
                        "total_len": len(out),
                    })
                    return out
        except (SyntaxError, ValueError, TypeError):
            pass
        return s

    def _normalize_mcp_response_to_str(self, response) -> str:
        """
        Normalize MCP agent response to a string for JSON extraction / save.
        MCP agent.run() can return a list of message parts like [{"text": "...", "type": "text", "index": 0}].
        Extract and concatenate "text" fields so _extract_json_from_response / _save_code_file receive a string.
        If response is a string, try _normalize_string_response in case it's a stringified list of parts.
        """
        if isinstance(response, str):
            return self._normalize_string_response(response)
        if isinstance(response, list):
            parts = []
            for item in response:
                if isinstance(item, dict) and "text" in item:
                    t = item["text"]
                    if isinstance(t, str):
                        parts.append(t)
                    else:
                        parts.append(str(t))
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                out = "\n".join(parts)
                self._log_event({
                    "Content": "Normalized MCP response",
                    "reason": "response was list of message parts; extracted text for JSON parse/save",
                    "num_parts": len(parts),
                    "total_len": len(out),
                })
                return out
            return ""
        return str(response) if response is not None else ""

        
    
    def _parse_json_to_dict(self, txt: str) -> dict:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return {}
        block = m.group(0)
        try:
            return json.loads(block)
        except Exception:
            block = re.sub(r"(\w+):", r'"\1":', block)  # naive key quote
            block = block.replace("True", "true").replace("False", "false")
            try:
                return json.loads(block)
            except Exception:
                return {}

    def _try_parse_json(self, raw: str) -> Optional[dict]:
        """Try to parse a string as JSON; return dict if successful, None otherwise. No logging."""
        if not raw or not raw.strip():
            return None
        try:
            data = json.loads(raw.strip(), strict=False)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _extract_balanced_brace_json(self, text: str) -> Optional[str]:
        """Find the first { and return the substring up to the matching }. Returns None if not found or unbalanced."""
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        quote = None
        i = start
        while i < len(text):
            c = text[i]
            if escape:
                escape = False
                i += 1
                continue
            if c == "\\" and in_string:
                escape = True
                i += 1
                continue
            if not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
                elif c in ("'", '"'):
                    in_string = True
                    quote = c
            else:
                if c == quote:
                    in_string = False
            i += 1
        return None

    def _dedup_repetitive_text(self, text: str, min_phrase_len: int = 40, max_repeats: int = 3) -> str:
        """Detect and remove repetitive text loops in LLM output.
        Scans for phrases of `min_phrase_len` chars that repeat more than
        `max_repeats` times. Truncates at the end of the first occurrence
        of the repeated phrase, keeping one copy intact."""
        if not text or len(text) < min_phrase_len * (max_repeats + 1):
            return text
        tail = text[-min_phrase_len:]
        count = text.count(tail)
        if count <= max_repeats:
            return text
        first_end = text.find(tail) + len(tail)
        return text[:first_end]

    def _try_close_truncated_json(self, raw: str) -> Optional[dict]:
        """Attempt to recover a truncated JSON response by closing open
        strings, arrays, and objects. Works when the LLM produced valid
        JSON at the start but the response was cut off (no closing fence,
        missing braces)."""
        if not raw or '{' not in raw:
            return None
        text = raw.strip()
        if text.startswith('```'):
            nl = text.find('\n')
            if nl > 0:
                text = text[nl + 1:]
        end_fence = text.rfind('```')
        if end_fence > 0:
            text = text[:end_fence]
        text = text.strip()
        if not text.startswith('{'):
            brace = text.find('{')
            if brace < 0:
                return None
            text = text[brace:]

        # Track nesting structure to close in correct order
        stack = []  # tracks nesting: '{', '['
        in_string = False
        escaped = False
        for ch in text:
            if escaped:
                escaped = False
                continue
            if ch == '\\':
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                stack.append('{')
            elif ch == '[':
                stack.append('[')
            elif ch == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
            elif ch == ']':
                if stack and stack[-1] == '[':
                    stack.pop()

        if not in_string and not stack:
            try:
                data = json.loads(text, strict=False)
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None

        # Build closing sequence
        text = text.rstrip()
        if text.endswith(','):
            text = text[:-1]
        if in_string:
            text += '"'
        closing = ''.join('}' if ch == '{' else ']' for ch in reversed(stack))
        text += closing

        try:
            data = json.loads(text, strict=False)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    def _try_iterative_quote_fix(self, raw: str, max_rounds: int = 60) -> Optional[dict]:
        """Fix JSON with unescaped quotes by iteratively escaping the quote that
        causes json.loads to fail. Works for any JSON structure (nested objects, arrays).
        Each round: try parse → on failure, find the offending unescaped quote near
        the error position → escape it → retry."""
        if not raw or '{' not in raw:
            return None
        text = raw.strip()
        for _ in range(max_rounds):
            try:
                data = json.loads(text, strict=False)
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError as e:
                pos = e.pos
                if pos is None or pos <= 0:
                    return None
                fixed = False
                # Search backwards from error position for the nearest unescaped "
                # within a reasonable window (the problematic quote may be a few
                # chars back, e.g. in `"word", next` the " is before the comma)
                scan_start = max(0, pos - 20)
                for q in range(pos, scan_start - 1, -1):
                    if q < len(text) and text[q] == '"':
                        bs = 0
                        j = q - 1
                        while j >= 0 and text[j] == '\\':
                            bs += 1
                            j -= 1
                        if bs % 2 == 0:
                            text = text[:q] + '\\"' + text[q + 1:]
                            fixed = True
                            break
                if not fixed:
                    return None
            except Exception:
                return None
        return None

    def _try_fix_unescaped_quotes_json(self, raw: str) -> Optional[dict]:
        """Fix JSON with unescaped double quotes inside string values.
        Common LLM issue: {"key": "text with "quoted" word"}
        should be:         {"key": "text with \\"quoted\\" word"}
        Works for flat JSON objects with string values.
        """
        if not raw or '{' not in raw:
            return None
        text = raw.strip()
        key_pattern = re.compile(r'"([^"]{1,50})"\s*:')
        keys = list(key_pattern.finditer(text))
        if not keys:
            return None

        structural_quotes = set()
        for ki, km in enumerate(keys):
            structural_quotes.add(km.start())
            try:
                key_end_quote = text.index('"', km.start() + 1)
            except ValueError:
                continue
            structural_quotes.add(key_end_quote)

            try:
                colon_pos = text.index(':', key_end_quote)
            except ValueError:
                continue
            rest_after_colon = text[colon_pos + 1:].lstrip()
            if not rest_after_colon.startswith('"'):
                continue
            try:
                value_open = text.index('"', colon_pos + 1)
            except ValueError:
                continue
            structural_quotes.add(value_open)

            if ki + 1 < len(keys):
                search_end = keys[ki + 1].start()
            else:
                search_end = text.rindex('}')
            try:
                value_close = text.rindex('"', value_open + 1, search_end)
                structural_quotes.add(value_close)
            except ValueError:
                continue

        result = []
        for i, ch in enumerate(text):
            if ch == '"' and i not in structural_quotes:
                backslashes = 0
                j = i - 1
                while j >= 0 and text[j] == '\\':
                    backslashes += 1
                    j -= 1
                if backslashes % 2 == 0:
                    result.append('\\"')
                else:
                    result.append(ch)
            else:
                result.append(ch)

        fixed_text = ''.join(result)
        try:
            data = json.loads(fixed_text, strict=False)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _extract_json_from_response(self, response: str, error_reason: list | None = None) -> Optional[dict]:
        """
        Extract JSON dictionary from LLM response using robust extraction logic.
        Tries in order: (1) ```json ... ``` block, (2) raw JSON parse of full content,
        (3) first balanced { ... } object in content, (4) generic ``` ... ``` block.
        Accepts str or MCP-style list of message parts; normalized via _normalize_mcp_response_to_str.
        Returns the parsed dict or None.
        If error_reason is not None and parsing fails, appends the failure reason to it.
        """
        content = None
        try:
            response = self._normalize_mcp_response_to_str(response)
            if not response:
                return None
            content = response
            if not content.strip():
                return None

            # --- Pre-processing: remove repetitive text loops ---
            content = self._dedup_repetitive_text(content)

            # --- Strategy 1: ```json ... ``` block ---
            json_start_pattern = re.compile(r"```json\s*\r?\n", re.IGNORECASE)
            start_match = json_start_pattern.search(content)
            if start_match:
                start_pos = start_match.end()
                remaining = content[start_pos:]
                end_positions = list(re.finditer(r"```", remaining))
                if end_positions:
                    last_end_pos = end_positions[-1].start()
                    extracted = remaining[:last_end_pos].strip()
                    data = self._try_parse_json(extracted)
                    if data is not None:
                        return data
                    # Strategy 1b: fix unescaped quotes in the extracted block
                    data = self._try_iterative_quote_fix(extracted)
                    if data is not None:
                        return data
                data = self._try_parse_json(remaining)
                if data is not None:
                    return data

            # --- Strategy 2: Parse entire content as JSON (raw JSON, no fence) ---
            data = self._try_parse_json(content)
            if data is not None:
                return data

            # --- Strategy 3: Find first balanced { ... } and parse ---
            candidate = self._extract_balanced_brace_json(content)
            if candidate:
                data = self._try_parse_json(candidate)
                if data is not None:
                    return data

            # --- Strategy 4: Generic ``` ... ``` (no "json" tag) ---
            generic_fence = re.compile(r"```\s*\r?\n(.*?)\r?\n```", re.DOTALL)
            match = generic_fence.search(content)
            if match:
                data = self._try_parse_json(match.group(1))
                if data is not None:
                    return data

            # --- Strategy 5: Fix unescaped quotes inside JSON string values (flat) ---
            candidate = self._extract_balanced_brace_json(content)
            if candidate:
                data = self._try_fix_unescaped_quotes_json(candidate)
                if data is not None:
                    return data

            # --- Strategy 6: Iterative quote fix on raw content (handles nested JSON) ---
            data = self._try_iterative_quote_fix(content)
            if data is not None:
                return data

            # --- Strategy 7: Close truncated JSON (output token limit hit) ---
            data = self._try_close_truncated_json(content)
            if data is not None:
                return data

            reason = "Extract JSON failed - all strategies"
            self._log_event({
                "Content": reason,
                "content_preview": content[:300] if content else "None"
            })
            if error_reason is not None:
                error_reason.append(reason)
            return None
        except Exception as e:
            try:
                preview = content[:500] if content else "None"
            except NameError:
                preview = str(response)[:500] if response else "None"
            reason = f"Extract JSON failed - exception: {e}"
            self._log_event({
                "Content": "Extract JSON failed - exception",
                "error": str(e),
                "error_type": type(e).__name__,
                "content_preview": preview,
            })
            if error_reason is not None:
                error_reason.append(reason)
            return None

    def _extract_code_from_fence(self, response: str, content_tag: str = "CODE", error_reason: list | None = None) -> str | None:
        """Extract code between <{content_tag}> and </{content_tag}> (XML-style fence)."""
        if not response or not isinstance(response, str):
            return None
        start_tag = f"<{content_tag}>"
        end_tag = f"</{content_tag}>"
        start_idx = response.find(start_tag)
        if start_idx < 0:
            self._log_event({"Content": "Extract code fence failed", "reason": f"start tag not found: {start_tag}"})
            if error_reason is not None:
                error_reason.append(f"start tag not found: {start_tag}")
            return None
        start_idx += len(start_tag)
        end_idx = response.find(end_tag, start_idx)
        if end_idx < 0:
            # Check if they used start tag again instead of closing tag (e.g. <CODE> instead of </CODE>)
            second_start = response.find(start_tag, start_idx)
            if second_start >= 0:
                reason = f"You used {start_tag} again instead of the closing tag. The correct closing tag is {end_tag}. Use {end_tag} to close the code block, not {start_tag}."
            else:
                reason = f"end tag not found: {end_tag}"
            self._log_event({"Content": "Extract code fence failed", "reason": reason})
            if error_reason is not None:
                error_reason.append(reason)
            return None
        return response[start_idx:end_idx].strip()

    # --- _save_code_file (legacy: JSON-embedded code) - commented for rollback ---
    # def _save_code_file(self, response: str, anci: str = "", code_dir: str = None, isarbi: bool = False, content_tag: str = "code_content", file_ext: str = None):
    #     # Normalize if MCP returned list-of-parts (defensive; caller should normalize too)
    #     if not isinstance(response, str):
    #         self._log_event({
    #             "Content": "Save code failed - response format",
    #             "reason": f"expected str, got {type(response).__name__}",
    #             "diagnosis": "LLM likely generated valid output but MCP returned list of message parts. Normalize with _normalize_mcp_response_to_str before save.",
    #             "response_type": type(response).__name__,
    #             "response_preview": str(response)[:500] if response is not None else "None",
    #         })
    #         response = self._normalize_mcp_response_to_str(response)
    #         if not response:
    #             return None
    #     try:
    #         data = self._extract_json_from_response(response)
    #         if data is None:
    #             ...
    #     ... (full legacy implementation commented)

    def _save_code_file(self, response: str, anci: str = "", code_dir: str = None, isarbi: bool = False, content_tag: str = "CODE", file_ext: str = None):
        """Save code from <{content_tag}>...</{content_tag}> fence (XML-style) to file."""
        # Normalize if MCP returned list-of-parts (defensive; caller should normalize too)
        if not isinstance(response, str):
            self._log_event({
                "Content": "Save code failed - response format",
                "reason": f"expected str, got {type(response).__name__}",
                "diagnosis": "LLM likely generated valid output but MCP returned list of message parts. Normalize with _normalize_mcp_response_to_str before save.",
                "response_type": type(response).__name__,
                "response_preview": str(response)[:500] if response is not None else "None",
            })
            response = self._normalize_mcp_response_to_str(response)
            if not response:
                return None

        # Extract code from fence
        code_content = self._extract_code_from_fence(response, content_tag)
        if not code_content:
            self._log_event({"Content": "Save code failed", "reason": "no code content in fence"})
            return None

        # Determine file extension
        if file_ext is not None:
            ext = file_ext.lstrip('.')
        else:
            data = self._extract_json_from_response(response)
            if data is None:
                self._log_event({"Content": "Save code failed", "reason": "file_ext not provided and JSON parse failed"})
                return None
            code_type = data.get("code_type")
            if not code_type:
                self._log_event({"Content": "Save code failed", "reason": "missing code_type and file_ext not provided"})
                return None
            ct = str(code_type).strip().lower()
            if ct in ("python", "py"):
                ext = "py"
            elif ct in ("julia", "jl"):
                ext = "jl"
            elif ct in ("orca", "inp", "input"):
                # DFT / quantum chemistry workflows (e.g. ORCA) use text input files.
                ext = "inp"
            else:
                self._log_event({"Content": "Save code failed", "reason": f"unsupported code_type: {code_type}"})
                return None

        # Save code to file
        subplot_name = None
        if not isarbi:
            data = self._extract_json_from_response(response)
            if data:
                subplot_name = data.get("subplot_name")

        if code_dir is None:
            code_dir = Path(self.current_code_dir)
        else:
            code_dir = Path(code_dir)
        code_dir.mkdir(parents=True, exist_ok=True)

        if isarbi:
            filename = f"{anci}.{ext}"
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            safe = (str(subplot_name).replace(' ', '_') if subplot_name else "generated")
            filename = f"{safe}_{ts}_{anci}.{ext}"

        path = code_dir / filename
        # Fix corrupted output: literal backslash-n instead of real newlines
        if isinstance(code_content, str) and ("\\" + "n") in code_content and code_content.count("\n") < 3:
            try:
                code_content = code_content.encode("utf-8").decode("unicode_escape")
            except Exception:
                code_content = code_content.replace("\\" + "n", "\n")
        try:
            if path.exists():
                path.unlink()
            path.write_text(code_content, encoding='utf-8')
            return {"subplot_name": subplot_name, "code_file": str(path), "code_content": code_content}
        except Exception as e:
            self._log_event({"Content": "Save code failed", "error": str(e)})
            return None


    def _save_jsonl_file(self, response: str, anci: str = "", jsonl_dir: str = None, isarbi: bool = False, issub: bool = False, subname: str = ""):

        try:
            data = self._extract_json_from_response(response)
            if data is None:
                self._log_event({"Content": "Save jsonl failed", "reason": "response is not valid JSON", "response": response})
                return None
        except Exception as e:
            self._log_event({"Content": "Save jsonl failed", "reason": "response is not valid JSON", "error": str(e), "response": response})
            return None

        # Create formatted JSON content: save entire dictionary with readable formatting
        jsonl_content = json.dumps(data, ensure_ascii=False, indent=2)

        # Write file
        if jsonl_dir is None:
            jsonl_dir = Path(self.current_code_dir)
        else:
            jsonl_dir = Path(jsonl_dir)
        jsonl_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectory if requested
        if issub and subname:
            jsonl_dir = jsonl_dir / subname
            jsonl_dir.mkdir(parents=True, exist_ok=True)

        if isarbi == False:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"Unnamed_guideline_{ts}_{anci}.jsonl"
        else:
            filename = f"{anci}.jsonl"

        path = jsonl_dir / filename
        try:
            path.write_text(jsonl_content, encoding='utf-8')
            return {"jsonl_file": str(path), "jsonl_content": jsonl_content}
        except Exception as e:
            self._log_event({"Content": "Save jsonl failed", "error": str(e)})
            return None
        

    def _save_txt_file(self, response: str, anci: str = "", txt_dir: str = None, isarbi: bool = False) -> str:
        os.makedirs(txt_dir, exist_ok=True)
        if isarbi == False:
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"Unnamed_README_{ts}_{anci}.txt"
        else:
            fname = anci if anci.lower().endswith(".txt") else f"{anci}.txt"

        path = os.path.join(txt_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(response or "")
        return path


    # LLM calling unit--------------------------------------
    def _send_chat(self, user_model: str, user_prompt: str, temperature: float = None, top_p: float = None, api_type: str = "yidong", iscaltoken: bool = False):
        """
        Send chat messages to the LLM and get the response.
        Returns: (response_text, token_dict) where token_dict is None if iscaltoken=False
        """
        model_name = user_model
        settings = get_model_settings(model_name)
        if temperature is not None:
            temp_set = temperature
        else:
            temp_set = settings.get("temperature")

        if top_p is not None:
            top_p_set = top_p
        else:
            top_p_set = settings.get("top_p")

        QMB_author_prompt = self._load_prompt(topic="general_dft", prompt_file="QMB_author")
        message_to_send = [
            {"role": "system", "content": QMB_author_prompt},
            {"role": "user", "content": user_prompt}
        ]

        model_params = {
            "temperature": temp_set,
            "top_p": top_p_set,
            # add other parameters as needed
        }

        try:
            if api_type == "openrouter":
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=message_to_send,
                    **model_params
                )
                return response.choices[0].message.content.strip(), None
            elif api_type == "yidong":
                response_text, status_code, token_dict = send_chat_yidong.send_chat_diverse_model(
                    user_model=model_name,
                    role_prompt=QMB_author_prompt,
                    user_prompt=user_prompt,
                    temperature=temp_set,
                    top_p=top_p_set,
                    iscaltoken=iscaltoken,
                )
                if status_code != 200:
                    print(f"calling model {model_name} failed with status code: {status_code}, the response text is: {response_text}")
                return response_text, token_dict
        except Exception as e:
            return f"calling model {model_name} failed with error: {str(e)}", None

    
    def _send_chat_robust(self, user_model: str, user_prompt: str, max_iter: int = 3, temperature: float = None, top_p: float = None, iscaltoken: bool = False, require_code_block: bool = False, code_tags: list[str] | None = None):
        """
        Send chat messages to the LLM and get the response, with validation.
        Retries up to max_iter times if the response cannot be parsed as a JSON dictionary,
        or (when require_code_block=True) if any code block fence fails to parse.

        Args:
            user_model: Model name to use
            user_prompt: User prompt
            max_iter: Maximum number of retry attempts (default: 3)
            iscaltoken: Whether to calculate and return token statistics
            require_code_block: If True, also validate that all code_tags can be extracted
            code_tags: List of content_tags to validate (default ["CODE"])

        Returns:
            If iscaltoken=False: response string
            If iscaltoken=True: (response_string, token_dict)
        """
        if code_tags is None:
            code_tags = ["CODE"]
        max_iter = 5  # TODO: temporarily bypass max_iter param
        last_parse_error = None
        response_text, token_dict = None, None
        for attempt in range(1, max_iter + 1):
            prompt = user_prompt
            if last_parse_error:
                prompt = user_prompt + "\n\n[PARSING ERROR FEEDBACK]\nLast parse error: " + last_parse_error + "\nPlease fix your output format according to the error above."
            response_text, token_dict = self._send_chat(user_model, prompt, temperature=temperature, top_p=top_p, iscaltoken=iscaltoken)

            json_error_reason = []
            parsed_dict = self._extract_json_from_response(response_text, error_reason=json_error_reason)
            if parsed_dict is None:
                last_parse_error = json_error_reason[-1] if json_error_reason else "Failed to extract JSON dictionary from response"
                if attempt < max_iter:
                    self._log_event({"Content": "Send chat robust retry", "LLM": user_model, "attempt": attempt, "max_iter": max_iter, "reason": last_parse_error, "response_snippet": response_text or "None"})
                else:
                    self._log_event({"Content": "Send chat robust failed after max_iter", "LLM": user_model, "attempt": attempt, "max_iter": max_iter, "reason": last_parse_error, "response_snippet": response_text or "None"})
                continue

            if require_code_block:
                code_error_reason = []
                failed_tags = [tag for tag in code_tags if self._extract_code_from_fence(response_text, tag, error_reason=code_error_reason) is None]
                if failed_tags:
                    last_parse_error = code_error_reason[-1] if code_error_reason else f"Failed to extract code block(s) for tag(s): {failed_tags}"
                    if attempt < max_iter:
                        self._log_event({"Content": "Send chat robust retry", "LLM": user_model, "attempt": attempt, "max_iter": max_iter, "reason": last_parse_error, "response_snippet": response_text or "None"})
                    else:
                        self._log_event({"Content": "Send chat robust failed after max_iter", "LLM": user_model, "attempt": attempt, "max_iter": max_iter, "reason": last_parse_error, "response_snippet": response_text or "None"})
                    continue

            if attempt > 1:
                self._log_event({"Content": "Send chat robust succeeded after retry", "LLM": user_model, "attempt": attempt, "max_iter": max_iter})
            if iscaltoken:
                return response_text, token_dict
            return response_text

        if iscaltoken:
            return response_text, token_dict
        return response_text
        

    async def _send_chat_through_mcp_use(self, user_model: str, user_prompt: str) -> str:
        response = await MCP_toolbox.send_chat_through_mcp_use(user_model=user_model, user_prompt=user_prompt)
        return response
        
    async def _send_chat_through_mcp_dynamic(
        self, 
        user_model: str, 
        user_prompt: str, 
        fs_target_dir: Optional[str] = None, 
        reference_dir_dict: Optional[Dict[str, str]] = None,
        retrieve_rt_target_dir: Optional[str] = None,
        init_servers: Optional[List[str]] = None,
        program_path: Optional[str] = None,
        program_timeout: int = 30,
        program_dft_input_path: Optional[str] = None,
        program_dft_timeout: int = 600,
        max_steps: int = 100,
        api_backend: Optional[str] = None,
        iscaltool: bool = False,
    ):
        """
        Send chat through MCP with dynamically configured servers.
        Keeps original flow with program_dft_* (ORCA/DFT) and adds iscaltool (tool statistics).
        Returns: response (str) when iscaltool=False; (response, tool_stats) when iscaltool=True.
        """
        result = await MCP_toolbox.send_chat_through_mcp_dynamic(
            user_model=user_model, 
            user_prompt=user_prompt, 
            fs_target_dir=fs_target_dir, 
            references=reference_dir_dict,
            retrieve_rt_target_dir=retrieve_rt_target_dir,
            init_servers=init_servers,
            program_path=program_path,
            program_timeout=program_timeout,
            program_dft_input_path=program_dft_input_path,
            program_dft_timeout=program_dft_timeout,
            max_steps=max_steps,
            iscaltool=iscaltool,
        )
        return result

class PaperSummerizer(DFT_agent):

    def _extract_pdf(self, pdf_path: str = "../Paper_dataset", pdf_name: str = "None", max_pages: Optional[int] = None, IsImage: int = 0) -> dict:

        papers_dir = Path(pdf_path)
        pdf_path = papers_dir / (pdf_name + ".pdf")
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file '{pdf_path}' not found.")
        doc = fitz.open(str(pdf_path))
        num_pages = doc.page_count
        pages_out = []
        pages_to_process = range(num_pages) if max_pages is None else range(min(num_pages, max_pages))

        for i in pages_to_process:
            page = doc.load_page(i)
            text = page.get_text("text")
            # Extract embedded images
            images = []
            if IsImage:
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_b64 = base64.b64encode(image_bytes).decode("ascii")
                    images.append({"name": f"page{ i+1 }_img{ img_index }", "b64": img_b64})

                # Render page to PNG (for diagrams/figures) and encode
                pix = page.get_pixmap(alpha=False)
                png_bytes = pix.tobytes("png")
                render_b64 = base64.b64encode(png_bytes).decode("ascii")
            pages_out.append({
                "page_number": i + 1,
                "text": text,
                "images": images,
                #"render_png_b64": render_b64,
            })
        metadata = {
            "pdf_path": str(pdf_path),
            "page_count": num_pages,
        }

        doc.close()
        return {"path": str(pdf_path), "pages": pages_out, "metadata": metadata}

    def _extract_tex(self, tex_path: str = None, tex_name: str = "None") -> dict:
        
        # check
        if tex_path is None:
            raise ValueError("tex_path must be provided for LaTeX extraction.")

        papers_dir = Path(self.dataset_dir) / self.topic / tex_path
        # try specific name first
        candidate = papers_dir / (tex_name + ".tex")
        chosen = None
        if candidate.exists() and candidate.is_file():
            chosen = candidate
        else:
            raise FileNotFoundError(f"Tex file '{candidate}' not found.")

        # try a few encodings
        encodings = ("utf-8", "utf-8-sig", "latin-1")
        content = None
        for enc in encodings:
            try:
                content = chosen.read_text(encoding=enc)
                break
            except Exception:
                continue
        if content is None:
            raise IOError(f"Unable to read tex file '{chosen}' with supported encodings.")

        return {"path": str(chosen), "content": content, "type": "tex"}

    # public entry--------------------------------------
    def Extract_pdf_text(self, pdf_path: str,pdf_name: str, max_pages: Optional[int] = None) -> dict:
        """
        Extract text and images from a PDF file.
        """
        PDF_info = self._extract_pdf(pdf_path=pdf_path, pdf_name=pdf_name, max_pages=max_pages, IsImage=0)
        return PDF_info
    
    def Extract_tex_text(self, tex_path: str, tex_name: str) -> dict:
        """
        Extract text from a LaTeX source file.
        """
        TEX_info = self._extract_tex(tex_path=tex_path, tex_name=tex_name)
        return TEX_info


    def Extract_paper_basic(self, user_model: str, PDF_info: dict) -> dict:
        
        role_prompt = self._load_prompt("ExtractPaperBasic")
        pdf_json = json.dumps(PDF_info, ensure_ascii=False)
        user_prompt = role_prompt + "\n\n" + "PDF_INFO_JSON:\n" + pdf_json
        # response = self._send_chat(user_model, user_prompt)
        response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
        parsed_response = self._parse_json(response)

        output = {"Content": "Basic info", "LLM": user_model, "PDF_name": PDF_info['path'], "response": parsed_response}
        self._log_event(output)
        return output
    
    
    def Extract_figure_captions(self, user_model: str, PDF_info: dict) -> dict:
        
        role_prompt = self._load_prompt("ExtractFigureCaptions")
        pdf_json = json.dumps(PDF_info, ensure_ascii=False)
        user_prompt = role_prompt + "\n\n" + "PDF_INFO_JSON:\n" + pdf_json
        # response = self._send_chat(user_model, user_prompt)
        response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
        parsed_response = self._parse_json(response)

        output = {"Content": "Figure info from captions", "LLM": user_model, "PDF_name": PDF_info['path'], "response": parsed_response}
        self._log_event(output)
        return output


    def Plan_task_by_text(self, user_model: str, subplot_name: str, PDF_info: dict = None, User_requests: str = "None", Paper_basic_info: dict = None, Figure_info: dict = None, iscaltoken: bool = False) -> dict:
        """Compose a prompt from any subset of the provided inputs and ask the LLM to plan tasks.

        The function only includes sections that are provided (non-None).
        When iscaltoken=True, output includes "token_stats" from _send_chat_robust.
        """
        # role_prompt = self._load_prompt("PlanTaskByText")
        # role_prompt = self._load_prompt(topic="general_dft", prompt_file="PlanTaskByText_disabled")
        role_prompt = self._load_prompt(topic="general_dft", prompt_file="PlanTaskByText_freereasoning")

        # Construct user prompt
        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        parts.append("Target subplot/table name:\n")
        parts.append(subplot_name)
        parts.append("\n\n")
        if PDF_info is not None:
            parts.append("PDF_INFO:\n")
            parts.append(json.dumps(PDF_info, ensure_ascii=False))
            parts.append("\n\n")
        parts.append("User_requests:\n")
        parts.append(User_requests)
        parts.append("\n\n")
        if Paper_basic_info is not None:
            parts.append("PAPER_BASIC_INFO:\n")
            parts.append(json.dumps(Paper_basic_info, ensure_ascii=False))
            parts.append("\n\n")
        if Figure_info is not None:
            parts.append("FIGURE_CAPTIONS_INFO:\n")
            parts.append(json.dumps(Figure_info, ensure_ascii=False))
            parts.append("\n\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)

        if iscaltoken:
            response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
        else:
            response = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=False)
            token_dict = None
        parsed_response = self._parse_json(response)

        output = {
            "Content": "Implementation plan by text",
            "LLM": user_model,
            "subplot_name": parsed_response["subplot_name"] if parsed_response else None,
            "User_requests": parsed_response["User_requests"] if parsed_response else None,
            "plan": parsed_response.get("plan", None) if parsed_response else None,
            "scientific_problem": parsed_response.get("scientific_problem") if parsed_response else None,
            "elements_used": parsed_response.get("elements_used") if parsed_response else None,
            "token_stats": token_dict,
        }
        self._log_event(output)
        return output
    

class CodeGenerator(DFT_agent):

    def Generate_code_by_plan(self, user_model: str, Plan_info: dict = None, iscaltoken: bool = False) -> dict:
        """
        Generate code by plan using query summary (non-MCP workflow version).
        Combines GenerateCodeByPlan_Query_instruct_stand1 + Plan_info + query_summary + GenerateCode_rules_stand1.
        When iscaltoken=True, output includes "token_stats" from _send_chat_robust.
        """
        # Load prompt
        role_prompt = self._load_prompt(topic="general_dft", prompt_file="GenerateCodeByPlan_Query_instruct_stand1")
        
        # Load query_summary from query_dir (simplified)
        query_summary_file = Path(self.current_code_dir) / "query" / "query_summary.jsonl"
        query_summary_data = None
        
        if query_summary_file.exists():
            try:
                query_summary_data = json.loads(self._load_file(str(query_summary_file)))
            except Exception as e:
                self._log_event({"Content": "Failed to load query_summary", "error": str(e), "file": str(query_summary_file)})
                print(f"Warning: Failed to load query_summary: {e}")
        else:
            print(f"Warning: query_summary.jsonl not found in {query_summary_file.parent}, proceeding without query summary.")
        
        # Load code rules
        code_rules = self._load_prompt(topic=self.topic, prompt_file="GenerateCode_rules_stand1")
        
        # Construct prompt parts
        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        
        if Plan_info is not None:
            parts.append("PLAN_INFO:\n")
            parts.append(json.dumps(Plan_info, ensure_ascii=False))
            parts.append("\n\n")
        
        if query_summary_data is not None:
            parts.append("QUERY_SUMMARY:\n")
            parts.append(json.dumps(query_summary_data, ensure_ascii=False))
            parts.append("\n\n")
        
        parts.append("--Code generation rules--\n")
        parts.append(code_rules)
        parts.append("\n\n-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)
        
        # Call LLM (non-MCP workflow)
        if iscaltoken:
            response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True, require_code_block=True, code_tags=["CODE"])
        else:
            response = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=False, require_code_block=True, code_tags=["CODE"])
            token_dict = None

        # Save code file
        save_code_output = self._save_code_file(response, anci="code_LLM", isarbi=True, content_tag="CODE")
        subplot_name = save_code_output.get("subplot_name") if save_code_output else None
        code_file = save_code_output.get("code_file") if save_code_output else None
        code_content = save_code_output.get("code_content") if save_code_output else None
        
        output = {
            "Content": "Generated code",
            "LLM": user_model,
            "subplot_name": subplot_name,
            "code_file": code_file,
            "code_content": code_content,
            "token_stats": token_dict
        }
        
        self._log_event(output)
        return output

        
    
    def Generate_code_by_pdftext(self, user_model: str, PDF_info: dict, subplot_name: str) -> dict:

        role_prompt = self._load_prompt("GenerateCodeByPlan")

        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        parts.append(f"target subplot: {subplot_name}\n\n")
        if PDF_info is not None:
            parts.append("PDF_INFO:\n")
            parts.append(json.dumps(PDF_info, ensure_ascii=False))
            parts.append("\n\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)
        response, _ = self._send_chat(user_model, user_prompt)
        
        save_code_output = self._save_code_file(response, content_tag="CODE")
        subplot_name = save_code_output.get("subplot_name") if save_code_output else None
        code_file = save_code_output.get("code_file") if save_code_output else None
        output = {"Content": "Generate code", "LLM": user_model, "subplot_name": subplot_name, "code_file": code_file, "response": response}

        self._log_event(output)
        return output
    

    async def Generate_code_by_plan_mcp_use(self, user_model: str, Plan_info: dict = None, PDF_info: dict = None, isusequery: bool = False,max_steps:int=50) -> dict:

        # Select prompt and reference directory based on isusequery flag
        if isusequery:
            # Use query-based prompt and reference query_dir
            role_prompt = self._load_prompt(topic="general_dft", prompt_file="GenerateCodeByPlan_mcp_use_Query_instruct_stand1")
            code_rules = self._load_prompt(topic=self.topic, prompt_file="GenerateCode_rules_stand1")
            parts = [role_prompt, "\n\n --Code generation rules--\n", code_rules, "\n\n"]
            # Create query subdirectory in current_code_dir (same as Retrieve_knowledge_by_plan_mcp_use)
            query_dir = Path(self.current_code_dir) / "query"
            query_dir.mkdir(parents=True, exist_ok=True)
            reference_dir_dict = {"libs": str(query_dir)}
        else:
            # Use local search prompt and reference Human_code_library
            # role_prompt = self._load_prompt("GenerateCodeByPlan_mcp_use_WebSearch")
            # role_prompt = self._load_prompt("GenerateCodeByPlan_mcp_use_LocalSearch")
            role_prompt = self._load_prompt(topic=self.topic, prompt_file="GenerateCodeByPlan_mcp_use_LocalSearch_stand1")
            reference_dir_dict = {"libs": Path("../Human_code_library") / self.topic}
            parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]

        if Plan_info is not None:
            parts.append("PLAN_INFO:\n")
            parts.append(json.dumps(Plan_info, ensure_ascii=False))
            parts.append("\n\n")
        if PDF_info is not None:
            parts.append("PDF_INFO:\n")
            parts.append(json.dumps(PDF_info, ensure_ascii=False))
            parts.append("\n\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)
        # response = await self._send_chat_through_mcp_use(user_model, user_prompt)
        response = await self._send_chat_through_mcp_dynamic(
            user_model=user_model,
            user_prompt=user_prompt,
            fs_target_dir="../trash",
            reference_dir_dict=reference_dir_dict,
            init_servers=["filesystem-mcp"],
            max_steps=max_steps,
            
        )
        
        # MCP agent may return list of message parts; normalize to str for JSON extract/save
        response_str = self._normalize_mcp_response_to_str(response)
        save_code_output = self._save_code_file(response_str, anci="code_LLM", isarbi=True, content_tag="CODE")
        subplot_name = save_code_output.get("subplot_name") if save_code_output else None
        code_file = save_code_output.get("code_file") if save_code_output else None
        code_content = save_code_output.get("code_content") if save_code_output else None
        output = {"Content": "Generated code", "LLM": user_model, "subplot_name": subplot_name, "code_file": code_file, "code_content": code_content}

        self._log_event(output)
        return output


    async def Retrieve_knowledge_by_plan(self, user_model: str, Plan_info: dict) -> dict:

        # Generate all queries for RAG---------------------------------
        # load prompt
        role_prompt = self._load_prompt(topic="general_dft", prompt_file="RetrieveKnowledgeByPlan_stand1")
        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        if Plan_info is not None:
            parts.append("PLAN_INFO:\n")
            parts.append(json.dumps(Plan_info, ensure_ascii=False))
            parts.append("\n\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)
        # Create query subdirectory in current_code_dir if it doesn't exist
        query_dir = Path(self.current_code_dir) / "query"
        query_dir.mkdir(parents=True, exist_ok=True)
        # send chat with MCP
        response = await self._send_chat_through_mcp_dynamic(
            user_model=user_model,
            user_prompt=user_prompt,
            fs_target_dir=query_dir,
            init_servers=["filesystem-mcp"]
        )
        
        output = {"Content": "Queries for RAG", "LLM": user_model, "query_dir": str(query_dir), "response": response}
        self._log_event(output)
        return output


    async def Retrieve_knowledge_by_plan_mcp_use(self, user_model: str, Plan_info: dict, User_requests: str = None, pdf_name: str = None, iscaltoken: bool = False, iscaltool: bool = False, max_queries: int = 8) -> dict:
        """
        Retrieve knowledge by plan using MCP. When iscaltoken=True, output includes "token_stats" from summary LLM call.
        When iscaltool=True, MCP call returns tool_stats; output includes "tool_stats".
        pdf_name: Paper identifier (e.g. jacs.3c06046) for structure lookup; passed as PAPER_ID in prompt.
        max_queries: Maximum number of query JSON files to include in the summary step (default 10).
            Excess queries are truncated to stay within token limits. Set to None to use all queries.
        """
        # load prompt
        role_prompt = self._load_prompt(topic="general_dft", prompt_file="RetrieveKnowledgeByPlan_mcp_use_stand1")
        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        parts.append("TOPIC:\n")
        parts.append(self.topic)
        parts.append("\n\n")
        if pdf_name is not None:
            parts.append("PAPER_ID:\n")
            parts.append(pdf_name)
            parts.append("\n\n")
        if User_requests is not None:
            parts.append("USER_REQUESTS:\n")
            parts.append(User_requests)
            parts.append("\n\n")
        if Plan_info is not None:
            parts.append("PLAN_INFO:\n")
            parts.append(json.dumps(Plan_info, ensure_ascii=False))
            parts.append("\n\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)
        # Create query subdirectory in current_code_dir if it doesn't exist
        query_dir = Path(self.current_code_dir) / "query"
        query_dir.mkdir(parents=True, exist_ok=True)
        # Initialize output with query_dir to ensure it's always present
        output = {
            "Content": "Queries for RAG",
            "LLM": user_model,
            "query_dir": str(query_dir),
            "response": None,
            "token_stats": None,
            "tool_stats": None
        }
        
        # implement the retrieval process
        try:
            # send chat with MCP
            # Pass query_dir as retrieve_rt_target_dir so retrieval results are saved directly to files
            # Only use retrieve-mcp server for this task (no filesystem-mcp needed)
            if iscaltool:
                response, tool_stats = await self._send_chat_through_mcp_dynamic(
                    user_model=user_model,
                    user_prompt=user_prompt,
                    fs_target_dir=None,  # Not needed when only using retrieve-mcp
                    retrieve_rt_target_dir=str(query_dir),
                    init_servers=["retrieve-mcp"],
                    iscaltool=True
                )
                output["response"] = response
                output["tool_stats"] = tool_stats
            else:
                response = await self._send_chat_through_mcp_dynamic(
                    user_model=user_model,
                    user_prompt=user_prompt,
                    fs_target_dir=None,  # Not needed when only using retrieve-mcp
                    retrieve_rt_target_dir=str(query_dir),
                    init_servers=["retrieve-mcp"],
                    iscaltool=False
                )
                output["response"] = response
        except Exception as e:
            # If MCP call fails, still return output with query_dir for error handling
            output["response"] = f"Error during retrieval: {str(e)}"
            output["error"] = str(e)
            self._log_event({"Content": "Retrieve knowledge error", "error": str(e), "query_dir": str(query_dir)})

        # create the summary of all retrieved files
        try:
            # Step 1: Collect all JSON files from query_dir
            json_files = sorted(query_dir.glob("*.json"))
            if not json_files:
                print(f"No JSON files found in {query_dir}, skipping summary creation.")
            else:
                total_found = len(json_files)
                if max_queries is not None and total_found > max_queries:
                    json_files = json_files[:max_queries]
                    print(f"Found {total_found} JSON files in {query_dir}, using first {max_queries} for summary (max_queries limit).")
                else:
                    print(f"Found {len(json_files)} JSON files in {query_dir}, creating summary...")

                # Step 2: Load and combine all query JSON files
                all_queries = []
                for json_file in json_files:
                    try:
                        query_data = json.loads(self._load_file(str(json_file)))
                        all_queries.append(query_data)
                    except Exception as e:
                        self._log_event({"Content": "Failed to load query file", "file": str(json_file), "error": str(e)})
                        continue
                
                # Step 3: Create summary using LLM
                summary_prompt = self._load_prompt(topic="general_dft", prompt_file="RetrieveKnowledgeByPlan_summary_stand1")
                parts = [summary_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                parts.append("ALL_QUERY:\n")
                parts.append(json.dumps(all_queries, ensure_ascii=False, indent=2))
                parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)
                
                # summary_response = self._send_chat(user_model, user_prompt)
                if iscaltoken:
                    summary_response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
                    output["token_stats"] = token_dict
                else:
                    summary_response = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=False)
                self._save_jsonl_file(
                    response=summary_response,
                    anci="query_summary",
                    jsonl_dir=str(query_dir),
                    isarbi=True
                )
                print(f"Summary saved to {query_dir / 'query_summary.jsonl'}")
        except Exception as e:
            self._log_event({"Content": "Failed to create query summary", "error": str(e), "query_dir": str(query_dir)})
            print(f"Warning: Failed to create query summary: {e}")
        
        self._log_event(output)
        return output


    async def Refine_code_obey_rules_mcp_use(self, user_model: str, code_file: str = None, max_iter: int = 3) -> dict:
        # Step 1: Create rules directory and copy rules file
        rules_dir = Path(self.current_code_dir) / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy rules file from prompts/{topic}/GenerateCodeByPlan_mcp_use_Query_rules_stand1.txt
        source_rules_file = Path(self.prompt_dir) / self.topic / "GenerateCodeByPlan_mcp_use_Query_rules_stand1.txt"
        if not source_rules_file.exists():
            raise FileNotFoundError(f"Rules file not found: {source_rules_file}")
        
        dest_rules_file = rules_dir / "GenerateCodeByPlan_mcp_use_Query_rules_stand1.txt"
        shutil.copy2(source_rules_file, dest_rules_file)
        
        # Step 2: Detect code file if not provided
        if code_file is None:
            raise FileNotFoundError("code_file must be provided.")
        
        # Determine file extension
        if code_file.lower().endswith(".py"):
            file_ext = "py"
        elif code_file.lower().endswith(".jl"):
            file_ext = "jl"
        else:
            file_ext = code_file.split(".")[-1] if "." in code_file else "unknown"
        
        # Step 3: Load and format prompt
        role_prompt = self._load_prompt(topic="general_dft", prompt_file="RefineCodeByRules_mcp_use_stand1")
        formatted_prompt = role_prompt.format(
            code_file=code_file,
            file_ext=file_ext,
            max_iter=max_iter,
        )
        
        # Step 4: Call MCP agent
        print(f"refine code to obey rules (MCP) - max_iter={max_iter}...")
        response = await self._send_chat_through_mcp_dynamic(
            user_model=user_model,
            user_prompt=formatted_prompt,
            fs_target_dir=str(self.current_code_dir),
            init_servers=["filesystem-mcp"]
        )
        
        # Step 5: Return result
        output = {
            "Content": "Refine code obey rules (MCP)",
            "LLM": user_model,
            "code_file": code_file,
            "file_ext": file_ext,
            "max_iter": max_iter,
            "rules_dir": str(rules_dir),
            "response": response
        }
        self._log_event(output)
        return output


    def _refine_code_with_part(self, user_model: str, code_former: str, part_rules: str, part_name: str, iteration: int) -> Tuple[str, str]:
        """
        Refine code for a single rules part (reason -> act).
        
        Returns:
            tuple: (refined_code_content, summary)
        """
        # Step 1: Reasoning - analyze code and identify issues
        print(f"  [part {part_name}] reasoning...")
        reason_prompt = self._load_prompt(topic="general_dft", prompt_file="RefineCodeByRules_reason_stand1")
        parts = [reason_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        if code_former is not None:
            parts.append("CODE_FORMER:\n")
            parts.append(code_former)
            parts.append("\n\n")
        if part_rules is not None:
            parts.append("CODE_RULES:\n")
            parts.append(part_rules)
            parts.append("\n\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)
        # response = self._send_chat(user_model, user_prompt)
        response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
        parsed_violations = self._parse_json(response)
        violations_text = parsed_violations.get("violations", "")
        self._log_event({"Content": f"Refine code obey rules (reasoning) - part {part_name}", "LLM": user_model, "iteration": iteration, "response": parsed_violations})

        # Step 2: Action - modify code based on reasoning
        print(f"  [part {part_name}] action...")
        act_prompt = self._load_prompt(topic="general_dft", prompt_file="RefineCodeByRules_act_stand1")
        parts = [act_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        if code_former is not None:
            parts.append("CODE_FORMER:\n")
            parts.append(code_former)
            parts.append("\n\n")
        if violations_text:
            parts.append("VIOLATIONS:\n")
            parts.append(violations_text)
            parts.append("\n\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)
        # response = self._send_chat(user_model, user_prompt)
        response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
        parsed_response = self._parse_json(response)
        refined_code = parsed_response.get("code_content", code_former)
        summary = parsed_response.get("summary", "")
        self._log_event({"Content": f"Refine code obey rules (action) - part {part_name}", "LLM": user_model, "iteration": iteration, "response": parsed_response})
        
        return refined_code, summary

    async def Refine_code_obey_rules(self, user_model: str, judge_model: str, code_file: str = None, max_iter: int = 3) -> dict:

        # Refine loop structure
        flag_obey = False
        count_iter = 0

        # Step 1: Detect code file and create backup
        if code_file is None:
            code_path_py = Path(self.current_code_dir) / "code_LLM.py"
            code_path_jl = Path(self.current_code_dir) / "code_LLM.jl"
            if code_path_py.exists():
                code_file = "code_LLM.py"
                code_path = code_path_py
            elif code_path_jl.exists():
                code_file = "code_LLM.jl"
                code_path = code_path_jl
            else:
                raise FileNotFoundError("code_LLM.py or code_LLM.jl not found in the current code directory.")
        else:
            code_path = Path(code_file)
            if not code_path.exists():
                raise FileNotFoundError(f"Code file not found: {code_path}")

        # Create backup: code_LLM_origin.{ext}
        file_ext = code_path.suffix
        backup_file = code_path.parent / f"code_LLM_origin{file_ext}"
        shutil.copy2(code_path, backup_file)
        print(f"Backup created: {backup_file}")

        # Step 2: Load all code rules part files dynamically
        rules_parts = {}
        topic_dir = Path(self.prompt_dir) / self.topic
        part_files = sorted(topic_dir.glob("code_rules_part*.txt"))
        
        if not part_files:
            raise FileNotFoundError(f"No code_rules_part*.txt files found in {topic_dir}")
        
        for part_file in part_files:
            part_name = part_file.stem.replace("code_rules_", "")  # e.g., "part1", "part2"
            rules_parts[part_name] = self._load_file(str(part_file))
        
        print(f"Loaded {len(rules_parts)} rules part files: {sorted(rules_parts.keys())}")

        # Step 3: Load code content
        code_former = self._load_file(str(code_path))
        
        # Step 4: Initialize refine_summary JSONL file
        refine_summary_file = Path(self.current_code_dir) / "refine_summary.jsonl"
        if refine_summary_file.exists():
            refine_summary_file.unlink()  # Remove existing file to start fresh
        refine_summary_file.touch()  # Create empty file

        # begin iteration
        while flag_obey == False and count_iter < max_iter:
            count_iter += 1
            print(f"refine code to obey rules - iteration {count_iter}/{max_iter}...")

            # Refine code with each part sequentially
            all_summaries = []
            for part_name in sorted(rules_parts.keys()):
                part_rules = rules_parts[part_name]
                code_former, part_summary = self._refine_code_with_part(
                    user_model=user_model,
                    code_former=code_former,
                    part_rules=part_rules,
                    part_name=part_name,
                    iteration=count_iter
                )
                all_summaries.append(f"[{part_name}] {part_summary}")
                
                # Save refined code after each part using _save_code_file (fence format)
                code_type = "julia" if file_ext == ".jl" else "python"
                part_response = "```json\n" + json.dumps({"code_type": code_type}, ensure_ascii=False) + "\n```\n\n<CODE>\n" + code_former + "\n</CODE>"
                self._save_code_file(
                    response=part_response,
                    anci=f"code_LLM_{count_iter}_{part_name}",
                    code_dir=str(code_path.parent),
                    isarbi=True,
                    content_tag="CODE"
                )
            
            # Save final refined code using _save_code_file (fence format)
            code_type = "julia" if file_ext == ".jl" else "python"
            final_response = "```json\n" + json.dumps({"code_type": code_type}, ensure_ascii=False) + "\n```\n\n<CODE>\n" + code_former + "\n</CODE>"
            self._save_code_file(
                response=final_response,
                anci=f"code_LLM_{count_iter}",
                code_dir=str(code_path.parent),
                isarbi=True,
                content_tag="CODE"
            )
            
            # Log combined summary
            combined_summary = " | ".join(all_summaries)
            summary_entry = {
                "iteration": count_iter,
                "summary": combined_summary
            }
            with open(refine_summary_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")

            # Judge the code whether it obeys the rules (using all part rules concatenated)
            print(f"judge the code whether it obeys the rules - iteration {count_iter}/{max_iter}...")
            # Concatenate all part rules
            all_rules_combined = "\n\n".join([
                f"=== {part_name} ===\n{rules_parts[part_name]}"
                for part_name in sorted(rules_parts.keys())
            ])
            
            role_prompt = self._load_prompt(topic="general_dft", prompt_file="RefineCodeByRules_judge_stand1")
            parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            if code_former is not None:
                parts.append("CODE_FORMER:\n")
                parts.append(code_former)
                parts.append("\n\n")
            if all_rules_combined:
                parts.append("CODE_RULES:\n")
                parts.append(all_rules_combined)
                parts.append("\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            user_prompt = "".join(parts)
            # response = self._send_chat(judge_model, user_prompt)
            response = self._send_chat_robust(judge_model, user_prompt, max_iter=3)
            parsed_response = self._parse_json(response)
            flag_obey = parsed_response.get("flag_obey", False)
            self._log_event({"Content": "Judge the code whether it obeys the rules", "LLM": judge_model, "iteration": count_iter, "response": parsed_response})

        # Return result
        output = {
            "Content": "Refine code obey rules",
            "LLM": user_model,
            "judge_LLM": judge_model,
            "code_file": code_file,
            "backup_file": str(backup_file),
            "max_iter": max_iter,
            "iterations": count_iter,
            "flag_obey": flag_obey,
            "refine_summary_file": str(refine_summary_file)
        }
        self._log_event(output)
        return output


    async def FormatCheck_code_by_rules(self, user_model: str, judge_model: str, code_file: str = None, max_iter: int = 3, temp_incr: float = 0.15, code_backup_name: str = "code_LLM_origin", code_save_name: str = "code_LLM", tag_name: str = None, iscaltoken: bool = False, isjudgeinfo: bool = True):
        """
        Format check code by rules.
        If isjudgeinfo=True, the improve step receives JUDGE_PREVIOUS_INFO (flag_obey, judge_reason) from the previous iteration.
        Returns: (improved_code, flag_obey, count_iter) if iscaltoken=False
                 (improved_code, flag_obey, count_iter, user_model_token_stats, judge_model_token_stats) if iscaltoken=True
        """
        # Step 1: Detect code file and create backup
        if code_file is None:
            raise ValueError("code_file cannot be None. Code generation may have failed. Check logs for details.")
        
        code_path = Path(code_file)
        if not code_path.exists():
            raise FileNotFoundError(f"Code file not found: {code_path}")

        # Step 2: Create backup: rename code_LLM.{ext} to code_backup_name.{ext}
        file_ext = code_path.suffix
        backup_file = code_path.parent / f"{code_backup_name}{file_ext}"
        code_path.rename(backup_file)
        print(f"Backup created: {backup_file}")
        
        # Step 3: Initialize formatcheck summary file
        formatcheck_summary_file = Path(self.current_code_dir) / f"formatcheck_summary_{tag_name}.jsonl"
        if formatcheck_summary_file.exists():
            formatcheck_summary_file.unlink()  # Remove existing file to start fresh
        
        # Step 4: Load prompts (outside loop for efficiency)
        formatcheck_rules = self._load_prompt(topic=self.topic, prompt_file="formatcheck_rules")
        improve_prompt = self._load_prompt(topic="general_dft", prompt_file="FormatCheck_code_by_rules_improve_stand1")
        judge_prompt = self._load_prompt(topic="general_dft", prompt_file="FormatCheck_code_by_rules_judge_stand1")
        
        # Step 5: Iterative improve and judge
        # Get default temperature for user_model
        user_settings = get_model_settings(user_model)
        default_temperature = user_settings.get("temperature")
        
        flag_obey = False
        count_iter = 0
        previous_judge_reason = None
        previous_flag_obey = None
        user_model_token_dicts = []
        judge_model_token_dicts = []
        
        while not flag_obey and count_iter < max_iter:
            count_iter += 1
            print(f"Format check iteration {count_iter}/{max_iter}...")
            # Calculate temperature for this iteration (increase by 0.15 each time)
            current_temperature = default_temperature + temp_incr * (count_iter - 1)
            # Load code from origin for each iteration
            code_former = self._load_file(str(backup_file))
            
            # Improve the code to obey the format rules
            parts = [improve_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            if code_former is not None:
                parts.append("CODE_FORMER:\n")
                parts.append(code_former)
                parts.append("\n\n")
            if formatcheck_rules is not None:
                parts.append("FORMATCHECK_RULES:\n")
                parts.append(formatcheck_rules)
                parts.append("\n\n")
            if isjudgeinfo and previous_judge_reason is not None:
                parts.append("JUDGE_PREVIOUS_INFO:\n")
                parts.append(f"- flag_obey: {previous_flag_obey}\n")
                parts.append(f"- judge_reason: {previous_judge_reason}\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            user_prompt = "".join(parts)
            # response = self._send_chat(user_model, user_prompt)
            if iscaltoken:
                response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, temperature=current_temperature, iscaltoken=True, require_code_block=True, code_tags=["CODE"])
                if token_dict:
                    user_model_token_dicts.append(token_dict)
            else:
                response = self._send_chat_robust(user_model, user_prompt, max_iter=3, temperature=current_temperature, iscaltoken=False, require_code_block=True, code_tags=["CODE"])
            parsed_response = self._parse_json(response)
            improved_code = self._extract_code_from_fence(response, "CODE") or parsed_response.get("code_content", "")
            improve_summary = parsed_response.get("summary", "")
            self._save_code_file(response=response, anci=code_save_name, isarbi=True, file_ext=file_ext, content_tag="CODE")
            self._log_event({"Content": "Format check code by rules", "LLM": user_model, "iteration": count_iter, "current_temperature": current_temperature, "response": parsed_response})

            # Judge the code
            parts = [judge_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            if improved_code is not None:
                parts.append("IMPROVED_CODE:\n")
                parts.append(improved_code)
                parts.append("\n\n")
            if formatcheck_rules is not None:
                parts.append("FORMATCHECK_RULES:\n")
                parts.append(formatcheck_rules)
                parts.append("\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            user_prompt = "".join(parts)
            # response = self._send_chat(judge_model, user_prompt)
            if iscaltoken:
                response, token_dict = self._send_chat_robust(judge_model, user_prompt, max_iter=3, iscaltoken=True)
                if token_dict:
                    judge_model_token_dicts.append(token_dict)
            else:
                response = self._send_chat_robust(judge_model, user_prompt, max_iter=3, iscaltoken=False)
            parsed_response = self._parse_json(response)
            flag_obey = parsed_response.get("flag_obey", False)
            judge_reason = parsed_response.get("judge_reason", "")
            previous_judge_reason = judge_reason
            previous_flag_obey = flag_obey
            self._log_event({"Content": "Judge the code whether it obeys the rules", "LLM": judge_model, "iteration": count_iter, "response": parsed_response})

            # Save formatcheck summary for this iteration
            summary_entry = {
                "iteration": count_iter,
                "improve_summary": improve_summary,
                "flag_obey": flag_obey,
                "judge_reason": judge_reason
            }
            with open(formatcheck_summary_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")
            
            if flag_obey:
                print(f"Format check passed at iteration {count_iter}. Format check summary saved: {formatcheck_summary_file}")
                break
            else:
                print(f"Format check failed at iteration {count_iter}, continuing...")
        
        if not flag_obey:
            print(f"Format check reached max_iter ({max_iter}) without passing. Format check summary saved: {formatcheck_summary_file}")

        # Aggregate token statistics if requested
        user_model_token_stats = None
        judge_model_token_stats = None
        if iscaltoken:
            # Aggregate user_model token statistics
            if user_model_token_dicts:
                total_input = sum(d.get("input_tokens", 0) for d in user_model_token_dicts if d)
                total_output = sum(d.get("output_tokens", 0) for d in user_model_token_dicts if d)
                total = total_input + total_output
                user_model_token_stats = {"input_tokens": total_input, "output_tokens": total_output, "total": total}
            # Aggregate judge_model token statistics
            if judge_model_token_dicts:
                total_input = sum(d.get("input_tokens", 0) for d in judge_model_token_dicts if d)
                total_output = sum(d.get("output_tokens", 0) for d in judge_model_token_dicts if d)
                total = total_input + total_output
                judge_model_token_stats = {"input_tokens": total_input, "output_tokens": total_output, "total": total}

        # Log summary event
        summary_event = {
            "Content": "FormatCheck_code_by_rules summary",
            "user_model": user_model,
            "judge_model": judge_model,
            "max_iter": max_iter,
            "count_iter": count_iter,
            "flag_obey": flag_obey,
            "formatcheck_summary_file": str(formatcheck_summary_file)
        }
        if iscaltoken:
            summary_event["user_model_token_stats"] = user_model_token_stats
            summary_event["judge_model_token_stats"] = judge_model_token_stats
        self._log_event(summary_event)

        if iscaltoken:
            return improved_code, flag_obey, count_iter, user_model_token_stats, judge_model_token_stats
        else:
            return improved_code, flag_obey, count_iter



class RepoGenerator(DFT_agent):

    async def Generate_repo_by_codefile(self, user_model: str, code_file: str, fs_target_dir: str) -> dict:
        # Read code file
        # original_code = self._load_file("..\\Output_prepare\\prepare_PhysRevLett.80.5607_Fig._2_(a)\\phase_diagram_20251015_160439_.jl")
        original_code = self._load_file(Path(code_file))

        # Construct prompts
        role_prompt = self._load_prompt("GenerateRepo_mcp_use_by_codefile_stand1")
        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        if original_code is not None:
            parts.append("ORIGINAL_CODE:\n")
            parts.append(original_code)
            parts.append("\n\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)

        response = await self._send_chat_through_mcp_dynamic(
            user_model=user_model,
            user_prompt=user_prompt,
            fs_target_dir=fs_target_dir,
            init_servers=["filesystem-mcp"]
        )
        response = self._parse_json(response)

        # Try to load README from the generated filesystem target directory
        readme_text = None
        try:
            fs_path = Path(fs_target_dir)
            readme_path = fs_path / "README.txt"
            if not readme_path.exists():
                readme_path = fs_path / "README.md"
            if readme_path.exists():
                readme_text = self._load_file(readme_path)
        except Exception:
            readme_text = None
            raise RuntimeError("Failed to load README from generated repository.")

        output = {"Content": "Generated repository", "LLM": user_model, "repo_dir": fs_target_dir, "README": readme_text}
        
        self._log_event(output)
        return output
    
    
    async def Generate_repo_by_codefile_hardcode(self, user_model: str, code_file: str, fs_target_dir: str, iscaltoken: bool = False) -> dict:
        # Validate code_file
        if code_file is None:
            raise ValueError("code_file cannot be None. Code generation may have failed. Check logs for details.")
        
        code_path = Path(code_file)
        if not code_path.exists():
            raise FileNotFoundError(f"Code file not found: {code_path}. Code generation may have failed. Check logs for details.")
        
        # Clear existing repo directory so each run writes a fresh repo (same behavior as QMBagents)
        fs_path = Path(fs_target_dir)
        if fs_path.exists():
            shutil.rmtree(fs_path)
        
        # Read code file
        original_code = self._load_file(code_path)

        # Construct repository by hardcode
        code_editor_dft.split_functions(code_file, fs_target_dir)
        code_editor_dft.delete_unused_imports(fs_target_dir, count_type_annotations=True)

        # Generate README via LLM
        role_prompt = self._load_prompt(topic="general_dft", prompt_file="GenerateRepo_README_only_by_codefile_stand1")
        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        if original_code is not None:
            parts.append("ORIGINAL_CODE:\n")
            parts.append(original_code)
            parts.append("\n\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)

        # response = self._send_chat(user_model, user_prompt)
        if iscaltoken:
            response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
        else:
            response = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=False)
            token_dict = None
        response = self._parse_json(response)
        self._save_txt_file(response=response.get("content"), anci="README", txt_dir=fs_target_dir, isarbi=True)
        
        # Try to load README from the generated filesystem target directory
        readme_text = None
        try:
            fs_path = Path(fs_target_dir)
            readme_path = fs_path / "README.txt"
            if not readme_path.exists():
                readme_path = fs_path / "README.md"
            if readme_path.exists():
                readme_text = self._load_file(readme_path)
        except Exception:
            readme_text = None
            raise RuntimeError("Failed to load README from generated repository.")

        output = {"Content": "Generated repository", "LLM": user_model, "repo_dir": fs_target_dir, "README": readme_text, "token_stats": token_dict}
        
        self._log_event(output)
        return output
    

class CodeVerifier(DFT_agent):

    def _normalize_block_name(self, name: str) -> str:
        """Normalize block names for flexible matching.
        
        Handles common variations without hardcoding specific block types:
        - Case-insensitive matching
        - Plural to singular conversion (smart pattern matching)
        - Known aliases (based on ORCA software conventions)
        
        This is a general approach that doesn't require enumerating all possible block types.
        The normalization uses pattern-based rules rather than specific name mappings.
        """
        if not name:
            return ""
        name_lower = name.lower().strip()
        
        # Handle plural -> singular (smart pattern matching)
        # Only apply if it looks like a plural form (ends with 's' but not 'ss', 'us', 'is')
        # This avoids breaking words that naturally end in 's' like 'basis', 'cosmo', etc.
        if name_lower.endswith('s') and len(name_lower) > 3:
            # Check if it's likely a plural (doesn't end with common singular endings)
            # Words ending in 'ss', 'us', 'is', 'os' are often singular, so don't modify
            if not (name_lower.endswith('ss') or 
                    name_lower.endswith('us') or 
                    name_lower.endswith('is') or
                    name_lower.endswith('os')):
                # Likely a plural form, remove trailing 's'
                # This handles: methods->method, bases->basis (but not basis->basi)
                name_lower = name_lower[:-1]
        
        # Handle known aliases (legacy -> modern naming conventions)
        # This is minimal and based on ORCA software conventions, not arbitrary mappings
        # Only includes well-established aliases in the ORCA ecosystem
        alias_map = {
            "cosmo": "cpcm",  # COSMO is legacy name for CPCM in ORCA
        }
        if name_lower in alias_map:
            return alias_map[name_lower]
        
        return name_lower

    def _load_block_contents(self, targets: List[str], repo_dir: str) -> dict:
        """Load content of all target blocks from repo. Returns {block_id: content}."""
        repo_path = Path(repo_dir)
        block_contents = {}
        for block_id in targets:
            block_path = repo_path / block_id
            if block_path.exists():
                block_contents[block_id] = self._load_file(str(block_path))
            else:
                self._log_event({"Content": "Block file not found, skipping", "block_id": block_id, "repo_dir": repo_dir})
        return block_contents

    def _get_block_dependencies(self, user_model: str, targets: List[str], repo_dir: str, iscaltoken: bool = False) -> tuple:
        """
        Load all target block contents, call LLM to analyze dependencies, return dependency map.
        
        Returns:
            (dependencies: dict, token_stats: dict or None)
            dependencies: {block_id: [dep_block_id, ...]} - blocks that must be substituted together
        """
        block_contents = self._load_block_contents(targets, repo_dir)
        if not block_contents:
            return ({t: [] for t in targets}, None)
        
        dep_prompt = self._load_prompt(topic="general_dft", prompt_file="VerifyCode_sandbox_block_dependencies_stand1")
        parts = [dep_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n\n"]
        for block_id, content in block_contents.items():
            parts.append(f"BLOCK_ID: {block_id}\n")
            parts.append("CONTENT:\n")
            parts.append(content)
            if not content.endswith("\n"):
                parts.append("\n")
            parts.append("\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)
        
        if iscaltoken:
            response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
        else:
            response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
            token_dict = None
        
        parsed = self._parse_json(response)
        dependencies = (parsed or {}).get("dependencies", {})
        
        # Ensure every target has an entry; fill missing with empty list
        for t in targets:
            if t not in dependencies:
                dependencies[t] = []
        
        self._log_event({
            "Content": "Block dependency analysis",
            "LLM": user_model,
            "targets": targets,
            "dependencies": dependencies
        })
        
        return (dependencies, token_dict if iscaltoken else None)



    async def Combine_library_zerotest(self, user_model: str, CodeVerifier_library_dir: str = None, current_sandbox_dir: str = None, user_requests: str = None) -> dict:

        role_prompt = self._load_prompt("VerifyCode_mcp_use_sandbox_zerotest_stand1")
        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        if CodeVerifier_library_dir is not None:
            parts.append("CodeVerifier_library:\n")
            parts.append(json.dumps(CodeVerifier_library_dir, ensure_ascii=False))
            parts.append("\n\n")
        if user_requests is not None:
            parts.append("User_requests:\n")
            parts.append(json.dumps(user_requests, ensure_ascii=False))
            parts.append("\n\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)

        if CodeVerifier_library_dir is not None:
            response = await self._send_chat_through_mcp_dynamic(
                user_model=user_model,
                user_prompt=user_prompt,
                fs_target_dir=current_sandbox_dir,
                reference_dir_dict={"libs": CodeVerifier_library_dir},
                init_servers=["filesystem-mcp"]
            )
        else:
            response = await self._send_chat_through_mcp_dynamic(
                user_model=user_model,
                user_prompt=user_prompt,
                fs_target_dir=current_sandbox_dir,
                init_servers=["filesystem-mcp"]
            )
        save_code_output = self._save_code_file(response, code_dir=current_sandbox_dir, content_tag="CODE")
        code_file = save_code_output.get("code_file") if save_code_output else None
        output = {"Content": "Zerotest code", "LLM": user_model, "code_file": code_file, "response": response}

        # self._log_event(output)
        return output
    
    
    async def Verify_code_single_repocode_generation(self, user_model: str, CodeVerifier_library_dir: str = None, current_sandbox_dir: str = None, repo_dir: str = None, upper_num: int = 0, ismcp: bool = False) ->dict:
       
        # Extract information from the target repo----------------------------------------
        prefix_list = ["site", "hamiltonian", "initialstate", "observable", "effector"]
        func_list = []
        repo_path = Path(repo_dir)
        for file in repo_path.iterdir():
            if not file.is_file():
                continue
            for prefix in prefix_list:
                if file.name.startswith(prefix):
                    func_name = file.stem  
                    func_list.append(f"{func_name}.{file.suffix.lstrip('.')}")
        # print(f"Extracted functions: {func_list}") # for debug
    
        # Extract the README file-----------------------------------------------------------
        readme_path = Path(repo_dir) / "README.txt"
        readme_content = self._load_file(readme_path)
        # print(readme_content) # for debug

        # Iterative verification------------------------------------------------------------
        for i in range(len(func_list)):
            if upper_num >0 and i >= upper_num:
                break

            target_function_name = func_list[i]
            target_function = self._load_file(Path(repo_dir) / target_function_name)
            # print(target_function)  # for debug

            if ismcp == True:
                role_prompt = self._load_prompt("VerifyCode_mcp_use_sandbox_single_repocode_generation_stand1")
                parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                if target_function is not None:
                    parts.append(f"TARGET_FUNCTION:\n")
                    parts.append(target_function)
                    parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)

                response = await self._send_chat_through_mcp_dynamic(
                    user_model=user_model,
                    user_prompt=user_prompt,
                    fs_target_dir=current_sandbox_dir,
                    reference_dir_dict={"libs": CodeVerifier_library_dir},
                    init_servers=["filesystem-mcp"]
                )
            if ismcp == False: # To achieve higher efficiency
                role_prompt = self._load_prompt("VerifyCode_sandbox_single_repocode_generation_stand1")
                lib_text = None
                if CodeVerifier_library_dir is not None:
                    lib_dir = Path(CodeVerifier_library_dir)
                    py_path = lib_dir / "CodeVerifier_library_compact.py"
                    jl_path = lib_dir / "CodeVerifier_library_compact.jl"
                    if py_path.exists():
                        lib_text = self._load_file(py_path)
                    elif jl_path.exists():
                        lib_text = self._load_file(jl_path)
                main_function = None
                if repo_dir is not None:
                    py_path = Path(repo_dir) / "main.py"
                    jl_path = Path(repo_dir) / "main.jl"
                    if py_path.exists():
                        main_function = self._load_file(py_path)
                    elif jl_path.exists():
                        main_function = self._load_file(jl_path)

                parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                if lib_text is not None:
                    parts.append("CODEVERIFIER_LIBRARY_COMPACT:\n")
                    parts.append(lib_text)
                    parts.append("\n\n")
                if main_function is not None:
                    parts.append("MAIN_FUNCTION:\n")
                    parts.append(main_function)
                    parts.append("\n\n")
                if target_function is not None:
                    parts.append(f"TARGET_FUNCTION:\n")
                    parts.append(target_function)
                    parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)

                print(f"verify code: {target_function_name} generating...")
                response, _ = self._send_chat(user_model=user_model, user_prompt=user_prompt)
            
            function_name_nosuffix = target_function_name.rsplit('.', 1)[0]
            save_code_output = self._save_code_file(response, anci=f"verify_{function_name_nosuffix}", code_dir=current_sandbox_dir, isarbi=True, content_tag="CODE")
            

        output = {"Content": "Single repocode verification code generation", "LLM": user_model, "repo_dir": repo_dir, "sandbox_dir": current_sandbox_dir}
        self._log_event(output)

        return output
    

    async def Verify_code_single_repocode_generation_concurr(self, user_model: str, CodeVerifier_library_dir: str = None, current_sandbox_dir: str = None, repo_dir: str = None, upper_num: int = 0, concurr_num: int = 5) ->dict:
       
        # Extract information from the target repo----------------------------------------
        if self.topic == "dmrg":
            prefix_list = ["site", "hamiltonian", "initialstate", "observable", "effector"]
        func_list = []
        repo_path = Path(repo_dir)
        for file in repo_path.iterdir():
            if not file.is_file():
                continue
            for prefix in prefix_list:
                if file.name.startswith(prefix):
                    func_name = file.stem  
                    func_list.append(f"{func_name}.{file.suffix.lstrip('.')}")
        # print(f"Extracted functions: {func_list}") # for debug
    
        # Extract the README file-----------------------------------------------------------
        readme_path = Path(repo_dir) / "README.txt"
        readme_content = self._load_file(readme_path)
        # print(readme_content) # for debug

        # Iterative verification (concurrent)-------------------------------------------------
        # Prepare list of target functions respecting upper_num
        if upper_num > 0:
            targets = func_list[:upper_num]
        else:
            targets = func_list

        sem = asyncio.Semaphore(concurr_num)

        async def _process_target(target_function_name: str):
            async with sem:
                target_function = self._load_file(Path(repo_dir) / target_function_name)

                role_prompt = self._load_prompt(topic=self.topic, prompt_file="VerifyCode_sandbox_single_repocode_generation_stand1")
                lib_text = None
                if CodeVerifier_library_dir is not None:
                    lib_dir = Path(CodeVerifier_library_dir)
                    py_path = lib_dir / "CodeVerifier_library_compact.py"
                    jl_path = lib_dir / "CodeVerifier_library_compact.jl"
                    if py_path.exists():
                        lib_text = self._load_file(py_path)
                    elif jl_path.exists():
                        lib_text = self._load_file(jl_path)
                if lib_text is None:
                    raise ValueError("CodeVerifier_library_compact file not found in the specified library directory.")
                
                main_function = None
                if repo_dir is not None:
                    py_path = Path(repo_dir) / "main.py"
                    jl_path = Path(repo_dir) / "main.jl"
                    if py_path.exists():
                        main_function = self._load_file(py_path)
                    elif jl_path.exists():
                        main_function = self._load_file(jl_path)

                parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                if lib_text is not None:
                    parts.append("CODEVERIFIER_LIBRARY_COMPACT:\n")
                    parts.append(lib_text)
                    parts.append("\n\n")
                if main_function is not None:
                    parts.append("MAIN_FUNCTION:\n")
                    parts.append(main_function)
                    parts.append("\n\n")
                if target_function is not None:
                    parts.append(f"TARGET_FUNCTION:\n")
                    parts.append(target_function)
                    parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)

                print(f"verify code: {target_function_name} generating...")

                # _send_chat is blocking; run it in a thread to avoid blocking the event loop
                response = await asyncio.to_thread(self._send_chat, user_model, user_prompt)

                function_name_nosuffix = target_function_name.rsplit('.', 1)[0]
                # _save_code_file does IO; run in thread as well (use keyword args to ensure correct mapping)
                await asyncio.to_thread(self._save_code_file, response, anci=f"verify_{function_name_nosuffix}", code_dir=current_sandbox_dir, isarbi=True, content_tag="CODE")

                return target_function_name

        # schedule workers and wait
        tasks = [asyncio.create_task(_process_target(name)) for name in targets]
        if tasks:
            await asyncio.gather(*tasks)
        output = {"Content": "Single repocode verification code generation", "LLM": user_model, "repo_dir": repo_dir, "sandbox_dir": current_sandbox_dir}
        self._log_event(output)

        return output


    async def Verify_code_single_repocode_generation_autolibrary_concurr(self, user_model: str, current_sandbox_dir: str = None, code_file: str = None, repo_dir: str = None, upper_num: int = 0, concurr_num: int = 5) ->dict:
       
        # Extract information from the target repo----------------------------------------
        if self.topic == "dmrg":
            prefix_list = ["site", "hamiltonian", "initialstate", "observable", "effector"]
        elif self.topic == "nnwf":
            prefix_list = ["hilbert", "statemodel", "compset", "hamiltonian", "observable", "effector"]
        func_list = []
        repo_path = Path(repo_dir)
        for file in repo_path.iterdir():
            if not file.is_file():
                continue
            for prefix in prefix_list:
                if file.name.startswith(prefix):
                    func_name = file.stem  
                    func_list.append(f"{func_name}.{file.suffix.lstrip('.')}")
        # print(f"Extracted functions: {func_list}") # for debug

        # Decide which library to use--------------------------------------------------------
        CodeVerifier_library_dir = "../CodeVerifier_library_auto/" + "CodeVerifier_library_" + self.topic + "/verifier_code"
        
        # Get all available library files
        lib_dir = Path(CodeVerifier_library_dir)
        available_library_files = []
        if lib_dir.exists():
            for file in lib_dir.iterdir():
                if file.is_file():
                    available_library_files.append(file.name)
        if available_library_files is None:
            raise ValueError("No available library files found in the specified library directory.")
        
        # Load code_file and decide which library file(s) to use (non-concurrent, execute once)
        code_file_content = None
        if code_file is not None:
            code_file_path = Path(code_file)
            if code_file_path.exists():
                code_file_content = self._load_file(code_file_path)
        if code_file_content is None:
            raise ValueError("code file content is None. ")
        
        # Use LLM to decide which library file(s) to use based on code_file content
        selected_library_files = []
        if code_file_content is not None and available_library_files:
            decide_prompt = self._load_prompt(topic="general_dft", prompt_file="VerifyCode_sandbox_single_repocode_generation_decidelibrary_stand1")
            parts = [decide_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            parts.append("TARGET_FUNCTION:\n")
            parts.append(code_file_content)
            parts.append("\n\n")
            parts.append("AVAILABLE_LIBRARY_FILES:\n")
            parts.append(json.dumps(available_library_files, ensure_ascii=False))
            parts.append("\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            decide_user_prompt = "".join(parts)
            
            decide_response = self._send_chat_robust(user_model, decide_user_prompt, max_iter=3)
            decide_parsed = self._parse_json(decide_response)
            selected_library_files = (decide_parsed or {}).get("selected_files", [])
            self._log_event({"Content": "Decide which library file(s) to use based on code_file content", "LLM": user_model, "selected_library_files": selected_library_files})
            
            if not selected_library_files:
                # Fallback: use first available library file when LLM returned empty
                selected_library_files = [available_library_files[0]]
                self._log_event({"Content": "Fallback: using first available library file (LLM returned empty)", "selected_library_files": selected_library_files})
        else:
            raise ValueError("code_file is required and must exist, or no available library files found")
    
        # Extract the README file-----------------------------------------------------------
        readme_path = Path(repo_dir) / "README.txt"
        readme_content = self._load_file(readme_path)
        # print(readme_content) # for debug

        # Iterative verification (concurrent)-------------------------------------------------
        # Prepare list of target functions respecting upper_num
        if upper_num > 0:
            targets = func_list[:upper_num]
        else:
            targets = func_list

        sem = asyncio.Semaphore(concurr_num)

        async def _process_target(target_function_name: str, selected_files: list):
            async with sem:
                target_function = self._load_file(Path(repo_dir) / target_function_name)

                # Load selected library file(s) and combine them
                lib_text = None
                for selected_file in selected_files:
                    selected_path = lib_dir / selected_file
                    if selected_path.exists():
                        file_content = self._load_file(selected_path)
                        if lib_text is None:
                            lib_text = file_content
                        else:
                            # Combine multiple library files
                            lib_text = lib_text + "\n\n" + file_content
                if lib_text is None:
                    raise ValueError(f"Selected library file(s) {selected_files} not found in {CodeVerifier_library_dir}")

                role_prompt = self._load_prompt(topic=self.topic, prompt_file="VerifyCode_sandbox_single_repocode_generation_stand1")
                
                main_function = None
                if repo_dir is not None:
                    py_path = Path(repo_dir) / "main.py"
                    jl_path = Path(repo_dir) / "main.jl"
                    if py_path.exists():
                        main_function = self._load_file(py_path)
                    elif jl_path.exists():
                        main_function = self._load_file(jl_path)

                parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                if lib_text is not None:
                    parts.append("CODEVERIFIER_LIBRARY_COMPACT:\n")
                    parts.append(lib_text)
                    parts.append("\n\n")
                if main_function is not None:
                    parts.append("MAIN_FUNCTION:\n")
                    parts.append(main_function)
                    parts.append("\n\n")
                if target_function is not None:
                    parts.append(f"TARGET_FUNCTION:\n")
                    parts.append(target_function)
                    parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)

                print(f"verify code: {target_function_name} generating...")

                # _send_chat is blocking; run it in a thread to avoid blocking the event loop
                response = await asyncio.to_thread(self._send_chat_robust, user_model, user_prompt, max_iter=3)

                function_name_nosuffix = target_function_name.rsplit('.', 1)[0]
                # _save_code_file does IO; run in thread as well (use keyword args to ensure correct mapping)
                await asyncio.to_thread(self._save_code_file, response, anci=f"verify_{function_name_nosuffix}", code_dir=current_sandbox_dir, isarbi=True, content_tag="CODE")

                return target_function_name

        # schedule workers and wait
        tasks = [asyncio.create_task(_process_target(name, selected_library_files)) for name in targets]
        if tasks:
            await asyncio.gather(*tasks)
        output = {"Content": "Single repocode verification code generation", "LLM": user_model, "repo_dir": repo_dir, "sandbox_dir": current_sandbox_dir}
        self._log_event(output)

        return output


    async def Verify_code_single_repocode_generation_autolibrary_split_concurr(self, user_model: str, current_sandbox_dir: str = None, code_file: str = None, repo_dir: str = None, upper_num: int = 0, concurr_num: int = 5, iscaltoken: bool = False) ->dict:
        """
        Generate unittest files by substituting target blocks into verifier templates.
        
        Workflow:
        1. Extract target functions from repo (e.g., keywords_none.inp, inputblocks_cpcm.inp)
        2. Select verifier library files using LLM
        3. For each target (concurrently):
           - Copy target block to verify_{name}_tobetest.inp
           - Find matching block in verifier template
           - Substitute target block into template
           - Save as verify_{name}_unittest.inp
        
        Execution is handled by Verify_code_single_repocode_execution_concurr (separate step).
        When iscaltoken=True, output includes "token_stats" from the decide-library LLM call.
        """
       
        # Extract information from the target repo----------------------------------------
        if self.topic == "dmrg":
            prefix_list = ["site", "hamiltonian", "initialstate", "observable", "effector"]
        elif self.topic == "nnwf":
            prefix_list = ["hilbert", "statemodel", "compset", "hamiltonian", "observable", "effector"]
        elif self.topic == "qcmb":
            prefix_list = ["hamiltonian", "circinit", "circevol", "effector", "observable"]
        elif self.topic == "dft_qc":
            prefix_list = ["keywords", "inputblocks", "geometryblocks"]
        func_list = []
        repo_path = Path(repo_dir)
        for file in repo_path.iterdir():
            if not file.is_file():
                continue
            for prefix in prefix_list:
                if file.name.startswith(prefix):
                    print(file)
                    func_name = file.stem  
                    func_list.append(f"{func_name}.{file.suffix.lstrip('.')}")
        print(f"Extracted functions: {func_list}") # for debug

        
        # Decide which library to use--------------------------------------------------------
        CodeVerifier_library_dir = "../CodeVerifier_library_auto/" + "CodeVerifier_library_" + self.topic + "/verifier_code"
        
        # Get all available library files
        lib_dir = Path(CodeVerifier_library_dir)
        available_library_files = []
        if lib_dir.exists():
            for file in lib_dir.iterdir():
                if file.is_file():
                    available_library_files.append(file.name)
        if not available_library_files:  # Check if list is empty (not None)
            raise ValueError(f"No available library files found in the specified library directory: {CodeVerifier_library_dir}")
        
        # Load code_file and decide which library file(s) to use (non-concurrent, execute once)
        code_file_content = None
        if code_file is not None and code_file != "":
            code_file_path = Path(code_file)
            if code_file_path.exists():
                code_file_content = self._load_file(code_file_path)
            else:
                raise ValueError(f"code_file does not exist: {code_file}")
        else:
            raise ValueError(f"code_file parameter is required but was: {code_file}")
        
        if code_file_content is None:
            raise ValueError(f"Failed to load code_file content from: {code_file}")
        
        # Use LLM to decide which library file(s) to use based on code_file content
        selected_library_files = []
        if code_file_content is not None and available_library_files:
            decide_prompt = self._load_prompt(topic="general_dft", prompt_file="VerifyCode_sandbox_single_repocode_generation_decidelibrary_stand1")
            parts = [decide_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            parts.append("TARGET_FUNCTION:\n")
            parts.append(code_file_content)
            parts.append("\n\n")
            parts.append("AVAILABLE_LIBRARY_FILES:\n")
            parts.append(json.dumps(available_library_files, ensure_ascii=False))
            parts.append("\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            decide_user_prompt = "".join(parts)
            
            decide_token_dict = None
            if iscaltoken:
                decide_response, decide_token_dict = self._send_chat_robust(user_model, decide_user_prompt, max_iter=3, iscaltoken=True)
            else:
                decide_response = self._send_chat_robust(user_model, decide_user_prompt, max_iter=3)
            decide_parsed = self._parse_json(decide_response)
            selected_library_files = (decide_parsed or {}).get("selected_files", [])
            self._log_event({"Content": "Decide which library file(s) to use based on code_file content", "LLM": user_model, "selected_library_files": selected_library_files})
            
            if not selected_library_files:
                # Fallback: use first available library file when LLM returned empty
                selected_library_files = [available_library_files[0]]
                self._log_event({"Content": "Fallback: using first available library file (LLM returned empty)", "selected_library_files": selected_library_files})
        else:
            if not available_library_files:
                raise ValueError(f"No available library files found in directory: {CodeVerifier_library_dir}. Please ensure verifier code files exist in this directory.")
            else:
                raise ValueError(f"Unexpected error: code_file_content is None but code_file was provided: {code_file}")
    
        # Extract the README file-----------------------------------------------------------
        readme_path = Path(repo_dir) / "README.txt"
        readme_content = self._load_file(readme_path)
        # print(readme_content) # for debug

        # Iterative verification (concurrent)-------------------------------------------------
        # Prepare list of target functions respecting upper_num
        if upper_num > 0:
            targets = func_list[:upper_num]
        else:
            targets = func_list

        # Get block dependencies for multi-block substitution (dft_qc only)
        dependencies = {}
        dep_token_dict = None
        if self.topic == "dft_qc":
            dep_result, dep_token_dict = self._get_block_dependencies(user_model, targets, repo_dir, iscaltoken)
            dependencies = dep_result
        else:
            dependencies = {t: [] for t in targets}

        sem = asyncio.Semaphore(concurr_num)

        async def _process_target(target_function_name: str, selected_files: list, dependencies_map: dict):
            async with sem:
                # Step 1: Copy and rename the target function to-be-test
                target_function_path = Path(repo_dir) / target_function_name
                target_function = self._load_file(target_function_path)
                function_name_nosuffix = target_function_name.rsplit('.', 1)[0]
                file_ext = target_function_path.suffix
                
                # Copy target function to sandbox directory with new name
                sandbox_path = Path(current_sandbox_dir)
                sandbox_path.mkdir(parents=True, exist_ok=True)
                tobetest_file = sandbox_path / f"verify_{function_name_nosuffix}_tobetest{file_ext}"
                shutil.copy2(target_function_path, tobetest_file)

                # Step 2: Direct block substitution (skip LLM environment generation)
                # Parse target function name to extract block type and sub-type
                # Format: {block_type}_{sub_type} or {block_type}_none
                # Examples: keywords_none, inputblocks_cpcm, geometryblocks_none
                target_parts = function_name_nosuffix.split('_', 1)
                if len(target_parts) == 1:
                    target_block_type = target_parts[0]
                    target_sub_type = "none"
                else:
                    target_block_type = target_parts[0]
                    target_sub_type = target_parts[1] if target_parts[1] != "none" else ""
                
                # Find matching verifier file that contains the same block type
                verifier_file_content = None
                verifier_file_path = None
                for selected_file in selected_files:
                    selected_path = lib_dir / selected_file
                    if selected_path.exists():
                        file_content = self._load_file(selected_path)
                        # Check if this verifier file contains the matching block type
                        # Try two formats:
                        # 1. #Tags_{block_type}_{sub_type} format (standard format)
                        # 2. Plain comment format like "# {block_name}" (legacy format in some verifier files)
                        
                        # Format 1: #Tags_ pattern
                        tag_pattern = re.compile(
                            rf"^\s*#Tags_{re.escape(target_block_type)}(?:_([A-Za-z0-9]+))?\s*$",
                            re.MULTILINE
                        )
                        has_tag_format = tag_pattern.search(file_content)
                        
                        # Format 2: Plain comment format or ORCA syntax patterns
                        # General pattern matching that works for any sub_type without hardcoding specific names
                        has_comment_format = False
                        if target_block_type == "inputblocks":
                            # For inputblocks, check for both comment format and ORCA syntax format
                            # Normalize sub_type for flexible matching
                            normalized_sub_type = self._normalize_block_name(target_sub_type) if target_sub_type else ""
                            
                            if normalized_sub_type:
                                # Pattern 1: Comment format like "# method", "# basis", "# cpcm"
                                # Escape the sub_type for regex and allow optional plural/singular
                                comment_pattern = rf"^\s*#\s+{re.escape(normalized_sub_type)}s?\s"
                                has_comment_format = bool(re.search(comment_pattern, file_content, re.MULTILINE | re.IGNORECASE))
                                
                                # Pattern 2: ORCA syntax format like "%method", "%basis", "%cpcm"
                                # This is more reliable as it matches actual ORCA input syntax
                                if not has_comment_format:
                                    orca_pattern = rf"^\s*%{re.escape(normalized_sub_type)}s?\b"
                                    has_comment_format = bool(re.search(orca_pattern, file_content, re.MULTILINE | re.IGNORECASE))
                        elif target_block_type == "keywords":
                            # Keywords block: look for "!" line (ORCA keyword line) or "# keywords" comment
                            has_comment_format = bool(re.search(r"^\s*!\s+", file_content, re.MULTILINE) or 
                                                      re.search(r"^\s*#\s+keywords?\s", file_content, re.MULTILINE | re.IGNORECASE))
                        elif target_block_type == "geometryblocks":
                            # Geometry block: look for "* xyz" pattern or "# geometry" comment
                            has_comment_format = bool(re.search(r"^\s*\*\s+(xyz|xyzfile)", file_content, re.MULTILINE) or
                                                      re.search(r"^\s*#\s+geometry\s", file_content, re.MULTILINE | re.IGNORECASE))
                        
                        if has_tag_format or has_comment_format:
                            verifier_file_content = file_content
                            verifier_file_path = selected_path
                            break
                
                # If no verifier has the matching block, use first selected file as base for block insertion
                if verifier_file_content is None:
                    for selected_file in selected_files:
                        selected_path = lib_dir / selected_file
                        if selected_path.exists():
                            verifier_file_content = self._load_file(selected_path)
                            verifier_file_path = selected_path
                            break
                    if verifier_file_content is None:
                        raise ValueError(f"No verifier files found in {lib_dir} from selected files: {selected_files}")
                




                # Copy selected verifier (library) content to sandbox as background file (same as QMB: sandbox has background)
                background_file = sandbox_path / f"verify_{function_name_nosuffix}_background{file_ext}"
                background_file.write_text(verifier_file_content, encoding='utf-8')
                
                # Step 3: Perform direct block substitution or block insertion
                # Parse verifier file to find blocks
                from utils.code_editor_dft import _find_tag_starts, _slice_chunks
                
                verifier_lines = verifier_file_content.splitlines(keepends=True)
                tag_starts = _find_tag_starts(verifier_lines)
                
                # If no #Tags_ markers found, try to parse based on ORCA syntax patterns
                # This is more general than hardcoding comment text mappings
                if not tag_starts:
                    tag_starts = []
                    i = 0
                    while i < len(verifier_lines):
                        line = verifier_lines[i]
                        stripped = line.strip()
                        
                        # Detect block types by ORCA syntax patterns (not by comment text)
                        # This works for any block type without hardcoding mappings
                        
                        # Keywords block: starts with "!" (ORCA keyword line)
                        if stripped.startswith("!"):
                            if not any(bt == "keywords" for _, bt, _ in tag_starts):
                                tag_starts.append((i, "keywords", ""))
                        
                        # Input blocks: start with "%" followed by block name (e.g., %method, %basis, %cpcm)
                        elif stripped.startswith("%"):
                            # Extract block name from "%blockname" or "%blockname ..."
                            block_match = re.match(r"^\s*%([a-zA-Z0-9_]+)", stripped, re.IGNORECASE)
                            if block_match:
                                block_name = block_match.group(1).lower()
                                # Normalize block names using general normalization function
                                # This handles plural/singular and aliases without hardcoding specific names
                                normalized_name = self._normalize_block_name(block_name)
                                tag_starts.append((i, "inputblocks", normalized_name))
                        
                        # Geometry block: starts with "* xyz" or "* xyzfile"
                        elif stripped.startswith("* xyz") or stripped.startswith("* xyzfile"):
                            if not any(bt == "geometryblocks" for _, bt, _ in tag_starts):
                                tag_starts.append((i, "geometryblocks", ""))
                        
                        i += 1
                
                chunks = _slice_chunks(verifier_lines, tag_starts)

                # Build substitution_set = primary target + its dependencies (all substituted simultaneously)
                substitution_set = [target_function_name] + (dependencies_map.get(target_function_name) or [])
                substitution_set = list(dict.fromkeys(substitution_set))
                substitution_set = [b for b in substitution_set if (Path(repo_dir) / b).exists()]
                if not substitution_set:
                    substitution_set = [target_function_name]

                block_contents = {}
                for bid in substitution_set:
                    bp = Path(repo_dir) / bid
                    if bp.exists():
                        block_contents[bid] = self._load_file(str(bp))

                def _chunk_to_block_id(bt: str, sub: str) -> Optional[str]:
                    if sub and sub.lower() == "unknown":
                        return None
                    suf = (sub or "none").lower()
                    return f"{bt}_{suf}{file_ext}"

                def _prepare_block_lines(content: str, bt: str, sub: str) -> List[str]:
                    lines = content.splitlines(keepends=True)
                    expected_tag = f"#Tags_{bt}"
                    if sub and sub != "none" and bt == "inputblocks":
                        expected_tag += f"_{sub}"
                    has_tag = False
                    for ln in lines:
                        s = ln.strip()
                        if s and s.startswith("#Tags_") and s.startswith(expected_tag):
                            has_tag = True
                            break
                    if not has_tag:
                        lines.insert(0, f"{expected_tag}\n")
                    if not lines or not lines[-1].endswith("\n"):
                        lines.append("\n")
                    return lines

                matching_blocks = set()
                chunk_to_block = {}
                for idx, (start, end, bt, sub) in enumerate(chunks):
                    bid = _chunk_to_block_id(bt, sub)
                    if bid and bid in substitution_set:
                        matching_blocks.add(bid)
                        chunk_to_block[idx] = bid

                unmatched_blocks = [b for b in substitution_set if b not in matching_blocks]

                # Step A: Build base content by substituting matched blocks in place
                new_lines = []
                if chunks:
                    for idx, (start, end, bt, sub) in enumerate(chunks):
                        if idx == 0:
                            new_lines.extend(verifier_lines[:start])
                        else:
                            prev_end = chunks[idx - 1][1]
                            new_lines.extend(verifier_lines[prev_end:start])
                        if idx in chunk_to_block:
                            bid = chunk_to_block[idx]
                            new_lines.extend(_prepare_block_lines(block_contents[bid], bt, sub))
                        else:
                            new_lines.extend(verifier_lines[start:end])
                    new_lines.extend(verifier_lines[chunks[-1][1]:])
                else:
                    new_lines.extend(verifier_lines)

                # Step B: Insert unmatched blocks by type
                if unmatched_blocks:
                    def _block_type_of(bid):
                        return Path(bid).stem.split("_", 1)[0]

                    unmatched_kw = [b for b in unmatched_blocks if _block_type_of(b) == "keywords"]
                    unmatched_ib = [b for b in unmatched_blocks if _block_type_of(b) == "inputblocks"]
                    unmatched_geo = [b for b in unmatched_blocks if _block_type_of(b) == "geometryblocks"]

                    def _build_insert_lines(bids):
                        lines = []
                        for bid in bids:
                            if bid in block_contents:
                                parts = Path(bid).stem.split("_", 1)
                                bt = parts[0]
                                sub = (parts[1] or "none") if len(parts) > 1 else "none"
                                lines.extend(_prepare_block_lines(block_contents[bid], bt, sub))
                        return lines

                    def _find_geometry_start(lines):
                        for i, ln in enumerate(lines):
                            s = ln.strip()
                            if s.startswith("* xyz") or s.startswith("* xyzfile"):
                                for j in range(i - 1, -1, -1):
                                    if lines[j].strip().startswith("#Tags_geometryblocks") or lines[j].strip().startswith("# geometry"):
                                        return j
                                return i
                        return len(lines)

                    # Geometry: append at end
                    if unmatched_geo:
                        new_lines.append("\n")
                        new_lines.extend(_build_insert_lines(unmatched_geo))

                    # Inputblocks: insert before geometry
                    if unmatched_ib:
                        geom_start = _find_geometry_start(new_lines)
                        insert_lines = ["\n"] + _build_insert_lines(unmatched_ib) + ["\n"]
                        new_lines = new_lines[:geom_start] + insert_lines + new_lines[geom_start:]

                    # Keywords: prepend at beginning
                    if unmatched_kw:
                        insert_lines = _build_insert_lines(unmatched_kw) + ["\n"]
                        new_lines = insert_lines + new_lines

                if matching_blocks and not unmatched_blocks:
                    log_content = "Direct block substitution completed"
                elif matching_blocks and unmatched_blocks:
                    log_content = f"Direct substitution + insertion of unmatched blocks: {unmatched_blocks}"
                    print(f"verify code: {target_function_name} generating (substitution + insertion, matched={list(matching_blocks)}, unmatched={unmatched_blocks})...")
                else:
                    log_content = f"Block insertion completed (no matching block in verifier), inserted: {unmatched_blocks}"
                    print(f"verify code: {target_function_name} generating (block insertion, blocks: {substitution_set})...")
                
                # Save as unittest file
                merged_file = sandbox_path / f"verify_{function_name_nosuffix}_unittest{file_ext}"
                merged_content = "".join(new_lines)
                merged_file.write_text(merged_content, encoding='utf-8')
                
                self._log_event({
                    "Content": log_content,
                    "function_name": function_name_nosuffix,
                    "verifier_file": str(verifier_file_path),
                    "target_block": f"{target_block_type}_{target_sub_type}",
                    "substitution_set": substitution_set,
                    "unittest_file": str(merged_file)
                })

                return target_function_name

        # schedule workers and wait
        tasks = [asyncio.create_task(_process_target(name, selected_library_files, dependencies)) for name in targets]
        if tasks:
            await asyncio.gather(*tasks)
        output = {"Content": "Single repocode verification code generation", "LLM": user_model, "repo_dir": repo_dir, "sandbox_dir": current_sandbox_dir}
        if iscaltoken and (decide_token_dict or dep_token_dict):
            total_input = (decide_token_dict or {}).get("input_tokens", 0)
            total_output = (decide_token_dict or {}).get("output_tokens", 0)
            if dep_token_dict:
                total_input += dep_token_dict.get("input_tokens", 0)
                total_output += dep_token_dict.get("output_tokens", 0)
            if total_input + total_output > 0:
                output["token_stats"] = {
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "total": total_input + total_output,
                }
        self._log_event(output)

        return output


    def Verify_code_single_repocode_execution(self, user_model: str, current_sandbox_dir: str = None, upper_num: int = 0, use_mcp_run_orca: bool = False, mcp_orca_timeout: int = 60):

        # find all .py/.jl/.inp files in the sandbox directory--------------------------------
        files_list = []
        if current_sandbox_dir is None:
            raise ValueError("current_sandbox_dir is None.")
        sandbox_path = Path(current_sandbox_dir)
        if not sandbox_path.exists():
            raise FileNotFoundError(f"sandbox directory '{sandbox_path}' not found.")

        for f in sandbox_path.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() in (".py", ".jl", ".inp"):
                files_list.append(f.name)

        files_list = sorted(files_list)
        # print(f"Sandbox files (.py/.jl/.inp): {files_list}") # for debug

        # Iterative execution------------------------------------------------------------
        judge_results = []
        for i in range(len(files_list)):
            if upper_num > 0 and i >= upper_num:
                break

            target_file_name = files_list[i]
            target_file_path = sandbox_path / target_file_name
            target_file = self._load_file(target_file_path)
            print(f"verify code: {target_file_name} executing...")
            
            file_ext = target_file_path.suffix.lower()
            if file_ext == ".inp" and use_mcp_run_orca:
                # MCP run_orca + evaluate (short timeout, early terminate)
                role_prompt = self._load_prompt(topic="general_dft", prompt_file="VerifyCode_sandbox_single_repocode_execution_mcp_run_orca_stand1")
                parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                parts.append(f"TARGET_FILE_NAME:\n")
                parts.append(json.dumps(target_file_name, ensure_ascii=False))
                parts.append("\n\n")
                parts.append(f"TARGET_FILE_CONTENT:\n")
                parts.append(json.dumps(target_file, ensure_ascii=False))
                parts.append("\n\n")
                parts.append(f"ORCA_TIMEOUT_SECONDS:\n")
                parts.append(json.dumps(mcp_orca_timeout, ensure_ascii=False))
                parts.append("\n\n-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)
                response = asyncio.run(MCP_toolbox.send_chat_through_mcp_dynamic(
                    user_model=user_model,
                    user_prompt=user_prompt,
                    fs_target_dir=str(sandbox_path),
                    init_servers=["program-dft-mcp", "filesystem-mcp"],
                    program_dft_input_path=str(target_file_path),
                    program_dft_timeout=mcp_orca_timeout,
                ))
                if not isinstance(response, str):
                    response = self._normalize_mcp_response_to_str(response)
                response = self._parse_json(response)
                judge_results.append(response)
                continue

            # Select run function based on file extension (Python/Julia or .inp without MCP)
            if file_ext == ".py":
                run_func = run_program.run_python_file
            elif file_ext == ".jl":
                run_func = run_program.run_julia_file
            elif file_ext == ".inp":
                run_func = run_program.run_orca_file
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}")
            
            result_target_file = run_func(str(target_file_path), str(sandbox_path), 600)
            result_target_file_exitcode = result_target_file.get("exitcode")
            result_target_file_stdout = result_target_file.get("stdout", "")
            result_target_file_stderr = result_target_file.get("stderr", "")
            result_target_file_timeout = result_target_file.get("timeout", False)
            result_target_file_running_successfully = result_target_file.get("running_successfully")

            # Evaluate execution result by LLM--------------------------------
            role_prompt = self._load_prompt(topic="general_dft", prompt_file="VerifyCode_sandbox_single_repocode_execution_evaluation_stand1")
            parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            if target_file_name is not None:
                parts.append(f"TARGET_FILE_NAME:\n")
                parts.append(json.dumps(target_file_name, ensure_ascii=False))
                parts.append("\n\n")
            if target_file is not None:
                parts.append(f"TARGET_FILE_CONTENT:\n")
                parts.append(json.dumps(target_file, ensure_ascii=False))
                parts.append("\n\n")
            if result_target_file_exitcode is not None:
                parts.append(f"RESULT_TARGET_FILE_EXITCODE:\n")
                parts.append(json.dumps(result_target_file_exitcode, ensure_ascii=False))
                parts.append("\n\n")
            if result_target_file_stdout is not None:
                stdout_truncated = result_target_file_stdout
                if len(stdout_truncated) > 15000:
                    # Keep final 15000 chars - contains important termination messages, results, and errors
                    truncated_chars = len(result_target_file_stdout) - 15000
                    stdout_truncated = f"... (truncated: {truncated_chars} characters from beginning) ...\n\n" + stdout_truncated[-15000:]
                parts.append(f"RESULT_TARGET_FILE_STDOUT:\n")
                parts.append(json.dumps(stdout_truncated, ensure_ascii=False))
                parts.append("\n\n")
            if result_target_file_stderr is not None:
                parts.append(f"RESULT_TARGET_FILE_STDERR:\n")
                parts.append(json.dumps(result_target_file_stderr, ensure_ascii=False))
                parts.append("\n\n")
            if result_target_file_timeout is not None:
                parts.append(f"RESULT_TARGET_FILE_TIMEOUT:\n")
                parts.append(json.dumps(result_target_file_timeout, ensure_ascii=False))
                parts.append("\n\n")
            if result_target_file_running_successfully is not None:
                parts.append(f"RESULT_TARGET_FILE_RUNNING_SUCCESSFULLY:\n")
                parts.append(json.dumps(result_target_file_running_successfully, ensure_ascii=False))
                parts.append("\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            user_prompt = "".join(parts)

            response, _ = self._send_chat(user_model=user_model, user_prompt=user_prompt)
            response = self._parse_json(response)
            # Make sure the stderr saved in the log is the exact stderr returned
            # by run_program.run_orca_file (tail‑truncated there).
            if isinstance(response, dict):
                response["stderr"] = result_target_file_stderr
            judge_results.append(response)

        # return aggregate result dict-----------------------------------------
        output = {
            "Content": "Single repocode executability judgement",
            "LLM": user_model,
            "repo_dir": current_sandbox_dir,
            "judge_result": judge_results,
        }
        self._log_event(output)
        return output
    

    async def Verify_code_single_repocode_execution_concurr(self, user_model: str, current_sandbox_dir: str = None, upper_num: int = 0, concurr_num: int = 5, exe_range: List[str] = None, islog: bool = True, use_mcp_run_orca: bool = False, mcp_orca_timeout: int = 60, iscaltoken: bool = False) ->dict:
        """
        Execute all *_unittest files in sandbox directory and evaluate results.
        
        Workflow:
        1. Find all *_unittest.{py,jl,inp} files in sandbox directory
        2. For each file (concurrently):
           - Execute file:
             * .py → run_program.run_python_file
             * .jl → run_program.run_julia_file
             * .inp → run_program.run_orca_file (or MCP if use_mcp_run_orca=True)
           - Collect execution results (exitcode, stdout, stderr, timeout, running_successfully)
           - Evaluate using VerifyCode_sandbox_single_repocode_execution_evaluation_stand1 prompt
           - Return judge result
        
        Returns:
            dict with "judge_result" list and optionally "token_stats" when iscaltoken=True
        """

        # find all .py/.jl/.inp files ending with "_unittest" in the sandbox directory--------------------------------
        files_list = []
        if current_sandbox_dir is None:
            raise ValueError("current_sandbox_dir is None.")
        sandbox_path = Path(current_sandbox_dir)
        if not sandbox_path.exists():
            raise FileNotFoundError(f"sandbox directory '{sandbox_path}' not found.")

        for f in sandbox_path.iterdir():
            if not f.is_file():
                continue
            # Include .py, .jl, and .inp files (ORCA input files)
            if f.suffix.lower() in (".py", ".jl", ".inp"):
                # Only include files ending with "_unittest.{ext}"
                if f.stem.endswith("_unittest"):
                    # If exe_range is specified, only include files matching the pattern verify_{exe_range[i]}_unittest
                    if exe_range is not None:
                        # Extract function name from filename: verify_{function_name}_unittest
                        function_name = f.stem.replace("verify_", "").replace("_unittest", "")
                        if function_name in exe_range:
                            files_list.append(f.name)
                    else:
                        # If exe_range is None, include all _unittest files
                        files_list.append(f.name)

        files_list = sorted(files_list)
        # print(f"Sandbox files (*_unittest.py/jl/inp): {files_list}") # for debug

        # Iterative execution (concurrent)-------------------------------------------------
        # Prepare list of targets respecting upper_num
        targets = files_list[:upper_num] if (upper_num > 0) else files_list

        sem = asyncio.Semaphore(concurr_num)

        async def _process_target(target_file_name: str):
            async with sem:
                target_file_path = sandbox_path / target_file_name
                target_file = self._load_file(target_file_path)
                print(f"verify code: {target_file_name} executing...")

                file_ext = target_file_path.suffix.lower()
                if file_ext == ".inp" and use_mcp_run_orca:
                    # MCP run_orca + evaluate (short timeout, early terminate)
                    role_prompt = self._load_prompt(topic="general_dft", prompt_file="VerifyCode_sandbox_single_repocode_execution_mcp_run_orca_stand1")
                    parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                    parts.append(f"TARGET_FILE_NAME:\n")
                    parts.append(json.dumps(target_file_name, ensure_ascii=False))
                    parts.append("\n\n")
                    parts.append(f"TARGET_FILE_CONTENT:\n")
                    parts.append(json.dumps(target_file, ensure_ascii=False))
                    parts.append("\n\n")
                    parts.append(f"ORCA_TIMEOUT_SECONDS:\n")
                    parts.append(json.dumps(mcp_orca_timeout, ensure_ascii=False))
                    parts.append("\n\n-- END AGGREGATED INPUT --\n")
                    user_prompt = "".join(parts)
                    response = await MCP_toolbox.send_chat_through_mcp_dynamic(
                        user_model=user_model,
                        user_prompt=user_prompt,
                        fs_target_dir=str(sandbox_path),
                        init_servers=["program-dft-mcp", "filesystem-mcp"],
                        program_dft_input_path=str(target_file_path),
                        program_dft_timeout=mcp_orca_timeout,
                    )
                    if not isinstance(response, str):
                        response = self._normalize_mcp_response_to_str(response)
                    response = self._parse_json(response)
                    return (response, None) if iscaltoken else response

                # Select run function based on file extension (Python/Julia or .inp without MCP)
                if file_ext == ".py":
                    run_func = run_program.run_python_file
                elif file_ext == ".jl":
                    run_func = run_program.run_julia_file
                elif file_ext == ".inp":
                    run_func = run_program.run_orca_file
                else:
                    raise ValueError(f"Unsupported file extension: {file_ext}")
                
                # run function is blocking; run in thread
                result_target_file = await asyncio.to_thread(run_func, str(target_file_path), str(sandbox_path), 600)
                result_target_file_exitcode = result_target_file.get("exitcode")
                result_target_file_stdout = result_target_file.get("stdout", "")
                result_target_file_stderr = result_target_file.get("stderr", "")
                result_target_file_timeout = result_target_file.get("timeout", False)
                result_target_file_running_successfully = result_target_file.get("running_successfully")

                # Evaluate execution result by LLM--------------------------------
                role_prompt = self._load_prompt(topic="general_dft", prompt_file="VerifyCode_sandbox_single_repocode_execution_evaluation_stand1")
                parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                if target_file_name is not None:
                    parts.append(f"TARGET_FILE_NAME:\n")
                    parts.append(json.dumps(target_file_name, ensure_ascii=False))
                    parts.append("\n\n")
                if target_file is not None:
                    parts.append(f"TARGET_FILE_CONTENT:\n")
                    parts.append(json.dumps(target_file, ensure_ascii=False))
                    parts.append("\n\n")
                if result_target_file_exitcode is not None:
                    parts.append(f"RESULT_TARGET_FILE_EXITCODE:\n")
                    parts.append(json.dumps(result_target_file_exitcode, ensure_ascii=False))
                    parts.append("\n\n")
                # Include stdout for ORCA execution - contains full output file content
                if result_target_file_stdout is not None:
                    parts.append(f"RESULT_TARGET_FILE_STDOUT:\n")
                    stdout_truncated = result_target_file_stdout
                    if len(stdout_truncated) > 5000:
                        truncated_chars = len(result_target_file_stdout) - 5000
                        stdout_truncated = f"... (truncated: {truncated_chars} characters from beginning) ...\n\n" + stdout_truncated[-5000:]
                    parts.append(json.dumps(stdout_truncated, ensure_ascii=False))
                    parts.append("\n\n")
                if result_target_file_stderr is not None:
                    parts.append(f"RESULT_TARGET_FILE_STDERR:\n")
                    parts.append(json.dumps(result_target_file_stderr, ensure_ascii=False))
                    parts.append("\n\n")
                if result_target_file_timeout is not None:
                    parts.append(f"RESULT_TARGET_FILE_TIMEOUT:\n")
                    parts.append(json.dumps(result_target_file_timeout, ensure_ascii=False))
                    parts.append("\n\n")
                if result_target_file_running_successfully is not None:
                    parts.append(f"RESULT_TARGET_FILE_RUNNING_SUCCESSFULLY:\n")
                    parts.append(json.dumps(result_target_file_running_successfully, ensure_ascii=False))
                    parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)

                if iscaltoken:
                    response, token_dict = await asyncio.to_thread(self._send_chat_robust, user_model, user_prompt, max_iter=3, iscaltoken=True)
                else:
                    response = await asyncio.to_thread(self._send_chat_robust, user_model, user_prompt, max_iter=3)
                    token_dict = None
                response = self._parse_json(response)
                # Ensure that stderr in the logged judge_result matches the stderr
                # returned by run_program.run_orca_file.
                if isinstance(response, dict):
                    response["stderr"] = result_target_file_stderr
                return (response, token_dict) if iscaltoken else response

        # schedule workers and wait
        tasks = [asyncio.create_task(_process_target(name)) for name in targets]
        judge_results = []
        total_input = 0
        total_output = 0
        if tasks:
            results = await asyncio.gather(*tasks)
            if iscaltoken:
                for r in results:
                    if isinstance(r, tuple) and len(r) == 2:
                        resp, token_dict = r
                        judge_results.append(resp)
                        if token_dict:
                            total_input += token_dict.get("input_tokens", 0)
                            total_output += token_dict.get("output_tokens", 0)
                    else:
                        judge_results.append(r)
            else:
                judge_results.extend(results)

        token_stats = (
            {"input_tokens": total_input, "output_tokens": total_output, "total": total_input + total_output}
            if iscaltoken and (total_input + total_output) > 0
            else None
        )
        output = {
            "Content": "Single repocode executability judgement",
            "LLM": user_model,
            "repo_dir": current_sandbox_dir,
            "judge_result": judge_results,
        }
        if token_stats:
            output["token_stats"] = token_stats
        if islog:
            self._log_event(output)
        return output


    async def Verify_code_single_repocode_split_refine_concurr(self, user_model: str, current_sandbox_dir: str = None, upper_num: int = 0, concurr_num: int = 5, iscaltoken: bool = False) ->dict:
        # Step 1: Read former verify code execution results, focus on the 'undetermined' results
        if self.current_log_file is None:
            raise ValueError("current_log_file is None. Cannot read execution results.")
        log_file_path = Path(self.current_log_file)
        if not log_file_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_file_path}")
        
        # Read and parse the jsonl file
        last_judgement_entry = None
        with open(log_file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Split by double newlines to get individual JSON objects
            entries = content.split("\n\n")
            for entry_str in entries:
                entry_str = entry_str.strip()
                if not entry_str:
                    continue
                try:
                    entry = json.loads(entry_str, strict=False)
                    if entry.get("Content") == "Single repocode executability judgement":
                        last_judgement_entry = entry
                except json.JSONDecodeError:
                    continue
        if last_judgement_entry is None:
            print("No 'Single repocode executability judgement' entry found in log file.")
            raise ValueError("No 'Single repocode executability judgement' entry found in log file.")
        
        # Extract undetermined target_file_name from judge_result, along with error and stderr
        judge_results = last_judgement_entry.get("judge_result", [])
        undetermined_files = []
        for result in judge_results:
            if result.get("judge") != "correct" and result.get("judge") != "wrong":
                target_file_name = result.get("target_file_name")
                if target_file_name:
                    undetermined_files.append({
                        "target_file_name": target_file_name,
                        "errors": result.get("errors", ""),
                        "stderr": result.get("stderr", "")
                    })
        print(f"Found {len(undetermined_files)} undetermined files: {[f['target_file_name'] for f in undetermined_files]}")

        # Step 2: Define and wait for the concurrent tasks for refining the undetermined results
        if not undetermined_files:
            print("No undetermined files to refine.")
            return {"Content": "Split refine - no undetermined files", "refined_files": []}
        if current_sandbox_dir is None:
            raise ValueError("current_sandbox_dir is None. Cannot refine code.")
        
        sandbox_path = Path(current_sandbox_dir)
        if not sandbox_path.exists():
            raise FileNotFoundError(f"Sandbox directory not found: {sandbox_path}")
        
        # Filter to targets that have at least background + tobetest.
        # For dft_qc / .inp: entrypoint is optional (ORCA input is concatenated segments).
        # For other topics: require background + tobetest + entrypoint.
        refinable_files = []
        for file_info in undetermined_files:
            target_file_name = file_info["target_file_name"]
            file_ext = Path(target_file_name).suffix
            function_name_nosuffix = target_file_name.replace("_unittest" + file_ext, "").replace("verify_", "")
            bg = sandbox_path / f"verify_{function_name_nosuffix}_background{file_ext}"
            tb = sandbox_path / f"verify_{function_name_nosuffix}_tobetest{file_ext}"
            ep = sandbox_path / f"verify_{function_name_nosuffix}_entrypoint{file_ext}"
            is_dft_orca = (self.topic == "dft_qc" or file_ext.lower() == ".inp")
            if bg.exists() and tb.exists():
                if is_dft_orca or ep.exists():
                    refinable_files.append(file_info)
                else:
                    missing = [ep.name]
                    print(f"Skipping refine for {target_file_name}: missing split files {missing}")
                    self._log_event({
                        "Content": "Split refine skip - missing split files",
                        "target_file_name": target_file_name,
                        "missing": missing,
                    })
            else:
                missing = [p.name for p in (bg, tb, ep) if not p.exists()]
                print(f"Skipping refine for {target_file_name}: missing split files {missing}")
                self._log_event({
                    "Content": "Split refine skip - missing split files",
                    "target_file_name": target_file_name,
                    "missing": missing,
                })
        if not refinable_files:
            print("No undetermined files with split structure (background/tobetest, and entrypoint for non-DFT) to refine.")
            return {"Content": "Split refine - no undetermined files with split structure", "refined_files": [], "skipped": len(undetermined_files)}
        
        # Prepare list of targets respecting upper_num
        targets = refinable_files[:upper_num] if (upper_num > 0) else refinable_files
        
        sem = asyncio.Semaphore(concurr_num)
        
        async def _process_refine_target(file_info: dict):
            async with sem:
                target_file_name = file_info["target_file_name"]
                error_analysis = file_info.get("errors", "")
                stderr_info = file_info.get("stderr", "")
                
                # Extract function name and file extension
                # target_file_name format: verify_{function_name}_unittest.{ext}
                # We need to extract {function_name}
                file_ext = Path(target_file_name).suffix
                function_name_nosuffix = target_file_name.replace("_unittest" + file_ext, "").replace("verify_", "")
                
                # Load background, tobetest, and optionally entrypoint
                background_file = sandbox_path / f"verify_{function_name_nosuffix}_background{file_ext}"
                tobetest_file = sandbox_path / f"verify_{function_name_nosuffix}_tobetest{file_ext}"
                entrypoint_file = sandbox_path / f"verify_{function_name_nosuffix}_entrypoint{file_ext}"
                is_dft_orca = (self.topic == "dft_qc" or file_ext.lower() == ".inp")
                
                if not background_file.exists() or not tobetest_file.exists():
                    missing_files = [str(p) for p in (background_file, tobetest_file) if not p.exists()]
                    error_msg = f"Missing files for {function_name_nosuffix}: {', '.join(missing_files)}"
                    print(f"Skipping refine for {target_file_name}: {error_msg}")
                    self._log_event({
                        "Content": "Refine verify code skipped - missing files (should have been filtered)",
                        "function_name": function_name_nosuffix,
                        "target_file_name": target_file_name,
                        "missing_files": missing_files
                    })
                    return None
                
                background_code = self._load_file(background_file)
                tobetest_code = self._load_file(tobetest_file)
                entrypoint_code_content = self._load_file(entrypoint_file) if entrypoint_file.exists() else ""
                
                # Build user prompt: for ORCA/DFT, entrypoint is optional (no separate entry point)
                role_prompt = self._load_prompt(topic="general_dft", prompt_file="VerifyCode_sandbox_single_repocode_refine_stand1")
                parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                parts.append("BACKGROUND_CODE:\n")
                parts.append(background_code)
                parts.append("\n\n")
                parts.append("TOBETEST_CODE:\n")
                parts.append(tobetest_code)
                parts.append("\n\n")
                if is_dft_orca and not entrypoint_code_content:
                    parts.append("ENTRYPOINT_CODE: N/A (ORCA/DFT input; no separate entry point.)\n\n")
                else:
                    parts.append("ENTRYPOINT_CODE:\n")
                    parts.append(entrypoint_code_content)
                    parts.append("\n\n")
                parts.append("ERROR_ANALYSIS:\n")
                parts.append(error_analysis)
                parts.append("\n\n")
                parts.append("STDERR:\n")
                parts.append(stderr_info)
                parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)
                
                print(f"refining code: {target_file_name}...")
                
                # For ORCA/DFT, entrypoint is optional; require at least BACKGROUND
                if iscaltoken:
                    response, token_dict = await asyncio.to_thread(self._send_chat_robust, user_model, user_prompt, max_iter=3, iscaltoken=True, require_code_block=True, code_tags=["BACKGROUND"])
                else:
                    response = await asyncio.to_thread(self._send_chat_robust, user_model, user_prompt, max_iter=3, require_code_block=True, code_tags=["BACKGROUND"])
                    token_dict = None
                parsed_response = self._parse_json(response)
                if not isinstance(parsed_response, dict):
                    parsed_response = {}
                refine_summary = parsed_response.get("refine_summary", "")
                
                # Save refined background only if present; save entrypoint only if non-empty
                if self._extract_code_from_fence(response, "BACKGROUND") is not None:
                    await asyncio.to_thread(
                        self._save_code_file,
                        response,
                        anci=f"verify_{function_name_nosuffix}_background",
                        code_dir=current_sandbox_dir,
                        isarbi=True,
                        content_tag="BACKGROUND",
                        file_ext=file_ext,
                    )
                if self._extract_code_from_fence(response, "ENTRYPOINT") is not None:
                    await asyncio.to_thread(
                        self._save_code_file,
                        response,
                        anci=f"verify_{function_name_nosuffix}_entrypoint",
                        code_dir=current_sandbox_dir,
                        isarbi=True,
                        content_tag="ENTRYPOINT",
                        file_ext=file_ext,
                    )
                
                # Re-merge: background + tobetest + entrypoint (only if entrypoint file exists)
                try:
                    merged_content_parts = []
                    merged_content_parts.append(self._load_file(background_file))
                    merged_content_parts.append("\n\n")
                    merged_content_parts.append(self._load_file(tobetest_file))
                    if entrypoint_file.exists():
                        merged_content_parts.append("\n\n")
                        merged_content_parts.append(self._load_file(entrypoint_file))
                    merged_content = "".join(merged_content_parts)
                    merged_file = sandbox_path / f"verify_{function_name_nosuffix}_unittest{file_ext}"
                    merged_file.write_text(merged_content, encoding='utf-8')
                except Exception as e:
                    error_msg = f"Failed to merge refined files for {function_name_nosuffix}: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    self._log_event({
                        "Content": "Refine verify code failed - merge error",
                        "function_name": function_name_nosuffix,
                        "error": str(e)
                    })
                
                # Return result dictionary similar to Verify_code_single_repocode_execution_concurr
                result_dict = {
                    "target_file_name": target_file_name,
                    "refine_summary": refine_summary
                }
                if iscaltoken:
                    return result_dict, token_dict
                return result_dict
        
        # Schedule workers and wait
        tasks = [asyncio.create_task(_process_refine_target(file_info)) for file_info in targets]
        refine_results = []
        refine_token_total_input = 0
        refine_token_total_output = 0
        if tasks:
            results = await asyncio.gather(*tasks)
            if iscaltoken:
                for r in results:
                    if isinstance(r, tuple) and len(r) == 2:
                        result_dict, token_dict = r
                        if result_dict is not None:
                            refine_results.append(result_dict)
                        if token_dict:
                            refine_token_total_input += token_dict.get("input_tokens", 0)
                            refine_token_total_output += token_dict.get("output_tokens", 0)
                    elif r is not None:
                        refine_results.append(r)
            else:
                refine_results = [r for r in results if r is not None]
        print(f"Refined {len(refine_results)} unit test files")
        
        # Step 3: Execute the refined unit test code and evaluate the results
        if not refine_results:
            # All refine attempts failed or were skipped - return early instead of raising
            print("No files were successfully refined (all were skipped or failed).")
            return {"Content": "Split refine - no files successfully refined", "refined_files": [], "attempted": len(targets)}
        
        # Extract target_file_name from refine_results to build exe_range
        # Also create a mapping from target_file_name to refine_summary
        exe_range = []
        refine_summary_map = {}
        for refine_result in refine_results:
            target_file_name = refine_result.get("target_file_name")
            if target_file_name:
                # Extract function name from target_file_name: verify_{function_name}_unittest.{ext}
                file_ext = Path(target_file_name).suffix
                function_name = target_file_name.replace("_unittest" + file_ext, "").replace("verify_", "")
                exe_range.append(function_name)
                refine_summary_map[target_file_name] = refine_result.get("refine_summary", "")
        
        # Call Verify_code_single_repocode_execution_concurr with exe_range
        execution_output = await self.Verify_code_single_repocode_execution_concurr(
            user_model=user_model,
            current_sandbox_dir=current_sandbox_dir,
            upper_num=0,  # Execute all files in exe_range
            concurr_num=concurr_num,
            exe_range=exe_range,
            islog=False,
            iscaltoken=iscaltoken,
        )
        
        # Modify the output: change Content and add refine_summary to each judge_result entry
        execution_output["Content"] = "Single repocode unit test environment refinement judgement"
        judge_results = execution_output.get("judge_result", [])
        for judge_result in judge_results:
            target_file_name = judge_result.get("target_file_name")
            if target_file_name and target_file_name in refine_summary_map:
                judge_result["refine_summary"] = refine_summary_map[target_file_name]
        
        # Aggregate token statistics from refine and execution
        if iscaltoken:
            execution_token = execution_output.get("token_stats") or {}
            total_input = refine_token_total_input + execution_token.get("input_tokens", 0)
            total_output = refine_token_total_output + execution_token.get("output_tokens", 0)
            if total_input + total_output > 0:
                execution_output["token_stats"] = {
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "total": total_input + total_output,
                }
        
        self._log_event(execution_output)
        return execution_output

    def Verify_code_generate_report(self, user_model: str, judge_token_stats: dict = None) -> dict:
        """
        Generate a compact report from log_unittest_xxx.jsonl file.
        Combines executability judgement and refinement judgement results.
        Appends a "Generate unittest report" event to the log with summary statistics.
        """
        if self.current_log_file is None:
            raise ValueError("current_log_file is None. Cannot read execution results.")
        log_file_path = Path(self.current_log_file)
        if not log_file_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_file_path}")

        last_executability_entry = None
        last_refinement_entry = None
        with open(log_file_path, "r", encoding="utf-8") as f:
            content = f.read()
            entries = content.split("\n\n")
            for entry_str in entries:
                entry_str = entry_str.strip()
                if not entry_str:
                    continue
                try:
                    entry = json.loads(entry_str, strict=False)
                    content_type = entry.get("Content")
                    if content_type == "Single repocode executability judgement":
                        last_executability_entry = entry
                    elif content_type == "Single repocode unit test environment refinement judgement":
                        last_refinement_entry = entry
                except json.JSONDecodeError:
                    continue

        if last_executability_entry is None:
            raise ValueError("No 'Single repocode executability judgement' entry found in log file.")

        executability_judge_result = last_executability_entry.get("judge_result", [])
        refinement_judge_result = last_refinement_entry.get("judge_result", []) if last_refinement_entry else []

        merged_dict = {}
        for result in executability_judge_result:
            target_file_name = result.get("target_file_name")
            if target_file_name:
                merged_dict[target_file_name] = {
                    "target_file_name": target_file_name,
                    "judge": result.get("judge", ""),
                    "errors": result.get("errors", "")
                }
        for result in refinement_judge_result:
            target_file_name = result.get("target_file_name")
            if target_file_name:
                merged_dict[target_file_name] = {
                    "target_file_name": target_file_name,
                    "judge": result.get("judge", ""),
                    "errors": result.get("errors", "")
                }

        merged_judge_result = list(merged_dict.values())
        total_count = len(merged_judge_result)
        correct_count = sum(1 for result in merged_judge_result if result.get("judge") == "correct")
        undetermined_count = sum(1 for result in merged_judge_result if result.get("judge") == "undetermined")
        correct_ratio = (correct_count / total_count) if total_count > 0 else 0.0
        undetermined_ratio = (undetermined_count / total_count) if total_count > 0 else 0.0

        report_file_name = log_file_path.name.replace("log_unittest", "report_unittest")
        report_file_path = log_file_path.parent / report_file_name
        report_dict = {
            "judge_result": merged_judge_result,
            "correct_ratio": correct_ratio,
            "undetermined_ratio": undetermined_ratio
        }
        if judge_token_stats:
            report_dict["judge_token_stats"] = judge_token_stats
        pretty_json = json.dumps(report_dict, ensure_ascii=False, indent=2)
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write(pretty_json + "\n\n")

        output = {
            "Content": "Generate unittest report",
            "LLM": user_model,
            "report_file": str(report_file_path),
            "total_functions": total_count,
            "correct_count": correct_count,
            "correct_ratio": correct_ratio,
            "undetermined_count": undetermined_count,
            "undetermined_ratio": undetermined_ratio
        }
        self._log_event(output)
        return output

    
class CodeIntegrator(DFT_agent):

    async def Generate_integration_guidelines(self, user_model: str, current_sandbox_dir: str = None, repo_dir: str = None, upper_num: int = 0) ->dict:
       
        # Extract information from the target repo----------------------------------------
        if self.topic == "dmrg":
            prefix = "run"
        elif self.topic == "nnwf":
            prefix = "run"
        elif self.topic == "qcmb":
            prefix = "run"
        elif self.topic == "dft_qc":
            prefix = "run"
        else:
            prefix = "run"
        func_list = []
        repo_path = Path(repo_dir)
        for file in repo_path.iterdir():
            if not file.is_file():
                continue
            if file.name.startswith(prefix):
                func_name = file.stem  
                func_list.append(f"{func_name}.{file.suffix.lstrip('.')}")
        # print(f"Extracted functions: {func_list}") # for debug

        # Generate guidelines for each run function------------------------------------------------------------
        for i in range(len(func_list)):
            if upper_num >0 and i >= upper_num:
                break

            target_function_name = func_list[i]
            target_function = self._load_file(Path(repo_dir) / target_function_name)
            # print(target_function)  # for debug
            print(f"{target_function_name} generating integration guidelines...")

            # role_prompt = self._load_prompt("IntegrateCode_sandbox_generate_guidelines_stand1")
            role_prompt = self._load_prompt(topic=self.topic, prompt_file="IntegrateCode_sandbox_generate_guidelines_details_stand1")
            parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            if target_function is not None:
                parts.append(f"RUN_FUNCTION:\n")
                parts.append(target_function)
                parts.append("\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            user_prompt = "".join(parts)
            # response = self._send_chat(user_model=user_model, user_prompt=user_prompt)
            response = self._send_chat_robust(user_model=user_model, user_prompt=user_prompt, max_iter=3)
            
            function_name_nosuffix = target_function_name.rsplit('.', 1)[0]
            save_jsonl_output = self._save_jsonl_file(response, anci=f"guideline_integrate_{function_name_nosuffix}", jsonl_dir=current_sandbox_dir, isarbi=True, issub=True, subname=function_name_nosuffix)
            

        output = {"Content": "Single run function integration -- guidelines generation", "LLM": user_model, "repo_dir": repo_dir, "sandbox_dir": current_sandbox_dir}
        self._log_event(output)

        return output


    async def Generate_integration_code(self, user_model: str, current_sandbox_dir: str, repo_dir: str, upper_num: int = 0) -> dict:
        
        repo_path = Path(repo_dir)
        # find all subdirectories in the sandbox directory (different run functions)--------------------------------
        run_function_list = []
        if current_sandbox_dir is None:
            raise ValueError("current_sandbox_dir is None.")
        sandbox_path = Path(current_sandbox_dir)
        if not sandbox_path.exists():
            raise FileNotFoundError(f"sandbox directory '{sandbox_path}' not found.")

        for d in sandbox_path.iterdir():
            if d.is_dir():
                run_function_list.append(d.name)

        run_function_list = sorted(run_function_list)
        print(f"Sandbox subdirectories: {run_function_list}") # for debug

        # Iterative code generation for integration code------------------------------------------------------------
        for i in range(len(run_function_list)):
            target_run_name = run_function_list[i]
            guideline_jsonl_path = sandbox_path / target_run_name / f"integrate_{target_run_name}.jsonl"
            guideline_jsonl = self._load_file(guideline_jsonl_path)
            guideline_obj = json.loads(guideline_jsonl)
            if guideline_obj is None:
                self._log_event({"Content": "Failed to parse guideline JSON", "path": str(guideline_jsonl_path)})
                continue

            run_function_name = guideline_obj.get('run_function')
            integration_guideline = guideline_obj.get('integration_guideline')
            run_function_content = None
            main_content = None
            # Try to load run_function file (py or jl)
            for ext in ['.py', '.jl']:
                run_func_path = repo_path / f"{run_function_name}{ext}"
                if run_func_path.exists():
                    run_function_content = self._load_file(run_func_path)
                    break
            # Try to load main file (py or jl)
            for ext in ['.py', '.jl']:
                main_path = repo_path / f"main{ext}"
                if main_path.exists():
                    main_content = self._load_file(main_path)
                    break

            # iterate over each level in integration_guideline if it's a mapping--------------
            # DO not use mcp--------compatable with guidelines generated by 'IntegrateCode_sandbox_generate_guidelines_details_stand1' prompt
            level_count = 0
            for level_dict in integration_guideline:
                if upper_num > 0 and level_count >= upper_num:
                    break

                # load all required element functions                
                element_functions_names = level_dict.get('element_functions')
                element_functions = [{} for _ in range(len(element_functions_names))]
                # print(f"element_functions: {element_functions}") # for debug
                for idx, func_name in enumerate(element_functions_names):
                    func_content = None
                    for ext in ['.py', '.jl']:
                        func_path = repo_path / f"{func_name}{ext}"
                        if func_path.exists():
                            func_content = self._load_file(func_path)
                            break
                    element_functions[idx] = {"name": func_name, "content": func_content}
                # print(f"{element_functions}")  # for debug
                
                role_prompt = self._load_prompt("IntegrateCode_sandbox_generate_integration_code_details_stand1")
                parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                if run_function_content is not None:
                    parts.append(f"RUN_FUNCTION_CONTENT:\n")
                    parts.append(json.dumps(run_function_content, ensure_ascii=False))
                    parts.append("\n\n")
                if level_dict is not None:
                    parts.append(f"LEVEL_GUIDELINE:\n")
                    parts.append(json.dumps(level_dict, ensure_ascii=False))
                    parts.append("\n\n")
                if element_functions is not None:
                    parts.append(f"ELEMENT_FUNCTIONS:\n")
                    parts.append(json.dumps(element_functions, ensure_ascii=False))
                    parts.append("\n\n")
                if main_content is not None:
                    parts.append(f"MAIN_CONTENT:\n")
                    parts.append(json.dumps(main_content, ensure_ascii=False))
                    parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)
                # print(f"{user_prompt}")  # for debug 

                response, _ = self._send_chat(user_model=user_model, user_prompt=user_prompt)
                save_code_output = self._save_code_file(response, anci=f"integrate_{target_run_name}_level{level_count+1}", code_dir=sandbox_path / target_run_name, isarbi=True, content_tag="CODE")
                level_count += 1


            # using filesystem-mcp--------compatable with guidelines generated by 'IntegrateCode_sandbox_generate_guidelines_stand1' prompt
            

        output = {"Content": "Single run function integration -- integration code generation", "LLM": user_model, "repo_dir": repo_dir, "sandbox_dir": current_sandbox_dir}
        self._log_event(output)

        return output
    

    async def Generate_integration_code_concurr(self, user_model: str, current_sandbox_dir: str, repo_dir: str, upper_num: int = 0, concurr_num: int = 5) -> dict:
        
        repo_path = Path(repo_dir)
        # find all subdirectories in the sandbox directory (different run functions)--------------------------------
        run_function_list = []
        if current_sandbox_dir is None:
            raise ValueError("current_sandbox_dir is None.")
        sandbox_path = Path(current_sandbox_dir)
        if not sandbox_path.exists():
            raise FileNotFoundError(f"sandbox directory '{sandbox_path}' not found.")

        for d in sandbox_path.iterdir():
            if d.is_dir():
                run_function_list.append(d.name)

        run_function_list = sorted(run_function_list)
        print(f"Sandbox subdirectories: {run_function_list}") # for debug

        # Iterative code generation for integration code------------------------------------------------------------
        for i in range(len(run_function_list)):
            target_run_name = run_function_list[i]
            guideline_jsonl_path = sandbox_path / target_run_name / f"guideline_integrate_{target_run_name}.jsonl"
            guideline_jsonl = self._load_file(guideline_jsonl_path)
            guideline_obj = json.loads(guideline_jsonl)
            if guideline_obj is None:
                self._log_event({"Content": "Failed to parse guideline JSON", "path": str(guideline_jsonl_path)})
                continue

            run_function_name = guideline_obj.get('run_function')
            integration_guideline = guideline_obj.get('integration_guideline')
            run_function_content = None
            main_content = None
            # Try to load run_function file (py or jl)
            for ext in ['.py', '.jl']:
                run_func_path = repo_path / f"{run_function_name}{ext}"
                if run_func_path.exists():
                    run_function_content = self._load_file(run_func_path)
                    break
            # Try to load main file (py or jl)
            for ext in ['.py', '.jl']:
                main_path = repo_path / f"main{ext}"
                if main_path.exists():
                    main_content = self._load_file(main_path)
                    break

            # iterate over each level in integration_guideline if it's a mapping--------------
            # DO not use mcp--------compatable with guidelines generated by 'IntegrateCode_sandbox_generate_guidelines_details_stand1' prompt
            level_dicts = integration_guideline[:upper_num] if upper_num > 0 else integration_guideline

            sem = asyncio.Semaphore(concurr_num)

            async def _process_level(level_dict: dict):
                async with sem:
                    # load all required element functions      
                    level_index = [key for key in level_dict if key.startswith('level')][0]
                    level_index.replace('-', '')
                    element_functions_names = level_dict.get('element_functions')
                    element_functions = [{} for _ in range(len(element_functions_names))]
                    print(f"{level_index} of {target_run_name} integtest code generating...")  
                    
                    for idx, func_name in enumerate(element_functions_names):
                        func_content = None
                        for ext in ['.py', '.jl']:
                            func_path = repo_path / f"{func_name}{ext}"
                            if func_path.exists():
                                func_content = self._load_file(func_path)
                                break
                        element_functions[idx] = {"name": func_name, "content": func_content}
                    # print(f"{element_functions}")  # for debug
                    
                    role_prompt = self._load_prompt(topic=self.topic, prompt_file="IntegrateCode_sandbox_generate_integration_code_details_stand1")
                    parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                    if run_function_content is not None:
                        parts.append(f"RUN_FUNCTION_CONTENT:\n")
                        parts.append(json.dumps(run_function_content, ensure_ascii=False))
                        parts.append("\n\n")
                    if level_dict is not None:
                        parts.append(f"LEVEL_GUIDELINE:\n")
                        parts.append(json.dumps(level_dict, ensure_ascii=False))
                        parts.append("\n\n")
                    if element_functions is not None:
                        parts.append(f"ELEMENT_FUNCTIONS:\n")
                        parts.append(json.dumps(element_functions, ensure_ascii=False))
                        parts.append("\n\n")
                    if main_content is not None:
                        parts.append(f"MAIN_CONTENT:\n")
                        parts.append(json.dumps(main_content, ensure_ascii=False))
                        parts.append("\n\n")
                    parts.append("-- END AGGREGATED INPUT --\n")
                    user_prompt = "".join(parts)
                    # print(f"{user_prompt}")  # for debug 

                    response = await asyncio.to_thread(self._send_chat, user_model=user_model, user_prompt=user_prompt)
                    save_code_output = await asyncio.to_thread(self._save_code_file, response, anci=f"integrate_{target_run_name}_{level_index}", code_dir=sandbox_path / target_run_name, isarbi=True, content_tag="CODE")
                    return save_code_output

            tasks = [asyncio.create_task(_process_level(ld)) for ld in level_dicts]
            if tasks:
                await asyncio.gather(*tasks) 
            # end of concurrency for levels under 1 run function          

        output = {"Content": "Single run function integration -- integration code generation", "LLM": user_model, "repo_dir": repo_dir, "sandbox_dir": current_sandbox_dir}
        self._log_event(output)
        return output


    async def Generate_integration_code_split_concurr(self, user_model: str, current_sandbox_dir: str = None, repo_dir: str = None, upper_num: int = 0, concurr_num: int = 5) -> dict:
        
        # Step 1: Prepare the enviroment
        repo_path = Path(repo_dir)
        run_function_list = []
        if current_sandbox_dir is None:
            raise ValueError("current_sandbox_dir is None.")
        sandbox_path = Path(current_sandbox_dir)
        if not sandbox_path.exists():
            raise FileNotFoundError(f"sandbox directory '{sandbox_path}' not found.")

        for d in sandbox_path.iterdir():
            if d.is_dir():
                run_function_list.append(d.name)
        run_function_list = sorted(run_function_list)
        print(f"Sandbox subdirectories: {run_function_list}") # for debug

        # Step 2: Generate integration code for each run function
        for i in range(len(run_function_list)):
            # Step 2.1: Load the guideline JSONL file
            target_run_name = run_function_list[i]
            guideline_jsonl_path = sandbox_path / target_run_name / f"guideline_integrate_{target_run_name}.jsonl"
            guideline_jsonl = self._load_file(guideline_jsonl_path)
            guideline_obj = json.loads(guideline_jsonl, strict=False)
            if guideline_obj is None:
                raise ValueError(f"Failed to parse guideline JSON from {guideline_jsonl_path}")

            # Step 2.2: Load the run function and main file
            run_function_name = guideline_obj.get('run_function')
            integration_guideline = guideline_obj.get('integration_guideline')
            run_function_content = None
            main_content = None
            # Try to load run_function file (py or jl)
            for ext in ['.py', '.jl']:
                run_func_path = repo_path / f"{run_function_name}{ext}"
                if run_func_path.exists():
                    run_function_content = self._load_file(run_func_path)
                    break
            # Try to load main file (py or jl)
            for ext in ['.py', '.jl']:
                main_path = repo_path / f"main{ext}"
                if main_path.exists():
                    main_content = self._load_file(main_path)
                    break

            # Step 2.3: Define the concurrency for each level
            level_dicts = integration_guideline[:upper_num] if upper_num > 0 else integration_guideline

            sem = asyncio.Semaphore(concurr_num)

            async def _process_level(level_dict: dict):
                async with sem:
                    # Step 2.3.1: Load the level dictionary
                    level_index = [key for key in level_dict if key.startswith('level')][0]
                    level_index.replace('-', '')
                    element_functions_names = level_dict.get('element_functions')
                    print(f"{level_index} of {target_run_name} integtest code generating...")  
                    
                    # Step 2.3.2: Prepare the element function code part (by hard code)
                    # Determine file extension from repo_dir
                    file_ext = None
                    if element_functions_names:
                        for ext in ['.py', '.jl']:
                            test_file = repo_path / f"{element_functions_names[0]}{ext}"
                            if test_file.exists():
                                file_ext = ext
                                break
                    if file_ext is None:
                        raise ValueError(f"Cannot determine file extension for element functions in {repo_path}")
                    
                    # Load and merge all element functions
                    elements_content_parts = []
                    for element_func_name in element_functions_names:
                        element_file_path = repo_path / f"{element_func_name}{file_ext}"
                        if element_file_path.exists():
                            element_content = self._load_file(element_file_path)
                            elements_content_parts.append(element_content)
                            elements_content_parts.append("\n\n")
                        else:
                            raise FileNotFoundError(f"Element function file not found: {element_file_path}")
                    
                    # Save merged element functions to sandbox directory
                    elements_file = sandbox_path / target_run_name / f"integrate_{target_run_name}_{level_index}_elements{file_ext}"
                    elements_file.parent.mkdir(parents=True, exist_ok=True)
                    elements_content = "".join(elements_content_parts).rstrip()
                    elements_file.write_text(elements_content, encoding='utf-8')

                    # Prepare the code file names
                    run_file = sandbox_path / target_run_name / f"integrate_{target_run_name}_{level_index}_run{file_ext}"
                    entrypoint_file = sandbox_path / target_run_name / f"integrate_{target_run_name}_{level_index}_entrypoint{file_ext}"
                    elements_file = sandbox_path / target_run_name / f"integrate_{target_run_name}_{level_index}_elements{file_ext}"

                    # Step 2.3.3: Generate the integration code part (run function and main function(entry point))
                    role_prompt = self._load_prompt(topic="general_dft", prompt_file="IntegrateCode_sandbox_single_level_environment_generation_stand1")
                    parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                    if level_dict is not None:
                        parts.append(f"LEVEL_GUIDELINE:\n")
                        parts.append(json.dumps(level_dict, ensure_ascii=False))
                        parts.append("\n\n")
                    if run_function_content is not None:
                        parts.append(f"RUN_FUNCTION_CONTENT:\n")
                        parts.append(run_function_content)
                        parts.append("\n\n")
                    if main_content is not None:
                        parts.append(f"MAIN_CONTENT:\n")
                        parts.append(main_content)
                        parts.append("\n\n")
                    parts.append("-- END AGGREGATED INPUT --\n")
                    user_prompt = "".join(parts)
                    response = await asyncio.to_thread(self._send_chat_robust, user_model=user_model, user_prompt=user_prompt, max_iter=3, require_code_block=True, code_tags=["RUNCODE", "ENTRYPOINT"])

                    # Step 2.3.4: Save the integration code parts and aggregate the complete integration code
                    await asyncio.to_thread(self._save_code_file, response, anci=f"integrate_{target_run_name}_{level_index}_run", code_dir=sandbox_path / target_run_name, isarbi=True, content_tag="RUNCODE", file_ext=file_ext)
                    await asyncio.to_thread(self._save_code_file, response, anci=f"integrate_{target_run_name}_{level_index}_entrypoint", code_dir=sandbox_path / target_run_name, isarbi=True, content_tag="ENTRYPOINT", file_ext=file_ext)

                    # Clean up the unused imports in run code and entrypoint code
                    code_editor_dft.delete_unused_imports(sandbox_path / target_run_name, count_type_annotations=True, file_names=[f"integrate_{target_run_name}_{level_index}_run{file_ext}", f"integrate_{target_run_name}_{level_index}_entrypoint{file_ext}"])

                    # Merge elements, run_code, and entrypoint_code into complete integration code
                    try:
                        merged_content_parts = []
                        merged_content_parts.append(self._load_file(elements_file))
                        merged_content_parts.append("\n\n")
                        merged_content_parts.append(self._load_file(run_file))
                        merged_content_parts.append("\n\n")
                        merged_content_parts.append(self._load_file(entrypoint_file))
                        merged_content = "".join(merged_content_parts)
                        merged_file = sandbox_path / target_run_name / f"integrate_{target_run_name}_{level_index}_integtest{file_ext}"
                        merged_file.write_text(merged_content, encoding='utf-8')
                    except Exception as e:
                        error_msg = f"Failed to merge refined files for {target_run_name}_{level_index}: {str(e)}"
                        print(f"[ERROR] {error_msg}")
                        self._log_event({
                            "Content": "Refine verify code failed - merge error",
                            "function_name": target_run_name,
                            "level_index": level_index,
                            "error": str(e)
                        })
                        
                    return target_run_name, level_index

            tasks = [asyncio.create_task(_process_level(ld)) for ld in level_dicts]
            if tasks:
                await asyncio.gather(*tasks) 
            # end of concurrency for levels under 1 run function          

        output = {"Content": "Single run function integration -- integration code generation", "LLM": user_model, "repo_dir": repo_dir, "sandbox_dir": current_sandbox_dir}
        self._log_event(output)
        return output
    
    
    def Integrate_code_single_run_execution(self, user_model: str, current_sandbox_dir: str = None, upper_num: int = 0):

        if current_sandbox_dir is None:
            raise ValueError("current_sandbox_dir is None.")
        sandbox_path = Path(current_sandbox_dir)
        if not sandbox_path.exists():
            raise FileNotFoundError(f"sandbox directory '{sandbox_path}' not found.")

        # find all subdirectories in the sandbox directory (different run functions)
        run_function_dirs = [d for d in sandbox_path.iterdir() if d.is_dir()]
        run_function_dirs = sorted(run_function_dirs)
        
        all_run_outputs = []

        # Iterative execution for each run function subdirectory
        for run_function_dir in run_function_dirs:
            files_list = []
            for f in run_function_dir.iterdir():
                if f.is_file() and f.suffix.lower() in (".py", ".jl"):
                    files_list.append(f)
            
            files_list = sorted(files_list)
            
            judge_results = []

            for i, target_file_path in enumerate(files_list):
                if upper_num > 0 and i >= upper_num:
                    break

                target_file_name = target_file_path.name
                target_file = self._load_file(target_file_path)
                print(f"Executing code in {run_function_dir.name}: {target_file_name}...")
                
                result_target_file = run_program.run_julia_file(target_file_path, timeout=600)
                result_target_file_exitcode = result_target_file["exitcode"]
                result_target_file_stderr = result_target_file["stderr"]

                # Evaluate execution result by LLM
                role_prompt = self._load_prompt("IntegrateCode_sandbox_single_run_execution_evaluation_stand1")
                parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                if target_file_name is not None:
                    parts.append(f"TARGET_FILE_NAME:\n")
                    parts.append(json.dumps(target_file_name, ensure_ascii=False))
                    parts.append("\n\n")
                if target_file is not None:
                    parts.append(f"TARGET_FILE_CONTENT:\n")
                    parts.append(json.dumps(target_file, ensure_ascii=False))
                    parts.append("\n\n")
                if result_target_file_exitcode is not None:
                    parts.append(f"RESULT_TARGET_FILE_EXITCODE:\n")
                    parts.append(json.dumps(result_target_file_exitcode, ensure_ascii=False))
                    parts.append("\n\n")
                if result_target_file_stderr is not None:
                    parts.append(f"RESULT_TARGET_FILE_STDERR:\n")
                    parts.append(json.dumps(result_target_file_stderr, ensure_ascii=False))
                    parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)

                response, _ = self._send_chat(user_model=user_model, user_prompt=user_prompt)
                response = self._parse_json(response)
                judge_results.append(response)

            # Create and log output for the current run function subdirectory
            output = {
                "Content": f"Integration test of {run_function_dir.name}",
                "LLM": user_model,
                "run_function_dir": str(run_function_dir),
                "judge_results": judge_results,
            }
            self._log_event(output)
            all_run_outputs.append(output)

        return all_run_outputs
    

    async def Integrate_code_single_run_execution_concurr(self, user_model: str, current_sandbox_dir: str = None, upper_num: int = 0, concurr_num: int = 5):

        if current_sandbox_dir is None:
            raise ValueError("current_sandbox_dir is None.")
        sandbox_path = Path(current_sandbox_dir)
        if not sandbox_path.exists():
            raise FileNotFoundError(f"sandbox directory '{sandbox_path}' not found.")

        # find all subdirectories in the sandbox directory (different run functions)
        run_function_dirs = [d for d in sandbox_path.iterdir() if d.is_dir()]
        run_function_dirs = sorted(run_function_dirs)
        
        all_run_outputs = []

        # Iterative execution for each run function subdirectory
        for run_function_dir in run_function_dirs:
            files_list = []
            for f in run_function_dir.iterdir():
                if f.is_file() and f.suffix.lower() in (".py", ".jl"):
                    # Only include files ending with "integtest.{ext}"
                    if f.stem.endswith("integtest"):
                        files_list.append(f)
            
            files_list = sorted(files_list)
            judge_results = []

            target_files = files_list[:upper_num] if upper_num > 0 else files_list
            sem = asyncio.Semaphore(concurr_num)

            async def _process_target_file(target_file_path: Path):
                target_file_name = target_file_path.name
                target_file = self._load_file(target_file_path)
                print(f"Executing code in {run_function_dir.name}: {target_file_name}...")
                
                # Select run function based on file extension
                file_ext = target_file_path.suffix.lower()
                if file_ext == ".py":
                    run_func = run_program.run_python_file
                elif file_ext == ".jl":
                    run_func = run_program.run_julia_file
                else:
                    raise ValueError(f"Unsupported file extension: {file_ext}")
                
                # run function is blocking; run in thread
                result_target_file = await asyncio.to_thread(run_func, str(target_file_path), str(run_function_dir), 600)
                result_target_file_exitcode = result_target_file["exitcode"]
                result_target_file_stderr = result_target_file["stderr"]

                # Evaluate execution result by LLM
                role_prompt = self._load_prompt(topic="general_dft", prompt_file="IntegrateCode_sandbox_single_run_execution_evaluation_stand1")
                parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                if target_file_name is not None:
                    parts.append(f"TARGET_FILE_NAME:\n")
                    parts.append(json.dumps(target_file_name, ensure_ascii=False))
                    parts.append("\n\n")
                if target_file is not None:
                    parts.append(f"TARGET_FILE_CONTENT:\n")
                    parts.append(json.dumps(target_file, ensure_ascii=False))
                    parts.append("\n\n")
                if result_target_file_exitcode is not None:
                    parts.append(f"RESULT_TARGET_FILE_EXITCODE:\n")
                    parts.append(json.dumps(result_target_file_exitcode, ensure_ascii=False))
                    parts.append("\n\n")
                if result_target_file_stderr is not None:
                    parts.append(f"RESULT_TARGET_FILE_STDERR:\n")
                    parts.append(json.dumps(result_target_file_stderr, ensure_ascii=False))
                    parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)

                # response = await asyncio.to_thread(self._send_chat, user_model=user_model, user_prompt=user_prompt)
                response = await asyncio.to_thread(self._send_chat_robust, user_model=user_model, user_prompt=user_prompt, max_iter=3)
                response = self._parse_json(response)
                return response
            
            tasks = [asyncio.create_task(_process_target_file(tf)) for tf in target_files]
            if tasks:
                results = await asyncio.gather(*tasks)
                judge_results.extend(results)

            # Create and log output for the current run function subdirectory
            output = {
                "Content": f"Integration test of {run_function_dir.name}",
                "LLM": user_model,
                "run_function_dir": str(run_function_dir),
                "judge_results": judge_results,
            }
            self._log_event(output)
            all_run_outputs.append(output)

        return all_run_outputs

    
class RubricsGrader(DFT_agent):

    def Generate_rubrics_scorecard(self, rubrics_dir: str, rubrics_name: str) -> dict:
        
        # load rubrics file-------------------------------------------------------------
        rubrics_dir = Path(rubrics_dir)
        rubrics_file_path = rubrics_dir / rubrics_name
        if not rubrics_file_path.exists() or not rubrics_file_path.is_file():
            raise FileNotFoundError(f"Rubrics file '{rubrics_file_path}' not found or is not a file.")
        
        # generate the scorecard--------------------------------------------------------
        # AST parse
        try:
            source = rubrics_file_path.read_text(encoding='utf-8')
        except Exception as e:
            raise RuntimeError(f"Failed to read rubrics file '{rubrics_file_path}': {e}")
        try:
            module = ast.parse(source, filename=str(rubrics_file_path))
        except SyntaxError as e:
            raise RuntimeError(f"Syntax error while parsing rubrics file '{rubrics_file_path}': {e}")

        dict_node = None
        for node in module.body:
            # handle assignments like: VAR = { ... }
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                dict_node = node.value
                break
            # handle expression like: { ... }
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Dict):
                dict_node = node.value
                break
        if dict_node is None:
            raise RuntimeError(f"No top-level dict literal found in rubrics file '{rubrics_file_path}'.")

        # Convert ast.Dict node to Python object recursively but only for simple literals
        def ast_literal_to_obj(n):
            if isinstance(n, ast.Dict):
                return {ast_literal_to_obj(k): ast_literal_to_obj(v) for k, v in zip(n.keys, n.values)}
            if isinstance(n, ast.List):
                return [ast_literal_to_obj(e) for e in n.elts]
            if isinstance(n, ast.Tuple):
                return tuple(ast_literal_to_obj(e) for e in n.elts)
            if isinstance(n, ast.Constant):
                return n.value
            if isinstance(n, ast.Str):  # Python <3.8
                return n.s
            # unsupported node types: return None
            return None
        rubrics_obj = ast_literal_to_obj(dict_node)
        if not isinstance(rubrics_obj, dict):
            raise RuntimeError(f"Parsed rubrics content is not a dict in '{rubrics_file_path}'.")

        # Recursively search for all 'issue' entries inside the nested structure
        issues = []
        def collect_issues_with_weight(obj, cumulative_weight=Decimal('1')):
            # obj: current node (dict/list/other), cumulative_weight: Decimal product of weights from root to parent of obj
            if isinstance(obj, dict):
                # Update cumulative weight if this dict has a 'weight' key
                local_weight = obj.get('weight') if 'weight' in obj else None
                try:
                    if isinstance(local_weight, (int, float, str)):
                        # use Decimal(str(...)) to avoid binary float issues
                        new_cum_weight = cumulative_weight * Decimal(str(local_weight))
                    else:
                        new_cum_weight = Decimal(cumulative_weight)
                except (InvalidOperation, Exception):
                    # fallback to previous cumulative_weight if conversion fails
                    new_cum_weight = Decimal(cumulative_weight)

                # If this dict has 'issue', capture the whole dict and write the computed cumulative weight into 'weight'
                if 'issue' in obj:
                    try:
                        unit = dict(obj)  # shallow copy
                    except Exception:
                        unit = obj
                    # overwrite the unit's weight with the computed cumulative weight (as a clean float)
                    try:
                        unit['weight'] = float(new_cum_weight)
                    except Exception:
                        # fallback to string representation
                        unit['weight'] = str(new_cum_weight)
                    issues.append(unit)

                # Recurse into children values with updated cumulative weight
                for v in obj.values():
                    collect_issues_with_weight(v, new_cum_weight)
            elif isinstance(obj, list):
                for item in obj:
                    collect_issues_with_weight(item, cumulative_weight)
        collect_issues_with_weight(rubrics_obj, cumulative_weight=Decimal('1'))

        # Extract guide_caption from top-level rubrics object if present
        guide_caption = None
        try:
            guide_caption = rubrics_obj.get('guide_caption') if isinstance(rubrics_obj, dict) else None
        except Exception:
            guide_caption = None

        # Compute normalization check: sum of weights should be 1 (within tolerance)
        total = Decimal('0')
        for u in issues:
            try:
                total += Decimal(str(u.get('weight', 0)))
            except Exception:
                # ignore non-numeric weights
                pass
        norm_ok = abs(float(total - Decimal('1'))) <= 1e-4

        # Output----------------------------------------------------------------------------------
        scorecard = {
            'rubrics_file': str(rubrics_file_path),
            'guide_caption': guide_caption,
            'task_confirm': "TBD",
            'issue_list': issues,
            'NormCheck': True if norm_ok else False,
            'total_weight': float(total),
        }
        # self._log_event(scorecard)

        return scorecard
    

    def _Compute_rubrics_grade(self, scorecard_graded: dict) -> dict:
        
        if not isinstance(scorecard_graded, dict):
            return scorecard_graded
        issues = scorecard_graded.get('issue_list') or []
        total = Decimal('0')

        # helper to extract judge value from an issue unit
        def extract_judge(unit):
            candidates = ['judge'] # candidate keys where judge might be stored
            for k in candidates:
                if k in unit:
                    v = unit.get(k)
                    # normalize booleans
                    if isinstance(v, bool):
                        return Decimal('1') if v else Decimal('0')
                    # numbers or numeric strings
                    try:
                        return Decimal(str(v))
                    except Exception:
                        # try boolean-like strings
                        if isinstance(v, str):
                            if v.lower() in ('true', '1'):
                                return Decimal('1')
                            if v.lower() in ('false', '0'):
                                return Decimal('0')
            # fallback: if there's a nested 'children' with judge info, try to find any numeric judge
            if 'children' in unit and isinstance(unit['children'], dict):
                for val in unit['children'].values():
                    if isinstance(val, (int, float)):
                        return Decimal(str(val))
                    if isinstance(val, str) and val.isdigit():
                        return Decimal(val)
            return Decimal('0')

        for u in issues:
            try:
                w = Decimal(str(u.get('weight', 0)))
            except Exception:
                w = Decimal('0')
            j = extract_judge(u)
            try:
                total += w * j
            except Exception:
                # if multiplication fails, skip this unit
                continue

        # write the final grade as float
        scorecard_graded['rubrics_grade'] = float(total)
        return scorecard_graded
    

    def Grade_rubrics_scorecard(self, user_model: str, paper_path: str, paper_name: str, code_file: str, scorecard: dict) -> dict:

        # load paper source file (PDF, tex...)----------------------------------------------
        paper_dir = Path(self.dataset_dir) / self.topic / paper_path
        paper_file_path = paper_dir / (paper_name + ".tex")
        if not paper_file_path.exists() or not paper_file_path.is_file():
            raise FileNotFoundError(f"Source file '{paper_file_path}' not found or is not a file.")
        paper_content = self._load_file(paper_file_path)

        # load LLM generated original code file (a single .py or .jl file)------------------
        code_file_path = Path(code_file)
        if not code_file_path.exists() or not code_file_path.is_file():
            raise FileNotFoundError(f"Code file '{code_file_path}' not found or is not a file.")
        code_content = self._load_file(code_file_path)

        # construct prompt-------------------------------------------------------------------
        role_prompt = self._load_prompt(topic="general_dft", prompt_file="GradeRubrics_scorecard_stand1")
        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        if paper_content is not None:
            parts.append(f"PaperFile (paper content):\n")
            parts.append(json.dumps(paper_content, ensure_ascii=False))
            parts.append("\n\n")
        if code_content is not None:
            parts.append(f"CodeFile (code content):\n")
            parts.append(code_content)
            parts.append("\n\n")
        if scorecard is not None:
            parts.append(f"scorecard:\n")
            parts.append(json.dumps(scorecard, ensure_ascii=False))
            parts.append("\n\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)

        response, _ = self._send_chat(user_model=user_model, user_prompt=user_prompt)
        response = self._parse_json(response)
        
        # output--------------------------------------------------------------------------------
        scorecard_graded = response

        # Compute final rubrics grade from the graded scorecard and attach it
        try:
            scorecard_graded = self._Compute_rubrics_grade(scorecard_graded)
        except Exception as e:
            # If grading fails, log the error but still return the raw graded scorecard
            self._log_event({"error": "Compute_rubrics_grade failed", "exception": str(e), "scorecard_graded": scorecard_graded})

        self._log_event(scorecard_graded)
        return scorecard_graded

    







