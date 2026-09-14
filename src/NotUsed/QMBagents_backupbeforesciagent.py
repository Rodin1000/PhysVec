# main.py
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
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Import from this project
from config import OPENROUTER_API_KEY, YIDONG_API_KEY
from config import get_model_settings
from utils import MCP_toolbox, run_program, code_editor, send_chat_yidong

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
class QMBagent:
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
            raise ValueError("API key must be provided for QMBagent.")
        if self.topic is None:
            raise ValueError("Topic must be specified for QMBagent.")


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

    
    def _extract_json_from_response(self, response: str) -> Optional[dict]:
        """
        Extract JSON dictionary from LLM response using robust extraction logic.
        Searches forward for ```json and backward for ``` to find the complete JSON block.
        Uses iterative approach to handle nested code blocks.
        Returns the parsed dict if successful, None otherwise.
        """
        try:
            # Ensure response is a string
            if response is None:
                return None
            if not isinstance(response, str):
                content = str(response)
            else:
                content = response
            # Pattern to find ```json at the start (search forward)
            json_start_pattern = re.compile(r"```json\s*\r?\n", re.IGNORECASE)

            # Iterative approach: try different combinations until we get a valid dict
            while True:
                # Find first ```json position (search forward)
                start_match = json_start_pattern.search(content)
                if not start_match:
                    # No start marker found, log and break
                    self._log_event({
                        "Content": "Extract JSON failed - no start marker",
                        "content_preview": content[:300] if content else "None"
                    })
                    break
                
                start_pos = start_match.end()
                
                # Search backward for ``` from the end
                # Find the last occurrence of ``` in the remaining content
                remaining = content[start_pos:]
                # Find all ``` positions in remaining content
                end_positions = []
                for match in re.finditer(r"```", remaining):
                    end_positions.append(match.start())
                
                if not end_positions:
                    # No end marker found, log and try nested approach
                    self._log_event({
                        "Content": "Extract JSON failed - no end marker",
                        "start_pos": start_pos,
                        "remaining_preview": remaining[:200] if remaining else "None"
                    })
                    break
                
                # Use the last ``` position (search backward)
                last_end_pos = end_positions[-1]
                extracted = remaining[:last_end_pos].strip()
                
                try:
                    data = json.loads(extracted, strict=False)
                    # print("loads success")
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError as e:
                    # Log detailed JSON parsing error
                    self._log_event({
                        "Content": "Extract JSON failed - JSON decode error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "error_line": getattr(e, 'lineno', None),
                        "error_col": getattr(e, 'colno', None),
                        "extracted_preview": extracted if extracted else "None",
                        "extracted_length": len(extracted) if extracted else 0
                    })
                except Exception as e:
                    # Log other parsing errors
                    self._log_event({
                        "Content": "Extract JSON failed - other error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "extracted_preview": extracted[:300] if extracted else "None"
                    })
                
                # If parsing failed, try removing one layer of code fences
                # This handles nested code blocks (iterative approach)
                json_fence = re.compile(r"```json\s*\r?\n(.*?)\r?\n```", re.DOTALL | re.IGNORECASE)
                
                match = json_fence.search(content)
                if not match:
                    break
                content = match.group(1).strip()

            # Final attempt: try parsing the entire content as JSON
            try:
                data = json.loads(content.strip())
                return data if isinstance(data, dict) else None
            except Exception as e:
                self._log_event({
                    "Content": "Extract JSON failed - final attempt",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "content_preview": content[:300] if content else "None"
                })
                pass
                
            return None
        except Exception as e:
            # Log the actual error for debugging
            self._log_event({
                "Content": "Extract JSON failed - exception",
                "error": str(e),
                "error_type": type(e).__name__,
                "content_preview": content[:500] if content else "None"
            })
            return None
            
    def _save_code_file(self, response: str, anci: str = "", code_dir: str = None, isarbi: bool = False, content_tag: str = "code_content", file_ext: str = None):

        try:
            data = self._extract_json_from_response(response)
            if data is None:
                self._log_event({"Content": "Save code failed-type 1", "reason": "response is not valid JSON", "response": response})
                return None
        except Exception as e:
            self._log_event({"Content": "Save code failed-type 1", "reason": "response is not valid JSON", "error": str(e), "response": response})
            return None

        try:
            subplot_name = data.get("subplot_name")
            code_type = data.get("code_type")
            code_content = data.get(content_tag)
        except Exception as e:
            self._log_event({"Content": "Save code failed", "reason": "response JSON missing fields", "error": str(e)})
            return None

        if not code_content:
            self._log_event({"Content": "Save code failed", "reason": "missing code_content"})
            return None

        # Determine file extension
        if file_ext is not None:
            # Use provided file_ext (remove leading dot if present)
            ext = file_ext.lstrip('.')
        else:
            # Use code_type from response
            if not code_type:
                self._log_event({"Content": "Save code failed", "reason": "missing code_type and file_ext not provided"})
                return None
            ct = str(code_type).strip().lower()
            if ct in ("python", "py"):
                ext = "py"
            elif ct in ("julia", "jl"):
                ext = "jl"
            else:
                self._log_event({"Content": "Save code failed", "reason": f"unsupported code_type: {code_type}"})
                return None

        # Decode escaped newlines in code_content if present
        if isinstance(code_content, str) and "\\n" in code_content and "\n" not in code_content:
            try:
                code_content = code_content.encode('utf-8').decode('unicode_escape')
            except Exception:
                code_content = code_content.replace('\\n', '\n')

        # Write file
        if code_dir is None:
            code_dir = Path(self.current_code_dir)
        else:
            code_dir = Path(code_dir)
        code_dir.mkdir(parents=True, exist_ok=True)

        if isarbi == False:
            ts = time.strftime("%Y%m%d_%H%M%S")
            safe = (str(subplot_name).replace(' ', '_') if subplot_name else "generated")
            filename = f"{safe}_{ts}_{anci}.{ext}"
        else:
            filename = f"{anci}.{ext}"

        path = code_dir / filename
        try:
            path.write_text(code_content, encoding='utf-8')
            # self._log_event({"Content": "Saved generated code", "path": str(path), "code_type": ct})
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
    def _send_chat(self, user_model: str, user_prompt: str, temperature: float = None, top_p: float = None, api_type: str = "yidong", iscaltoken: bool = False) -> Tuple[str, Optional[Dict]]:
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

        QMB_author_prompt = self._load_prompt(topic="general", prompt_file="QMB_author")
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
                    model = model_name,
                    messages = message_to_send,
                    **model_params
                )
                return response.choices[0].message.content.strip(), None
            elif api_type == "yidong":
                response_text, status_code, token_dict = send_chat_yidong.send_chat_diverse_model(user_model=model_name, role_prompt=QMB_author_prompt, user_prompt=user_prompt, temperature=temp_set, top_p=top_p_set, iscaltoken=iscaltoken)
                if status_code != 200:
                    print(f"calling model {model_name} failed with status code: {status_code}, the response text is: {response_text}")
                return response_text, token_dict
        except Exception as e:
            return f"calling model {model_name} failed with error: {str(e)}", None

    
    def _send_chat_robust(self, user_model: str, user_prompt: str, max_iter: int = 3, temperature: float = None, top_p: float = None, iscaltoken: bool = False):
        """
        Send chat messages to the LLM and get the response, with validation.
        Retries up to max_iter times if the response cannot be parsed as a JSON dictionary.
        
        Args:
            user_model: Model name to use
            user_prompt: User prompt
            max_iter: Maximum number of retry attempts (default: 3)
            iscaltoken: Whether to calculate and return token statistics
        
        Returns:
            If iscaltoken=False: response string
            If iscaltoken=True: (response_string, token_dict)
        """
        for attempt in range(1, max_iter + 1):
            response_text, token_dict = self._send_chat(user_model, user_prompt, temperature=temperature, top_p=top_p, iscaltoken=iscaltoken)
            
            # Try to extract JSON dictionary
            parsed_dict = self._extract_json_from_response(response_text)
            
            if parsed_dict is not None:
                # Successfully extracted a dictionary
                if attempt > 1:
                    self._log_event({
                        "Content": "Send chat robust succeeded after retry",
                        "LLM": user_model,
                        "attempt": attempt,
                        "max_iter": max_iter
                    })
                if iscaltoken:
                    return response_text, token_dict
                else:
                    return response_text
            
            # Failed to extract dictionary
            if attempt < max_iter:
                self._log_event({
                    "Content": "Send chat robust retry",
                    "LLM": user_model,
                    "attempt": attempt,
                    "max_iter": max_iter,
                    "reason": "Failed to extract JSON dictionary from response",
                    "response_snippet": response_text if response_text else "None"
                })
            else:
                # Last attempt failed
                self._log_event({
                    "Content": "Send chat robust failed after max_iter",
                    "LLM": user_model,
                    "attempt": attempt,
                    "max_iter": max_iter,
                    "reason": "Failed to extract JSON dictionary after all retries",
                    "response_snippet": response_text if response_text else "None"
                })
        
        # Return the last response even if parsing failed
        if iscaltoken:
            return response_text, token_dict
        else:
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
        max_steps: int = 50,
        iscaltool: bool = False,
    ):
        response = await MCP_toolbox.send_chat_through_mcp_dynamic(
            user_model=user_model, 
            user_prompt=user_prompt, 
            fs_target_dir=fs_target_dir, 
            references=reference_dir_dict,
            retrieve_rt_target_dir=retrieve_rt_target_dir,
            init_servers=init_servers,
            program_path=program_path,
            program_timeout=program_timeout,
            max_steps=max_steps,
            iscaltool=iscaltool,
        )
        return response

class PaperSummerizer(QMBagent):

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
        """
        # role_prompt = self._load_prompt("PlanTaskByText")
        # role_prompt = self._load_prompt(topic="general", prompt_file="PlanTaskByText_disabled")
        role_prompt = self._load_prompt(topic="general", prompt_file="PlanTaskByText_freereasoning")

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

        # response = self._send_chat(user_model, user_prompt)
        if iscaltoken:
            response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
        else:
            response = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=False)
            token_dict = None
        parsed_response = self._parse_json(response)

        output = {"Content": "Implementation plan by text", "LLM": user_model, "subplot_name": parsed_response["subplot_name"], "User_requests": parsed_response["User_requests"], "plan": parsed_response.get("plan", None), "token_stats": token_dict}
        self._log_event(output)
        return output
    

class CodeGenerator(QMBagent):

    def Generate_code_by_plan(self, user_model: str, Plan_info: dict = None, iscaltoken: bool = False) -> dict:
        """
        Generate code by plan using query summary (non-MCP workflow version).
        Combines GenerateCodeByPlan_Query_instruct_stand1 + Plan_info + query_summary + GenerateCode_rules_stand1.
        """
        # Load prompt
        role_prompt = self._load_prompt(topic="general", prompt_file="GenerateCodeByPlan_Query_instruct_stand1")
        
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
        # response = self._send_chat(user_model, user_prompt)
        if iscaltoken:
            response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=True)
        else:
            response = self._send_chat_robust(user_model, user_prompt, max_iter=3, iscaltoken=False)
            token_dict = None
        
        # Save code file
        save_code_output = self._save_code_file(response, anci="code_LLM", isarbi=True)
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
        response = self._send_chat(user_model, user_prompt)
        
        save_code_output = self._save_code_file(response)
        subplot_name = save_code_output.get("subplot_name") if save_code_output else None
        code_file = save_code_output.get("code_file") if save_code_output else None
        output = {"Content": "Generate code", "LLM": user_model, "subplot_name": subplot_name, "code_file": code_file, "response": response}

        self._log_event(output)
        return output
    

    async def Generate_code_by_plan_mcp_use(self, user_model: str, Plan_info: dict = None, PDF_info: dict = None, isusequery: bool = False) -> dict:

        # Select prompt and reference directory based on isusequery flag
        if isusequery:
            # Use query-based prompt and reference query_dir
            role_prompt = self._load_prompt(topic="general", prompt_file="GenerateCodeByPlan_mcp_use_Query_instruct_stand1")
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
            init_servers=["filesystem-mcp"]
        )
        
        save_code_output = self._save_code_file(response, anci = "code_LLM", isarbi=True)
        subplot_name = save_code_output.get("subplot_name") if save_code_output else None
        code_file = save_code_output.get("code_file") if save_code_output else None
        code_content = save_code_output.get("code_content") if save_code_output else None
        output = {"Content": "Generated code", "LLM": user_model, "subplot_name": subplot_name, "code_file": code_file, "code_content": code_content}

        self._log_event(output)
        return output


    async def Retrieve_knowledge_by_plan(self, user_model: str, Plan_info: dict) -> dict:

        # Generate all queries for RAG---------------------------------
        # load prompt
        role_prompt = self._load_prompt(topic="general", prompt_file="RetrieveKnowledgeByPlan_stand1")
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


    async def Retrieve_knowledge_by_plan_mcp_use(self, user_model: str, Plan_info: dict, User_requests: str = None, iscaltoken: bool = False, iscaltool: bool = False) -> dict:

        # load prompt
        role_prompt = self._load_prompt(topic="general", prompt_file="RetrieveKnowledgeByPlan_mcp_use_stand1")
        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        parts.append("TOPIC:\n")
        parts.append(self.topic)
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
                summary_prompt = self._load_prompt(topic="general", prompt_file="RetrieveKnowledgeByPlan_summary_stand1")
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
        role_prompt = self._load_prompt(topic="general", prompt_file="RefineCodeByRules_mcp_use_stand1")
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
        reason_prompt = self._load_prompt(topic="general", prompt_file="RefineCodeByRules_reason_stand1")
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
        act_prompt = self._load_prompt(topic="general", prompt_file="RefineCodeByRules_act_stand1")
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
                
                # Save refined code after each part using _save_code_file
                code_type = "julia" if file_ext == ".jl" else "python"
                part_response_json = json.dumps({
                    "code_type": code_type,
                    "code_content": code_former
                }, ensure_ascii=False)
                self._save_code_file(
                    response=part_response_json,
                    anci=f"code_LLM_{count_iter}_{part_name}",
                    code_dir=str(code_path.parent),
                    isarbi=True
                )
            
            # Save final refined code using _save_code_file
            code_type = "julia" if file_ext == ".jl" else "python"
            final_response_json = json.dumps({
                "code_type": code_type,
                "code_content": code_former
            }, ensure_ascii=False)
            self._save_code_file(
                response=final_response_json,
                anci=f"code_LLM_{count_iter}",
                code_dir=str(code_path.parent),
                isarbi=True
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
            
            role_prompt = self._load_prompt(topic="general", prompt_file="RefineCodeByRules_judge_stand1")
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


    async def FormatCheck_code_by_rules(self, user_model: str, judge_model: str, code_file: str = None, max_iter: int = 3, temp_incr: float = 0.15, code_backup_name: str = "code_LLM_origin", code_save_name: str = "code_LLM", tag_name: str = None, iscaltoken: bool = False):
        """
        Format check code by rules.
        Returns: (improved_code, flag_obey, count_iter) if iscaltoken=False
                 (improved_code, flag_obey, count_iter, user_model_token_stats, judge_model_token_stats) if iscaltoken=True
        """

        # Step 1: Detect code file and create backup
        code_path = Path(code_file)
        if not code_path.exists():
            raise FileNotFoundError(f"Code file not found: {code_path}")

        # Step 2: Create backup: rename code_LLM.{ext} to code_LLM_origin.{ext}
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
        improve_prompt = self._load_prompt(topic="general", prompt_file="FormatCheck_code_by_rules_improve_stand1")
        judge_prompt = self._load_prompt(topic="general", prompt_file="FormatCheck_code_by_rules_judge_stand1")
        
        # Step 5: Iterative improve and judge
        # Get default temperature for user_model
        user_settings = get_model_settings(user_model)
        default_temperature = user_settings.get("temperature")
        
        flag_obey = False
        count_iter = 0
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
            # Add code file extension information
            code_type_hint = "python" if file_ext == ".py" else "julia" if file_ext == ".jl" else None
            if code_type_hint:
                parts.append(f"CODE_TYPE:\n{code_type_hint}\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            user_prompt = "".join(parts)
            # response = self._send_chat(user_model, user_prompt)
            if iscaltoken:
                response, token_dict = self._send_chat_robust(user_model, user_prompt, max_iter=3, temperature=current_temperature, iscaltoken=True)
                if token_dict:
                    user_model_token_dicts.append(token_dict)
            else:
                response = self._send_chat_robust(user_model, user_prompt, max_iter=3, temperature=current_temperature, iscaltoken=False)
            parsed_response = self._parse_json(response)
            improved_code = parsed_response.get("code_content", "")
            improve_summary = parsed_response.get("summary", "")
            self._save_code_file(response=response, anci=code_save_name, isarbi=True, file_ext=file_ext)
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



class RepoGenerator(QMBagent):

    async def Generate_repo_by_codefile(self, user_model: str, code_file: str, fs_target_dir: str) -> dict:
        # Read code file
        # original_code = self._load_file("..\\Output_prepare\\prepare_PhysRevLett.80.5607_Fig._2_(a)\\phase_diagram_20251015_160439_.jl")
        original_code = self._load_file(Path(code_file))

        # Construct prompts
        role_prompt = self._load_prompt("GenerateRepo_mcp_use_by_codefile_stand1")
        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        if original_code is not None:
            parts.append("ORIGINAL_CODE:\n")
            parts.append(json.dumps(original_code, ensure_ascii=False))
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
        # Read code file
        original_code = self._load_file(Path(code_file))

        # Construct repository by hardcode
        code_editor.split_functions(code_file, fs_target_dir)
        code_editor.delete_unused_imports(fs_target_dir, count_type_annotations=True)

        # Generate README via LLM
        role_prompt = self._load_prompt(topic="general", prompt_file="GenerateRepo_README_only_by_codefile_stand1")
        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        if original_code is not None:
            parts.append("ORIGINAL_CODE:\n")
            parts.append(json.dumps(original_code, ensure_ascii=False))
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
    

class CodeVerifier(QMBagent):

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
        save_code_output = self._save_code_file(response, code_dir=current_sandbox_dir)
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
                    parts.append(json.dumps(target_function, ensure_ascii=False))
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
                    parts.append(json.dumps(lib_text, ensure_ascii=False))
                    parts.append("\n\n")
                if main_function is not None:
                    parts.append("MAIN_FUNCTION:\n")
                    parts.append(json.dumps(main_function, ensure_ascii=False))
                    parts.append("\n\n")
                if target_function is not None:
                    parts.append(f"TARGET_FUNCTION:\n")
                    parts.append(json.dumps(target_function, ensure_ascii=False))
                    parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)

                print(f"verify code: {target_function_name} generating...")
                response = self._send_chat(user_model=user_model, user_prompt=user_prompt)
            
            function_name_nosuffix = target_function_name.rsplit('.', 1)[0]
            save_code_output = self._save_code_file(response, anci=f"verify_{function_name_nosuffix}", code_dir=current_sandbox_dir, isarbi=True)
            

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
                    parts.append(json.dumps(lib_text, ensure_ascii=False))
                    parts.append("\n\n")
                if main_function is not None:
                    parts.append("MAIN_FUNCTION:\n")
                    parts.append(json.dumps(main_function, ensure_ascii=False))
                    parts.append("\n\n")
                if target_function is not None:
                    parts.append(f"TARGET_FUNCTION:\n")
                    parts.append(json.dumps(target_function, ensure_ascii=False))
                    parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)

                print(f"verify code: {target_function_name} generating...")

                # _send_chat is blocking; run it in a thread to avoid blocking the event loop
                response = await asyncio.to_thread(self._send_chat, user_model, user_prompt)

                function_name_nosuffix = target_function_name.rsplit('.', 1)[0]
                # _save_code_file does IO; run in thread as well (use keyword args to ensure correct mapping)
                await asyncio.to_thread(self._save_code_file, response, anci=f"verify_{function_name_nosuffix}", code_dir=current_sandbox_dir, isarbi=True)

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
            decide_prompt = self._load_prompt(topic="general", prompt_file="VerifyCode_sandbox_single_repocode_generation_decidelibrary_stand1")
            parts = [decide_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            parts.append("TARGET_FUNCTION:\n")
            parts.append(json.dumps(code_file_content, ensure_ascii=False))
            parts.append("\n\n")
            parts.append("AVAILABLE_LIBRARY_FILES:\n")
            parts.append(json.dumps(available_library_files, ensure_ascii=False))
            parts.append("\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            decide_user_prompt = "".join(parts)
            
            decide_response = self._send_chat_robust(user_model, decide_user_prompt, max_iter=3)
            decide_parsed = self._parse_json(decide_response)
            selected_library_files = decide_parsed.get("selected_files", [])
            self._log_event({"Content": "Decide which library file(s) to use based on code_file content", "LLM": user_model, "selected_library_files": selected_library_files})
            
            if not selected_library_files:
                raise ValueError("No library file selected based on code_file content")
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
                    parts.append(json.dumps(lib_text, ensure_ascii=False))
                    parts.append("\n\n")
                if main_function is not None:
                    parts.append("MAIN_FUNCTION:\n")
                    parts.append(json.dumps(main_function, ensure_ascii=False))
                    parts.append("\n\n")
                if target_function is not None:
                    parts.append(f"TARGET_FUNCTION:\n")
                    parts.append(json.dumps(target_function, ensure_ascii=False))
                    parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)

                print(f"verify code: {target_function_name} generating...")

                # _send_chat is blocking; run it in a thread to avoid blocking the event loop
                response = await asyncio.to_thread(self._send_chat_robust, user_model, user_prompt, max_iter=3)

                function_name_nosuffix = target_function_name.rsplit('.', 1)[0]
                # _save_code_file does IO; run in thread as well (use keyword args to ensure correct mapping)
                await asyncio.to_thread(self._save_code_file, response, anci=f"verify_{function_name_nosuffix}", code_dir=current_sandbox_dir, isarbi=True)

                return target_function_name

        # schedule workers and wait
        tasks = [asyncio.create_task(_process_target(name, selected_library_files)) for name in targets]
        if tasks:
            await asyncio.gather(*tasks)
        output = {"Content": "Single repocode verification code generation", "LLM": user_model, "repo_dir": repo_dir, "sandbox_dir": current_sandbox_dir}
        self._log_event(output)

        return output


    async def Verify_code_single_repocode_generation_autolibrary_split_concurr(self, user_model: str, current_sandbox_dir: str = None, code_file: str = None, repo_dir: str = None, upper_num: int = 0, concurr_num: int = 5, iscaltoken: bool = False) ->dict:
       
        # Extract information from the target repo----------------------------------------
        if self.topic == "dmrg":
            prefix_list = ["site", "hamiltonian", "initialstate", "observable", "effector"]
        elif self.topic == "nnwf":
            prefix_list = ["hilbert", "statemodel", "compset", "hamiltonian", "observable", "effector"]
        elif self.topic == "qcmb":
            prefix_list = ["hamiltonian", "circinit", "circevol", "effector", "observable"]
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
        decide_token_dict = None
        if code_file_content is not None and available_library_files:
            decide_prompt = self._load_prompt(topic="general", prompt_file="VerifyCode_sandbox_single_repocode_generation_decidelibrary_stand1")
            parts = [decide_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            parts.append("TARGET_FUNCTION:\n")
            parts.append(json.dumps(code_file_content, ensure_ascii=False))
            parts.append("\n\n")
            parts.append("AVAILABLE_LIBRARY_FILES:\n")
            parts.append(json.dumps(available_library_files, ensure_ascii=False))
            parts.append("\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            decide_user_prompt = "".join(parts)
            
            if iscaltoken:
                decide_response, decide_token_dict = self._send_chat_robust(user_model, decide_user_prompt, max_iter=3, iscaltoken=iscaltoken)
            else:
                decide_response = self._send_chat_robust(user_model, decide_user_prompt, max_iter=3)
            decide_parsed = self._parse_json(decide_response)
            selected_library_files = decide_parsed.get("selected_files", [])
            self._log_event({"Content": "Decide which library file(s) to use based on code_file content", "LLM": user_model, "selected_library_files": selected_library_files})
            
            if not selected_library_files:
                raise ValueError("No library file selected based on code_file content")
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
        
        # Initialize token statistics
        decide_token = decide_token_dict or {}

        async def _process_target(target_function_name: str, selected_files: list):
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

                # Step 2: Generate the unit test environment (verifier functions & main function & entry point)
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

                role_prompt = self._load_prompt(topic=self.topic, prompt_file="VerifyCode_sandbox_single_repocode_environment_generation_stand1")
                
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
                    parts.append(json.dumps(lib_text, ensure_ascii=False))
                    parts.append("\n\n")
                if main_function is not None:
                    parts.append("MAIN_FUNCTION:\n")
                    parts.append(json.dumps(main_function, ensure_ascii=False))
                    parts.append("\n\n")
                if target_function is not None:
                    parts.append(f"TARGET_FUNCTION:\n")
                    parts.append(json.dumps(target_function, ensure_ascii=False))
                    parts.append("\n\n")
                parts.append("-- END AGGREGATED INPUT --\n")
                user_prompt = "".join(parts)

                print(f"verify code: {target_function_name} generating...")

                # _send_chat is blocking; run it in a thread to avoid blocking the event loop
                if iscaltoken:
                    response, token_dict = await asyncio.to_thread(self._send_chat_robust, user_model, user_prompt, max_iter=3, iscaltoken=iscaltoken)
                else:
                    response = await asyncio.to_thread(self._send_chat_robust, user_model, user_prompt, max_iter=3)
                    token_dict = None

                # Save background_code and entrypoint_code separately
                await asyncio.to_thread(self._save_code_file, response, anci=f"verify_{function_name_nosuffix}_background", code_dir=current_sandbox_dir, isarbi=True, content_tag="background_code")
                await asyncio.to_thread(self._save_code_file, response, anci=f"verify_{function_name_nosuffix}_entrypoint", code_dir=current_sandbox_dir, isarbi=True, content_tag="entrypoint_code")

                # Step 3: Merge background, tobetest, and entrypoint files into a complete unit test code
                background_file = sandbox_path / f"verify_{function_name_nosuffix}_background{file_ext}"
                tobetest_file = sandbox_path / f"verify_{function_name_nosuffix}_tobetest{file_ext}"
                entrypoint_file = sandbox_path / f"verify_{function_name_nosuffix}_entrypoint{file_ext}"
                merged_file = sandbox_path / f"verify_{function_name_nosuffix}_unittest{file_ext}"
                
                # Check if all required files exist
                missing_files = []
                if not background_file.exists():
                    missing_files.append(str(background_file))
                if not tobetest_file.exists():
                    missing_files.append(str(tobetest_file))
                if not entrypoint_file.exists():
                    missing_files.append(str(entrypoint_file))
                
                if missing_files:
                    error_msg = f"Missing files for {function_name_nosuffix}: {', '.join(missing_files)}"
                    print(f"[ERROR] {error_msg}")
                    self._log_event({
                        "Content": "Merge verify code files failed - missing files",
                        "function_name": function_name_nosuffix,
                        "missing_files": missing_files
                    })
                else:
                    # Merge files in order: background --> tobetest --> entrypoint
                    merged_content_parts = []
                    try:
                        merged_content_parts.append(self._load_file(background_file))
                        merged_content_parts.append("\n\n")
                        merged_content_parts.append(self._load_file(tobetest_file))
                        merged_content_parts.append("\n\n")
                        merged_content_parts.append(self._load_file(entrypoint_file))
                        merged_content = "".join(merged_content_parts)
                        merged_file.write_text(merged_content, encoding='utf-8')
                    except Exception as e:
                        error_msg = f"Failed to merge files for {function_name_nosuffix}: {str(e)}"
                        print(f"[ERROR] {error_msg}")
                        self._log_event({
                            "Content": "Merge verify code files failed - exception",
                            "function_name": function_name_nosuffix,
                            "error": str(e)
                        })

                if iscaltoken:
                    return target_function_name, token_dict
                else:
                    return target_function_name

        # schedule workers and wait
        tasks = [asyncio.create_task(_process_target(name, selected_library_files)) for name in targets]
        if tasks:
            results = await asyncio.gather(*tasks)
        else:
            results = []
        
        # Aggregate token statistics
        total_input = decide_token.get("input_tokens", 0)
        total_output = decide_token.get("output_tokens", 0)
        if iscaltoken:
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    _, token_dict = result
                    if token_dict:
                        total_input += token_dict.get("input_tokens", 0)
                        total_output += token_dict.get("output_tokens", 0)
        
        token_stats = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total": total_input + total_output
        } if (total_input + total_output) > 0 else None
        
        output = {"Content": "Single repocode verification code generation", "LLM": user_model, "repo_dir": repo_dir, "sandbox_dir": current_sandbox_dir}
        if token_stats:
            output["token_stats"] = token_stats
        self._log_event(output)

        return output


    def Verify_code_single_repocode_execution(self, user_model: str, current_sandbox_dir: str = None, upper_num: int = 0):

        # find all .py/.jl files in the sandbox directory--------------------------------
        files_list = []
        if current_sandbox_dir is None:
            raise ValueError("current_sandbox_dir is None.")
        sandbox_path = Path(current_sandbox_dir)
        if not sandbox_path.exists():
            raise FileNotFoundError(f"sandbox directory '{sandbox_path}' not found.")

        for f in sandbox_path.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() in (".py", ".jl"):
                files_list.append(f.name)

        files_list = sorted(files_list)
        # print(f"Sandbox files (.py/.jl): {files_list}") # for debug

        # Iterative execution------------------------------------------------------------
        judge_results = []
        for i in range(len(files_list)):
            if upper_num > 0 and i >= upper_num:
                break

            target_file_name = files_list[i]
            target_file_path = sandbox_path / target_file_name
            target_file = self._load_file(target_file_path)
            print(f"verify code: {target_file_name} executing...")
            result_target_file = run_program.run_julia_file(target_file_path, timeout=600)
            result_target_file_exitcode = result_target_file["exitcode"]
            result_target_file_stderr = result_target_file["stderr"]

            # Evaluate execution result by LLM--------------------------------
            role_prompt = self._load_prompt("VerifyCode_sandbox_single_repocode_execution_evaluation_stand1")
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

            response = self._send_chat(user_model=user_model, user_prompt=user_prompt)
            response = self._parse_json(response)
            # print(response) # for debug
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
    

    async def Verify_code_single_repocode_execution_concurr(self, user_model: str, current_sandbox_dir: str = None, upper_num: int = 0, concurr_num: int = 5, exe_range: List[str] = None, islog: bool = True, iscaltoken: bool = False) ->dict:

        # find all .py/.jl files ending with "_unittest" in the sandbox directory--------------------------------
        files_list = []
        if current_sandbox_dir is None:
            raise ValueError("current_sandbox_dir is None.")
        sandbox_path = Path(current_sandbox_dir)
        if not sandbox_path.exists():
            raise FileNotFoundError(f"sandbox directory '{sandbox_path}' not found.")

        for f in sandbox_path.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() in (".py", ".jl"):
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
        # print(f"Sandbox files (*_unittest.py/jl): {files_list}") # for debug

        # Iterative execution (concurrent)-------------------------------------------------
        # Prepare list of targets respecting upper_num
        targets = files_list[:upper_num] if (upper_num > 0) else files_list

        sem = asyncio.Semaphore(concurr_num)

        async def _process_target(target_file_name: str):
            async with sem:
                target_file_path = sandbox_path / target_file_name
                target_file = self._load_file(target_file_path)
                print(f"verify code: {target_file_name} executing...")

                # Select run function based on file extension
                file_ext = target_file_path.suffix.lower()
                if file_ext == ".py":
                    run_func = run_program.run_python_file
                elif file_ext == ".jl":
                    run_func = run_program.run_julia_file
                else:
                    raise ValueError(f"Unsupported file extension: {file_ext}")
                
                # run function is blocking; run in thread
                result_target_file = await asyncio.to_thread(run_func, str(target_file_path), str(sandbox_path), 600)
                result_target_file_exitcode = result_target_file["exitcode"]
                result_target_file_stderr = result_target_file["stderr"]

                # Evaluate execution result by LLM--------------------------------
                role_prompt = self._load_prompt(topic="general", prompt_file="VerifyCode_sandbox_single_repocode_execution_evaluation_stand1")
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

                # _send_chat is blocking; run in thread
                # response = await asyncio.to_thread(self._send_chat, user_model, user_prompt)
                if iscaltoken:
                    response, token_dict = await asyncio.to_thread(self._send_chat_robust, user_model, user_prompt, max_iter=3, iscaltoken=iscaltoken)
                else:
                    response = await asyncio.to_thread(self._send_chat_robust, user_model, user_prompt, max_iter=3)
                    token_dict = None
                response = self._parse_json(response)
                if iscaltoken:
                    return response, token_dict
                else:
                    return response

        # schedule workers and wait
        tasks = [asyncio.create_task(_process_target(name)) for name in targets]
        judge_results = []
        if tasks:
            results = await asyncio.gather(*tasks)
            if iscaltoken:
                for result in results:
                    if isinstance(result, tuple) and len(result) == 2:
                        response, _ = result
                        judge_results.append(response)
                    else:
                        judge_results.append(result)
            else:
                judge_results.extend(results)
        else:
            results = []

        # Aggregate token statistics
        total_input = 0
        total_output = 0
        if iscaltoken:
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    _, token_dict = result
                    if token_dict:
                        total_input += token_dict.get("input_tokens", 0)
                        total_output += token_dict.get("output_tokens", 0)
        
        token_stats = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total": total_input + total_output
        } if (total_input + total_output) > 0 else None

        # return aggregate result dict-----------------------------------------
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
        
        # Prepare list of targets respecting upper_num
        targets = undetermined_files[:upper_num] if (upper_num > 0) else undetermined_files
        
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
                
                # Load background, tobetest, and entrypoint files
                background_file = sandbox_path / f"verify_{function_name_nosuffix}_background{file_ext}"
                tobetest_file = sandbox_path / f"verify_{function_name_nosuffix}_tobetest{file_ext}"
                entrypoint_file = sandbox_path / f"verify_{function_name_nosuffix}_entrypoint{file_ext}"
                
                # Check if all required files exist
                missing_files = []
                if not background_file.exists():
                    missing_files.append(str(background_file))
                if not tobetest_file.exists():
                    missing_files.append(str(tobetest_file))
                if not entrypoint_file.exists():
                    missing_files.append(str(entrypoint_file))
                
                if missing_files:
                    error_msg = f"Missing files for {function_name_nosuffix}: {', '.join(missing_files)}"
                    self._log_event({
                        "Content": "Refine verify code failed - missing files",
                        "function_name": function_name_nosuffix,
                        "missing_files": missing_files
                    })
                    raise ValueError(error_msg)
                
                # Load file contents
                background_code = self._load_file(background_file)
                tobetest_code = self._load_file(tobetest_file)
                entrypoint_code = self._load_file(entrypoint_file)
                
                # Load prompt and construct user prompt
                role_prompt = self._load_prompt(topic="general", prompt_file="VerifyCode_sandbox_single_repocode_refine_stand1")
                parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
                parts.append("BACKGROUND_CODE:\n")
                parts.append(background_code)
                parts.append("\n\n")
                parts.append("TOBETEST_CODE:\n")
                parts.append(tobetest_code)
                parts.append("\n\n")
                parts.append("ENTRYPOINT_CODE:\n")
                parts.append(entrypoint_code)
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
                
                # Call LLM to refine the code
                if iscaltoken:
                    response, token_dict = await asyncio.to_thread(self._send_chat_robust, user_model, user_prompt, max_iter=3, iscaltoken=iscaltoken)
                else:
                    response = await asyncio.to_thread(self._send_chat_robust, user_model, user_prompt, max_iter=3)
                    token_dict = None
                
                # Parse response and save refined background and entrypoint code
                parsed_response = self._parse_json(response)
                await asyncio.to_thread(self._save_code_file, response, anci=f"verify_{function_name_nosuffix}_background", code_dir=current_sandbox_dir, isarbi=True, content_tag="background_code")
                await asyncio.to_thread(self._save_code_file, response, anci=f"verify_{function_name_nosuffix}_entrypoint", code_dir=current_sandbox_dir, isarbi=True, content_tag="entrypoint_code")
                # Save refine_summary
                refine_summary = parsed_response.get("refine_summary", "")
                
                # Re-merge the files to update the unittest file
                try:
                    merged_content_parts = []
                    merged_content_parts.append(self._load_file(background_file))
                    merged_content_parts.append("\n\n")
                    merged_content_parts.append(self._load_file(tobetest_file))
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
                else:
                    return result_dict
        
        # Schedule workers and wait
        tasks = [asyncio.create_task(_process_refine_target(file_info)) for file_info in targets]
        refine_results = []
        refine_token_total_input = 0
        refine_token_total_output = 0
        if tasks:
            results = await asyncio.gather(*tasks)
            if iscaltoken:
                for result in results:
                    if isinstance(result, tuple) and len(result) == 2:
                        result_dict, token_dict = result
                        if result_dict is not None:
                            refine_results.append(result_dict)
                        if token_dict:
                            refine_token_total_input += token_dict.get("input_tokens", 0)
                            refine_token_total_output += token_dict.get("output_tokens", 0)
                    elif result is not None:
                        refine_results.append(result)
            else:
                refine_results = [r for r in results if r is not None]
        print(f"Refined {len(refine_results)} unit test files")
        
        # Step 3: Execute the refined unit test code and evaluate the results
        if not refine_results:
            raise ValueError("No refined files to execute.")
        
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
            iscaltoken=iscaltoken,
            concurr_num=concurr_num,
            exe_range=exe_range,
            islog=False
        )
        
        # Modify the output: change Content and add refine_summary to each judge_result entry
        execution_output["Content"] = "Single repocode unit test environment refinement judgement"
        judge_results = execution_output.get("judge_result", [])
        for judge_result in judge_results:
            target_file_name = judge_result.get("target_file_name")
            if target_file_name and target_file_name in refine_summary_map:
                judge_result["refine_summary"] = refine_summary_map[target_file_name]
        
        # Aggregate token statistics from refine and execution
        execution_token = execution_output.get("token_stats") or {}
        total_input = refine_token_total_input + execution_token.get("input_tokens", 0)
        total_output = refine_token_total_output + execution_token.get("output_tokens", 0)
        token_stats = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total": total_input + total_output
        } if (total_input + total_output) > 0 else None
        if token_stats:
            execution_output["token_stats"] = token_stats
        
        self._log_event(execution_output)
        return execution_output


    def Verify_code_generate_report(self, user_model: str, judge_token_stats: dict = None) ->dict:
        """
        Generate a compact report from log_unittest_xxx.jsonl file.
        Combines executability judgement and refinement judgement results.
        """
        # Step 1: Load the log file
        if self.current_log_file is None:
            raise ValueError("current_log_file is None. Cannot read execution results.")
        log_file_path = Path(self.current_log_file)
        if not log_file_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_file_path}")
        
        # Read and parse the jsonl file
        last_executability_entry = None
        last_refinement_entry = None
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
                    content_type = entry.get("Content")
                    if content_type == "Single repocode executability judgement":
                        last_executability_entry = entry
                    elif content_type == "Single repocode unit test environment refinement judgement":
                        last_refinement_entry = entry
                except json.JSONDecodeError:
                    continue
        
        if last_executability_entry is None:
            raise ValueError("No 'Single repocode executability judgement' entry found in log file.")

        # Step 2: Combine and clean up the initial log and refine log
        # Start with executability judgement results
        executability_judge_result = last_executability_entry.get("judge_result", [])
        refinement_judge_result = last_refinement_entry.get("judge_result", []) if last_refinement_entry else []
        
        # Create a dictionary to merge results, using target_file_name as key
        merged_dict = {}
        
        # First, add all executability results
        for result in executability_judge_result:
            target_file_name = result.get("target_file_name")
            if target_file_name:
                merged_dict[target_file_name] = {
                    "target_file_name": target_file_name,
                    "judge": result.get("judge", ""),
                    "errors": result.get("errors", "")
                }
        
        # Then, override with refinement results (if they exist)
        for result in refinement_judge_result:
            target_file_name = result.get("target_file_name")
            if target_file_name:
                merged_dict[target_file_name] = {
                    "target_file_name": target_file_name,
                    "judge": result.get("judge", ""),
                    "errors": result.get("errors", "")
                }
        
        # Convert dictionary to list
        merged_judge_result = list(merged_dict.values())

        # Step 3: Calculate the correct ratio
        total_count = len(merged_judge_result)
        correct_count = sum(1 for result in merged_judge_result if result.get("judge") == "correct")
        correct_ratio = (correct_count / total_count) if total_count > 0 else 0.0

        # Step 4: Save the report
        # Generate report file name: replace "log_unittest" with "report_unittest"
        report_file_name = log_file_path.name.replace("log_unittest", "report_unittest")
        report_file_path = log_file_path.parent / report_file_name
        
        # Create report dictionary
        report_dict = {
            "judge_result": merged_judge_result,
            "correct_ratio": correct_ratio
        }
        if judge_token_stats:
            report_dict["judge_token_stats"] = judge_token_stats
        
        # Write report file
        pretty_json = json.dumps(report_dict, ensure_ascii=False, indent=2)
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write(pretty_json + "\n\n")
        
        # Return result
        output = {
            "Content": "Generate unittest report",
            "LLM": user_model,
            "report_file": str(report_file_path),
            "total_functions": total_count,
            "correct_count": correct_count,
            "correct_ratio": correct_ratio
        }
        self._log_event(output)
        return output

    
class CodeIntegrator(QMBagent):

    async def Generate_integration_guidelines(self, user_model: str, current_sandbox_dir: str = None, repo_dir: str = None, upper_num: int = 0, iscaltoken: bool = False) ->dict:
       
        # Extract information from the target repo----------------------------------------
        if self.topic == "dmrg":
            prefix = "run"
        elif self.topic == "nnwf":
            prefix = "run"
        elif self.topic == "qcmb":
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
                parts.append(json.dumps(target_function, ensure_ascii=False))
                parts.append("\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            user_prompt = "".join(parts)
            # response = self._send_chat(user_model=user_model, user_prompt=user_prompt)
            if iscaltoken:
                response, token_dict = self._send_chat_robust(user_model=user_model, user_prompt=user_prompt, max_iter=3, iscaltoken=iscaltoken)
            else:
                response = self._send_chat_robust(user_model=user_model, user_prompt=user_prompt, max_iter=3)
                token_dict = None
            
            function_name_nosuffix = target_function_name.rsplit('.', 1)[0]
            save_jsonl_output = self._save_jsonl_file(response, anci=f"guideline_integrate_{function_name_nosuffix}", jsonl_dir=current_sandbox_dir, isarbi=True, issub=True, subname=function_name_nosuffix)
            
            # Collect token statistics
            if iscaltoken and token_dict:
                if 'total_input' not in locals():
                    total_input = 0
                    total_output = 0
                total_input += token_dict.get("input_tokens", 0)
                total_output += token_dict.get("output_tokens", 0)

        # Aggregate token statistics
        token_stats = None
        if iscaltoken and 'total_input' in locals():
            token_stats = {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total": total_input + total_output
            } if (total_input + total_output) > 0 else None

        output = {"Content": "Single run function integration -- guidelines generation", "LLM": user_model, "repo_dir": repo_dir, "sandbox_dir": current_sandbox_dir}
        if token_stats:
            output["token_stats"] = token_stats
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
                # Support both 'element_functions' (plural) and 'element_function' (singular)
                element_functions_names = level_dict.get('element_functions') or level_dict.get('element_function')
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

                response = self._send_chat(user_model=user_model, user_prompt=user_prompt)
                save_code_output = self._save_code_file(response, anci=f"integrate_{target_run_name}_level{level_count+1}", code_dir=sandbox_path / target_run_name, isarbi=True)
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
                    # Support both 'element_functions' (plural) and 'element_function' (singular)
                    element_functions_names = level_dict.get('element_functions') or level_dict.get('element_function')
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
                    save_code_output = await asyncio.to_thread(self._save_code_file, response, anci=f"integrate_{target_run_name}_{level_index}", code_dir=sandbox_path / target_run_name, isarbi=True)
                    return save_code_output

            tasks = [asyncio.create_task(_process_level(ld)) for ld in level_dicts]
            if tasks:
                await asyncio.gather(*tasks) 
            # end of concurrency for levels under 1 run function          

        output = {"Content": "Single run function integration -- integration code generation", "LLM": user_model, "repo_dir": repo_dir, "sandbox_dir": current_sandbox_dir}
        self._log_event(output)
        return output


    async def Generate_integration_code_split_concurr(self, user_model: str, current_sandbox_dir: str = None, repo_dir: str = None, upper_num: int = 0, concurr_num: int = 5, iscaltoken: bool = False) -> dict:
        
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
                    # Support both 'element_functions' (plural) and 'element_function' (singular)
                    element_functions_names = level_dict.get('element_functions') or level_dict.get('element_function')
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
                    # If failed, determine from main file
                    if file_ext is None:
                        for ext in ['.py', '.jl']:
                            main_path = repo_path / f"main{ext}"
                            if main_path.exists():
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
                    role_prompt = self._load_prompt(topic="general", prompt_file="IntegrateCode_sandbox_single_level_environment_generation_stand1")
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
                    # Add FILE_EXT parameter (remove leading dot if present)
                    file_ext_str = file_ext.lstrip('.') if file_ext else ""
                    if file_ext_str:
                        parts.append(f"FILE_EXT:\n{file_ext_str}\n\n")
                    parts.append("-- END AGGREGATED INPUT --\n")
                    user_prompt = "".join(parts)
                    if iscaltoken:
                        response, token_dict = await asyncio.to_thread(self._send_chat_robust, user_model=user_model, user_prompt=user_prompt, max_iter=3, iscaltoken=iscaltoken)
                    else:
                        response = await asyncio.to_thread(self._send_chat_robust, user_model=user_model, user_prompt=user_prompt, max_iter=3)
                        token_dict = None

                    # Step 2.3.4: Save the integration code parts and aggregate the complete integration code
                    await asyncio.to_thread(self._save_code_file, response, anci=f"integrate_{target_run_name}_{level_index}_run", code_dir=sandbox_path / target_run_name, isarbi=True, content_tag="run_code", file_ext=file_ext)
                    await asyncio.to_thread(self._save_code_file, response, anci=f"integrate_{target_run_name}_{level_index}_entrypoint", code_dir=sandbox_path / target_run_name, isarbi=True, content_tag="entrypoint_code", file_ext=file_ext)

                    # Clean up the unused imports in run code and entrypoint code
                    code_editor.delete_unused_imports(sandbox_path / target_run_name, count_type_annotations=True, file_names=[f"integrate_{target_run_name}_{level_index}_run{file_ext}", f"integrate_{target_run_name}_{level_index}_entrypoint{file_ext}"])

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
                        
                    if iscaltoken:
                        return target_run_name, level_index, token_dict
                    else:
                        return target_run_name, level_index

            tasks = [asyncio.create_task(_process_level(ld)) for ld in level_dicts]
            if tasks:
                results = await asyncio.gather(*tasks)
            else:
                results = []
            # end of concurrency for levels under 1 run function          
        
        # Aggregate token statistics
        total_input = 0
        total_output = 0
        if iscaltoken:
            for result in results:
                if isinstance(result, tuple) and len(result) == 3:
                    _, _, token_dict = result
                    if token_dict:
                        total_input += token_dict.get("input_tokens", 0)
                        total_output += token_dict.get("output_tokens", 0)
        
        token_stats = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total": total_input + total_output
        } if (total_input + total_output) > 0 else None

        output = {"Content": "Single run function integration -- integration code generation", "LLM": user_model, "repo_dir": repo_dir, "sandbox_dir": current_sandbox_dir}
        if token_stats:
            output["token_stats"] = token_stats
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

                response = self._send_chat(user_model=user_model, user_prompt=user_prompt)
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
    

    async def Integrate_code_single_run_execution_concurr(self, user_model: str, current_sandbox_dir: str = None, upper_num: int = 0, concurr_num: int = 5, iscaltoken: bool = False):

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
                role_prompt = self._load_prompt(topic="general", prompt_file="IntegrateCode_sandbox_single_run_execution_evaluation_stand1")
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
                if iscaltoken:
                    response, token_dict = await asyncio.to_thread(self._send_chat_robust, user_model=user_model, user_prompt=user_prompt, max_iter=3, iscaltoken=iscaltoken)
                else:
                    response = await asyncio.to_thread(self._send_chat_robust, user_model=user_model, user_prompt=user_prompt, max_iter=3)
                    token_dict = None
                response = self._parse_json(response)
                if iscaltoken:
                    return response, token_dict
                else:
                    return response
            
            tasks = [asyncio.create_task(_process_target_file(tf)) for tf in target_files]
            run_token_total_input = 0
            run_token_total_output = 0
            if tasks:
                results = await asyncio.gather(*tasks)
                if iscaltoken:
                    for result in results:
                        if isinstance(result, tuple) and len(result) == 2:
                            response, token_dict = result
                            judge_results.append(response)
                            if token_dict:
                                run_token_total_input += token_dict.get("input_tokens", 0)
                                run_token_total_output += token_dict.get("output_tokens", 0)
                        else:
                            judge_results.append(result)
                else:
                    judge_results.extend(results)

            # Create and log output for the current run function subdirectory
            output = {
                "Content": f"Integration test of {run_function_dir.name}",
                "LLM": user_model,
                "run_function_dir": str(run_function_dir),
                "judge_results": judge_results,
            }
            if iscaltoken and (run_token_total_input + run_token_total_output) > 0:
                output["token_stats"] = {
                    "input_tokens": run_token_total_input,
                    "output_tokens": run_token_total_output,
                    "total": run_token_total_input + run_token_total_output
                }
            self._log_event(output)
            all_run_outputs.append(output)

        return all_run_outputs


    def Integrate_code_generate_report(self, user_model: str, current_sandbox_dir: str = None, judge_token_stats: dict = None):
        """
        Generate a compact report from log_integtest_xxx.jsonl file.
        Analyzes integration test results for each run function and generates error analysis.
        """
        # Step 1: Load the log file
        if self.current_log_file is None:
            raise ValueError("current_log_file is None. Cannot read execution results.")
        log_file_path = Path(self.current_log_file)
        if not log_file_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_file_path}")
        
        # Read and parse the jsonl file
        run_function_entries = {}
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
                    content_type = entry.get("Content", "")
                    if content_type.startswith("Integration test of "):
                        run_function_name = content_type.replace("Integration test of ", "")
                        run_function_entries[run_function_name] = entry
                except json.JSONDecodeError:
                    continue
        
        if not run_function_entries:
            raise ValueError("No 'Integration test of' entries found in log file.")
        if current_sandbox_dir is None:
            raise ValueError("current_sandbox_dir is None. Cannot load guideline files.")
        sandbox_path = Path(current_sandbox_dir)
        if not sandbox_path.exists():
            raise FileNotFoundError(f"Sandbox directory not found: {sandbox_path}")

        # Step 2: Identify the highest successful level for each run function
        run_function_data = {}
        for run_name, entry in run_function_entries.items():
            judge_results = entry.get("judge_results", [])
            
            # Supplement level field if missing
            for result in judge_results:
                if "level" not in result:
                    target_file_name = result.get("target_file_name", "")
                    match = re.search(r'level-(\d+)', target_file_name)
                    if match:
                        result["level"] = f"level-{match.group(1)}"
            
            # Find highest successful level and highest level
            correct_levels = []
            all_levels = []
            for result in judge_results:
                level_str = result.get("level", "")
                match = re.search(r'level-(\d+)', level_str)
                if match:
                    level_num = int(match.group(1))
                    all_levels.append(level_num)
                    if result.get("judge") == "correct":
                        correct_levels.append(level_num)
            
            highest_correct_level = max(correct_levels) if correct_levels else 0
            highest_level = max(all_levels) if all_levels else 0
            total_levels = len(judge_results)
            
            # Check if all levels are correct: highest correct level equals highest level
            all_correct = (highest_correct_level == highest_level) and (highest_level > 0)
            
            run_function_data[run_name] = {
                "entry": entry,
                "judge_results": judge_results,
                "highest_correct_level": highest_correct_level,
                "total_levels": total_levels,
                "all_correct": all_correct
            }

        # Step 3: Create initial report dictionary structure
        report_dict = {
            "run_functions": [],
            "ave_correct_ratio": None
        }
        
        for run_name, data in run_function_data.items():
            if data["all_correct"]:
                run_function_dict = {
                    "run_function": run_name,
                    "error_analysis": "None",
                    "correct_ratio": 1.0
                }
            else:
                run_function_dict = {
                    "run_function": run_name,
                    "error_analysis": None,
                    "correct_ratio": None
                }
            report_dict["run_functions"].append(run_function_dict)

        # Step 4: Process all run functions to extract successful element_functions and analyze errors
        for run_name, data in run_function_data.items():
            # Find the run function dict in report_dict
            run_function_dict = None
            for rf_dict in report_dict["run_functions"]:
                if rf_dict["run_function"] == run_name:
                    run_function_dict = rf_dict
                    break
            
            if run_function_dict is None:
                continue
            
            highest_correct_level = data["highest_correct_level"]
            failed_level = highest_correct_level + 1
            
            # Load guideline file
            guideline_jsonl_path = sandbox_path / run_name / f"guideline_integrate_{run_name}.jsonl"
            guideline_jsonl = self._load_file(guideline_jsonl_path)
            guideline_obj = json.loads(guideline_jsonl, strict=False)
            integration_guideline = guideline_obj.get('integration_guideline', [])
            
            # Extract successful level guideline information (for all run functions)
            successful_level_info = None
            for level_dict in integration_guideline:
                if f"level-{highest_correct_level}" in level_dict:
                    successful_level_info = {
                        "level": f"level-{highest_correct_level}",
                        "description": level_dict[f"level-{highest_correct_level}"],
                        "element_functions": level_dict.get("element_functions", [])
                    }
                    break
            
            # Add successful_element_functions to report (for all run functions)
            if successful_level_info:
                run_function_dict["successful_element_functions"] = successful_level_info.get("element_functions", [])
            
            # For run functions that are all correct, skip error analysis
            if data["all_correct"]:
                continue
            
            # Extract failed level guideline information (only for not all correct)
            failed_level_info = None
            for level_dict in integration_guideline:
                if f"level-{failed_level}" in level_dict:
                    failed_level_info = {
                        "level": f"level-{failed_level}",
                        "description": level_dict[f"level-{failed_level}"],
                        "element_functions": level_dict.get("element_functions", [])
                    }
                    break
            
            # Load failed level code file
            failed_code = None
            for ext in ['.py', '.jl']:
                code_file_path = sandbox_path / run_name / f"integrate_{run_name}_level-{failed_level}_integtest{ext}"
                if code_file_path.exists():
                    failed_code = self._load_file(code_file_path)
                    break
            
            # Extract error information from judge_results
            failed_result = None
            for result in data["judge_results"]:
                if result.get("level") == f"level-{failed_level}":
                    failed_result = result
                    break
            errors = failed_result.get("errors", "")
            stderr = failed_result.get("stderr", "")
            error_info = {
                "errors": errors,
                "stderr": stderr
            }
            
            # Call LLM for error analysis
            role_prompt = self._load_prompt(topic="general", prompt_file="IntegrateCode_error_analysis_stand1")
            parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            parts.append("SUCCESSFUL_LEVEL_GUIDELINE:\n")
            parts.append(json.dumps(successful_level_info, ensure_ascii=False, indent=2))
            parts.append("\n\n")
            parts.append("FAILED_LEVEL_GUIDELINE:\n")
            parts.append(json.dumps(failed_level_info, ensure_ascii=False, indent=2))
            parts.append("\n\n")
            parts.append("FAILED_LEVEL_CODE:\n")
            parts.append(json.dumps(failed_code, ensure_ascii=False))
            parts.append("\n\n")
            parts.append("ERROR_INFO:\n")
            parts.append(json.dumps(error_info, ensure_ascii=False, indent=2))
            parts.append("\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            user_prompt = "".join(parts)
            response = self._send_chat_robust(user_model, user_prompt, max_iter=3)
            parsed_response = self._parse_json(response)
            error_analysis = parsed_response.get("error_analysis", "")
            
            run_function_dict["error_analysis"] = error_analysis
            
            # Calculate correct_ratio
            run_function_dict["correct_ratio"] = highest_correct_level / data["total_levels"] if data["total_levels"] > 0 else 0.0

        # Step 5: Calculate average correct ratio and save the report
        correct_ratios = [rf.get("correct_ratio", 0.0) for rf in report_dict["run_functions"]]
        report_dict["ave_correct_ratio"] = sum(correct_ratios) / len(correct_ratios) if correct_ratios else 0.0
        if judge_token_stats:
            report_dict["judge_token_stats"] = judge_token_stats
        
        # Generate report file name: replace "log_integtest" with "report_integtest"
        report_file_name = log_file_path.name.replace("log_integtest", "report_integtest")
        report_file_path = log_file_path.parent / report_file_name
        
        # Write report file
        pretty_json = json.dumps(report_dict, ensure_ascii=False, indent=2)
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write(pretty_json + "\n\n")
        
        # Return result
        output = {
            "Content": "Generate integration test report",
            "LLM": user_model,
            "report_file": str(report_file_path),
            "total_run_functions": len(report_dict["run_functions"]),
            "ave_correct_ratio": report_dict["ave_correct_ratio"]
        }
        self._log_event(output)
        return output

    
class RubricsGrader(QMBagent):

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
        role_prompt = self._load_prompt(topic="general", prompt_file="GradeRubrics_scorecard_stand1")
        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        if paper_content is not None:
            parts.append(f"PaperFile (paper content):\n")
            parts.append(json.dumps(paper_content, ensure_ascii=False))
            parts.append("\n\n")
        if code_content is not None:
            parts.append(f"CodeFile (code content):\n")
            parts.append(json.dumps(code_content, ensure_ascii=False))
            parts.append("\n\n")
        if scorecard is not None:
            parts.append(f"scorecard:\n")
            parts.append(json.dumps(scorecard, ensure_ascii=False))
            parts.append("\n\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)

        response = self._send_chat_robust(user_model=user_model, user_prompt=user_prompt, max_iter=3)
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

    
    # def Test_LimitingCases(self, code_file: str, tag_name: str, author_model: str, judge_model: str):







