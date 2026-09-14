# main.py
from openai import OpenAI
import time
import random
from pathlib import Path
import json
import base64
from typing import Optional, Dict
import fitz
import re
from langchain_openai import ChatOpenAI
import os
import sys
import asyncio
import ast
from decimal import Decimal, InvalidOperation
from mcp_use import MCPAgent, MCPClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Import from this project
from config import OPENROUTER_API_KEY
from config import get_model_settings
from utils import MCP_toolbox, run_program, code_editor

# Check the API key
if not OPENROUTER_API_KEY:
    raise ValueError("OpenRouter API key is not set. Please set the OPENROUTER_API_KEY environment variable.")


# File management
def create_run_log(pdf_name: str, subplotname: str = None, output_dir: str = "../logs", issubfile: bool = False, iscreatlog: bool = True, headname_dir: str = None, headname_log: str = None, idname_dir: str = None, idname_log: str = None) -> str:
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
            out_dir = Path(output_dir)
            safe_subplotname = None
        elif subplotname is None:
            out_dir = Path(output_dir) / f"{headname_dir}_{safe_name}_{idname_dir}"
            safe_subplotname = None
        else:
            safe_subplotname = subplotname.replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_')
            out_dir = Path(output_dir) / f"{headname_dir}_{safe_name}_{safe_subplotname}_{idname_dir}"
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
        prompt_dir: str = "../prompts",
        output_dir: str = "../Paper_docu/Content_papers",
        current_log_file: str = None,
        current_code_dir: str = "../Paper_docu/Codes",
    ):
        self.api_key = api_key
        self.prompt_dir = prompt_dir
        self.output_dir = output_dir
        self.current_log_file = current_log_file
        self.current_code_dir = current_code_dir

    # IO unit------------------------------------------------
    def _load_prompt(self, prompt_file: str) -> str:
        """
        Load a prompt from a text file.
        """
        # Use pathlib for robust path handling
        prompt_path = Path(self.prompt_dir) / (prompt_file + ".txt")
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
        if not isinstance(text, str):
            return text

        match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = text # If no wrapper, use the text as is

        try:
            return json.loads(json_str)
        except Exception:
            return text
        
    
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
            
    def _save_code_file(self, response: str, anci: str = "", code_dir: str = None, isarbi: bool = False):

        try:
            content = response
            json_fence = re.compile(r"```json\s*\r?\n(.*?)\r?\n```", re.DOTALL | re.IGNORECASE)
            generic_fence = re.compile(r"```\s*\r?\n(.*?)\r?\n```", re.DOTALL)

            while True:
                match = json_fence.search(content) or generic_fence.search(content)
                if not match:
                    break
                content = match.group(1).strip()

            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content
        except Exception as e:
            self._log_event({"Content": "Save code failed-type 1", "reason": "response is not valid JSON", "error": str(e), "response": response})
            return None

        try:
            subplot_name = data.get("subplot_name")
            code_type = data.get("code_type")
            code_content = data.get("code_content")
        except Exception as e:
            self._log_event({"Content": "Save code failed", "reason": "response JSON missing fields", "error": str(e)})
            return None

        if not code_type or not code_content:
            self._log_event({"Content": "Save code failed", "reason": "missing code_type or code_content"})
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
            match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = response
            if isinstance(json_str, str):
                data = json.loads(json_str)
            else:
                data = json_str
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
    def _send_chat(self, user_model: str, user_prompt: str) -> str:
        """
        Send chat messages to the LLM and get the response.
        """
        model_name = user_model
        settings = get_model_settings(model_name)

        QMB_author_prompt = self._load_prompt("QMB_author")
        message_to_send = [
            {"role": "system", "content": QMB_author_prompt},
            {"role": "user", "content": user_prompt}
        ]

        model_params = {
            "temperature": settings.get("temperature"),
            "top_p": settings.get("top_p"),
            # add other parameters as needed
        }

        try:
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
            response = client.chat.completions.create(
                model = model_name,
                messages = message_to_send,
                **model_params
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"calling model {model_name} failed with error: {str(e)}"
        

    async def _send_chat_through_mcp_use(self, user_model: str, user_prompt: str) -> str:
        response = await MCP_toolbox.send_chat_through_mcp_use(user_model=user_model, user_prompt=user_prompt)
        return response
        
    async def _send_chat_through_mcp_use_filesystem_dynamic(self, user_model: str, user_prompt: str, fs_target_dir: str, reference_dir_dict: Optional[Dict[str, str]] = None) -> str:
        response = await MCP_toolbox.send_chat_through_mcp_use_filesystem_dynamic(user_model=user_model, user_prompt=user_prompt, fs_target_dir=fs_target_dir, references=reference_dir_dict)
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

    def _extract_tex(self, tex_path: str = "../Paper_dataset", tex_name: str = "None") -> dict:
        
        papers_dir = Path(tex_path)
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
        response = self._send_chat(user_model, user_prompt)
        parsed_response = self._parse_json(response)

        output = {"Content": "Basic info", "LLM": user_model, "PDF_name": PDF_info['path'], "response": parsed_response}
        self._log_event(output)
        return output
    
    
    def Extract_figure_captions(self, user_model: str, PDF_info: dict) -> dict:
        
        role_prompt = self._load_prompt("ExtractFigureCaptions")
        pdf_json = json.dumps(PDF_info, ensure_ascii=False)
        user_prompt = role_prompt + "\n\n" + "PDF_INFO_JSON:\n" + pdf_json
        response = self._send_chat(user_model, user_prompt)
        parsed_response = self._parse_json(response)

        output = {"Content": "Figure info from captions", "LLM": user_model, "PDF_name": PDF_info['path'], "response": parsed_response}
        self._log_event(output)
        return output


    def Plan_task_by_text(self, user_model: str, subplot_name: str, PDF_info: dict = None, User_requests: str = "None", Paper_basic_info: dict = None, Figure_info: dict = None) -> dict:
        """Compose a prompt from any subset of the provided inputs and ask the LLM to plan tasks.

        The function only includes sections that are provided (non-None).
        """
        # role_prompt = self._load_prompt("PlanTaskByText")
        role_prompt = self._load_prompt("PlanTaskByText_disabled")


        parts = [role_prompt, "\n Focus on: ", subplot_name, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
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

        response = self._send_chat(user_model, user_prompt)
        parsed_response = self._parse_json(response)

        output = {"Content": "Implementation plan by text", "LLM": user_model, "subplot_name": parsed_response["subplot_name"], "User_requests": parsed_response["User_requests"]}
        self._log_event(output)
        return output
    

class CodeGenerator(QMBagent):

    def Generate_code_by_plan(self, user_model: str, Plan_info: dict, PDF_info: dict = None) -> dict:

        # role_prompt = self._load_prompt("GenerateCodeByPlan")
        role_prompt = self._load_prompt("GenerateCodeByPlan_stand1")

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
        response = self._send_chat(user_model, user_prompt)
        
        save_code_output = self._save_code_file(response)
        subplot_name = save_code_output.get("subplot_name") if save_code_output else None
        code_file = save_code_output.get("code_file") if save_code_output else None
        output = {"Content": "Generate code", "LLM": user_model, "subplot_name": subplot_name, "code_file": code_file, "response": response}

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
    

    async def Generate_code_by_plan_mcp_use(self, user_model: str, Plan_info: dict = None, PDF_info: dict = None) -> dict:

        # role_prompt = self._load_prompt("GenerateCodeByPlan_mcp_use_WebSearch")
        # role_prompt = self._load_prompt("GenerateCodeByPlan_mcp_use_LocalSearch")
        role_prompt = self._load_prompt("GenerateCodeByPlan_mcp_use_LocalSearch_stand1")

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
        response = await self._send_chat_through_mcp_use(user_model, user_prompt)
        
        save_code_output = self._save_code_file(response, anci = "code_LLM", isarbi=True)
        subplot_name = save_code_output.get("subplot_name") if save_code_output else None
        code_file = save_code_output.get("code_file") if save_code_output else None
        code_content = save_code_output.get("code_content") if save_code_output else None
        output = {"Content": "Generated code", "LLM": user_model, "subplot_name": subplot_name, "code_file": code_file, "code_content": code_content}

        self._log_event(output)
        return output
    

    async def Refine_code_obey_rules(self, user_model: str, code_file: str, max_iter: int = 3) -> dict:

        # Refine loop structure
        flag_obey = False
        count_iter = 0

        # prepare materials
        if code_file is not None:
            code_former = self._load_file(code_file)
            if code_file.lower().endswith(".jl"):
                package_range = self._load_file("../Project.toml")
        else:
            code_path_py = Path(self.current_code_dir) / "code_LLM.py"
            code_path_jl = Path(self.current_code_dir) / "code_LLM.jl"
            if code_path_py.exists():
                code_former = self._load_file(code_path_py)
            elif code_path_jl.exists():
                code_former = self._load_file(code_path_jl)
                package_range = self._load_file("../Project.toml")
            else:
                raise FileNotFoundError("code_LLM.py or code_LLM.jl not found in the current code directory.")

        # begin iteration
        while flag_obey == False and count_iter < max_iter:
            count_iter += 1

            # Refine step 1: check and correct basic errors (coding language usage, package usage, syntex usage)-------------
            print(f"refine code to obey rules - iteration {count_iter} - basic errors checking...")
            role_prompt = self._load_prompt("RefineCode_obey_rules_basicerrors_stand1")
            parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            if code_former is not None:
                parts.append("CODE_FORMER:\n")
                parts.append(json.dumps(code_former, ensure_ascii=False))
                parts.append("\n\n")
            if package_range is not None:
                parts.append("PACKAGE_RANGE:\n")
                parts.append(json.dumps(package_range, ensure_ascii=False))
                parts.append("\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            user_prompt = "".join(parts)
            # process by LLM
            response = self._send_chat(user_model, user_prompt)
            parsed_response = self._parse_json(response)
            # save and log
            self._save_code_file(response=response, anci=f"code_LLM_basiccheck_{count_iter}", isarbi=True)
            self._log_event({"Content": "Refine code - basic errors", "LLM": user_model, "iteration": count_iter, "response": parsed_response})
            code_former = parsed_response.get("code_content")
            
            # Refine step 2: check and correct element function definitions--------------------------
            print(f"refine code to obey rules - iteration {count_iter} - element functions checking...")
            role_prompt = self._load_prompt("RefineCode_obey_rules_elementfunc_stand1") 
            parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            if code_former is not None:
                parts.append("CODE_FORMER:\n")
                parts.append(json.dumps(code_former, ensure_ascii=False))
                parts.append("\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            user_prompt = "".join(parts)
            # process by LLM
            response = self._send_chat(user_model, user_prompt)
            parsed_response = self._parse_json(response)
            # save and log
            self._save_code_file(response=response, anci=f"code_LLM_elementcheck_{count_iter}", isarbi=True)
            self._log_event({"Content": "Refine code - element functions", "LLM": user_model, "iteration": count_iter, "response": parsed_response})
            code_former = parsed_response.get("code_content")

            # Refine step 3: check and correct task function definitions--------------------------
            print(f"refine code to obey rules - iteration {count_iter} - task functions checking...")
            role_prompt = self._load_prompt("RefineCode_obey_rules_taskfunc_stand1") 
            parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            if code_former is not None:
                parts.append("CODE_FORMER:\n")
                parts.append(json.dumps(code_former, ensure_ascii=False))
                parts.append("\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            user_prompt = "".join(parts)
            # process by LLM
            response = self._send_chat(user_model, user_prompt)
            parsed_response = self._parse_json(response)
            # save and log
            self._save_code_file(response=response, anci=f"code_LLM_taskcheck_{count_iter}", isarbi=True)
            self._log_event({"Content": "Refine code - task functions", "LLM": user_model, "iteration": count_iter, "response": parsed_response})
            code_former = parsed_response.get("code_content")


        return 0


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

        response = await self._send_chat_through_mcp_use_filesystem_dynamic(
            user_model=user_model,
            user_prompt=user_prompt,
            fs_target_dir=fs_target_dir,
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
    
    
    async def Generate_repo_by_codefile_hardcode(self, user_model: str, code_file: str, fs_target_dir: str) -> dict:
        # Read code file
        original_code = self._load_file(Path(code_file))

        # Construct repository by hardcode
        code_editor.split_functions(code_file, fs_target_dir)

        # Generate README via LLM
        role_prompt = self._load_prompt("GenerateRepo_README_only_by_codefile_stand1")
        parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
        if original_code is not None:
            parts.append("ORIGINAL_CODE:\n")
            parts.append(json.dumps(original_code, ensure_ascii=False))
            parts.append("\n\n")
        parts.append("-- END AGGREGATED INPUT --\n")
        user_prompt = "".join(parts)

        response = self._send_chat(user_model, user_prompt)
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

        output = {"Content": "Generated repository", "LLM": user_model, "repo_dir": fs_target_dir, "README": readme_text}
        
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
            response = await self._send_chat_through_mcp_use_filesystem_dynamic(
                user_model=user_model,
                user_prompt=user_prompt,
                fs_target_dir=current_sandbox_dir,
                reference_dir_dict={"libs": CodeVerifier_library_dir},
            )
        else:
            response = await self._send_chat_through_mcp_use_filesystem_dynamic(
                user_model=user_model,
                user_prompt=user_prompt,
                fs_target_dir=current_sandbox_dir,
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

                response = await self._send_chat_through_mcp_use_filesystem_dynamic(
                    user_model=user_model,
                    user_prompt=user_prompt,
                    fs_target_dir=current_sandbox_dir,
                    reference_dir_dict={"libs": CodeVerifier_library_dir},
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
    

    async def Verify_code_single_repocode_execution_concurr(self, user_model: str, current_sandbox_dir: str = None, upper_num: int = 0, concurr_num: int = 5) ->dict:

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

        # Iterative execution (concurrent)-------------------------------------------------
        # Prepare list of targets respecting upper_num
        targets = files_list[:upper_num] if (upper_num > 0) else files_list

        sem = asyncio.Semaphore(concurr_num)

        async def _process_target(target_file_name: str):
            async with sem:
                target_file_path = sandbox_path / target_file_name
                target_file = self._load_file(target_file_path)
                print(f"verify code: {target_file_name} executing...")

                # run_julia_file is blocking; run in thread
                result_target_file = await asyncio.to_thread(run_program.run_julia_file, target_file_path, timeout=600)
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

                # _send_chat is blocking; run in thread
                response = await asyncio.to_thread(self._send_chat, user_model, user_prompt)
                response = self._parse_json(response)
                return response

        # schedule workers and wait
        tasks = [asyncio.create_task(_process_target(name)) for name in targets]
        judge_results = []
        if tasks:
            results = await asyncio.gather(*tasks)
            judge_results.extend(results)

        # return aggregate result dict-----------------------------------------
        output = {
            "Content": "Single repocode executability judgement",
            "LLM": user_model,
            "repo_dir": current_sandbox_dir,
            "judge_result": judge_results,
        }
        self._log_event(output)
        return output

    
class CodeIntegrator(QMBagent):

    async def Generate_integration_guidelines(self, user_model: str, current_sandbox_dir: str = None, repo_dir: str = None, upper_num: int = 0) ->dict:
       
        # Extract information from the target repo----------------------------------------
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
            role_prompt = self._load_prompt("IntegrateCode_sandbox_generate_guidelines_details_stand1")
            parts = [role_prompt, "\n\n-- BEGIN AGGREGATED INPUT --\n"]
            if target_function is not None:
                parts.append(f"RUN_FUNCTION:\n")
                parts.append(json.dumps(target_function, ensure_ascii=False))
                parts.append("\n\n")
            parts.append("-- END AGGREGATED INPUT --\n")
            user_prompt = "".join(parts)
            response = self._send_chat(user_model=user_model, user_prompt=user_prompt)
            
            function_name_nosuffix = target_function_name.rsplit('.', 1)[0]
            save_jsonl_output = self._save_jsonl_file(response, anci=f"integrate_{function_name_nosuffix}", jsonl_dir=current_sandbox_dir, isarbi=True, issub=True, subname=function_name_nosuffix)
            

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
                    files_list.append(f)
            
            files_list = sorted(files_list)
            judge_results = []

            target_files = files_list[:upper_num] if upper_num > 0 else files_list
            sem = asyncio.Semaphore(concurr_num)

            async def _process_target_file(target_file_path: Path):
                target_file_name = target_file_path.name
                target_file = self._load_file(target_file_path)
                print(f"Executing code in {run_function_dir.name}: {target_file_name}...")
                
                result_target_file = await asyncio.to_thread(run_program.run_julia_file, target_file_path, timeout=600)
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

                response = await asyncio.to_thread(self._send_chat, user_model=user_model, user_prompt=user_prompt)
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
        paper_dir = Path(paper_path)
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
        role_prompt = self._load_prompt("GradeRubrics_scorecard_stand1")
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

        response = self._send_chat(user_model=user_model, user_prompt=user_prompt)
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

    







