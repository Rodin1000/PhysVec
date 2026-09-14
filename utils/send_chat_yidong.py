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
from config import OPENROUTER_API_KEY, YIDONG_API_KEY, YIDONG_API_KEY_1, YIDONG_API_KEY_2, YIDONG_API_KEY_3, YIDONG_API_KEY_4
from config import get_model_settings

# original MODEL_MAX_TOKENS by the limit of the model itself
# MODEL_MAX_TOKENS = {
#     "gemini-3-flash-preview": 65536,
#     "gemini-3-flash-preview-thinking": 65536,
#     "gemini-3-pro-preview": 65536,
#     "gemini-2.5-pro-preview-06-05": 65536,
#     "gemini-2.5-flash-nothinking": 65536,
#     "gpt-5.1-chat": 16384,
#     "gpt-5.1-2025-11-13": 16384,
#     "gpt-4o-2024-11-20": 16384,
#     "claude-sonnet-4-5-20250929": 64000,
#     "grok-4-fast-reasoning": 250000,
# }

MODEL_MAX_TOKENS = {
    "gemini-3-flash-preview": 8000,
    "gemini-3-flash-preview-thinking": 20000,
    "gemini-3-pro-preview": 30000,
    "gemini-2.5-pro-preview-06-05": 65536,
    "gemini-2.5-flash-nothinking": 30000,
    "gemini-2.5-flash-lite-thinking": 30000,
    "gpt-5.1-chat": 16384,
    "gpt-5.1-2025-11-13": 10000,
    "gpt-5.2": 10000,
    "gpt-4o-2024-11-20": 10000,
    "claude-sonnet-4-5-20250929": 15000,
    "claude-opus-4-5-20251101": 15000,
    "claude-3-haiku-20240307": 15000,
    "claude-3-5-sonnet-20241022": 15000,
    "claude-sonnet-4-20250514": 15000,
    "grok-4-fast-reasoning": 40000,
    "grok-4-fast-non-reasoning": 30000,
    "deepseek-r1": 20000,
    "deepseek-v3": 10000,
    "qwen3-30b-a3b-instruct-2507": 15000,
    "qwen3-next-80b-a3b-instruct": 15000,
    "qwen-plus": 15000,
    "qwen3.5-plus": 15000,
    "qwen3-max": 15000,
}

MODEL_THINKING_BUDGET = {
    "gemini-3-flash-preview": "minimal",    # "low"
    "gemini-3-flash-preview-thinking": "medium"
}

MODEL_API_KEY = {
    "gemini-3-flash-preview": YIDONG_API_KEY_1,
    "gemini-3-flash-preview-thinking": YIDONG_API_KEY_1,
    "gemini-3-pro-preview": YIDONG_API_KEY_1,
    "gemini-2.5-pro-preview-06-05": YIDONG_API_KEY_1,
    "gemini-2.5-flash-nothinking": YIDONG_API_KEY_1,
    "gemini-2.5-flash-lite-thinking": YIDONG_API_KEY_1,
    "gpt-5.1-chat": YIDONG_API_KEY_2,
    "gpt-5.1-2025-11-13": YIDONG_API_KEY_2,
    "gpt-5.2": YIDONG_API_KEY_2,
    "gpt-4o-2024-11-20": YIDONG_API_KEY_2,
    "claude-sonnet-4-5-20250929": YIDONG_API_KEY_3,
    "claude-opus-4-5-20251101": YIDONG_API_KEY_3,
    "claude-3-haiku-20240307": YIDONG_API_KEY_3,
    "claude-3-5-sonnet-20241022": YIDONG_API_KEY_3,
    "claude-sonnet-4-20250514": YIDONG_API_KEY_3,
    "grok-4-fast-reasoning": YIDONG_API_KEY_2,
    "grok-4-fast-non-reasoning": YIDONG_API_KEY_2,
    "deepseek-r1": YIDONG_API_KEY_2,
    "deepseek-v3": YIDONG_API_KEY_2,
    "qwen3-30b-a3b-instruct-2507": YIDONG_API_KEY_4,
    "qwen3-next-80b-a3b-instruct": YIDONG_API_KEY_4,
    "qwen-plus": YIDONG_API_KEY_4,
    "qwen3.5-plus": YIDONG_API_KEY_4,
    "qwen3-max": YIDONG_API_KEY_4,
}

NON_STREAM_MODELS = [
    "gemini-3-flash-preview",
]

REASONING_MODELS = [
    "gemini-3-flash-preview-thinking",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-nothinking",
    "grok-4-fast-reasoning",
    "deepseek-r1",
    "claude-3-5-sonnet-20241022",
    "claude-sonnet-4-20250514",
]


def _parse_non_stream_response(response: requests.Response, iscaltoken: bool) -> Tuple[str, int, Optional[Dict]]:
    """
    Parse non-stream response to extract text and token information.
    
    Args:
        response: requests.Response object
        iscaltoken: Flag for token calculation
    
    Returns:
        Tuple of (response_text, status_code, token_dict)
    """
    status_code = response.status_code
    token_dict = None
    
    try:
        response_text = response.text
        response_json = json.loads(response_text)
        response_text = response_json['content'][0]['text']
        
        # Extract token information if requested
        if iscaltoken:
            usage = response_json.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total = input_tokens + output_tokens
            token_dict = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total": total}
    except Exception as e:
        response_text = "Error: " + str(e)
    
    return response_text, status_code, token_dict


def _parse_deepseek_non_stream_response(response: requests.Response, iscaltoken: bool) -> Tuple[str, int, Optional[Dict]]:
    """
    Parse non-stream response from Deepseek (OpenAI-style format).
    
    Args:
        response: requests.Response object
        iscaltoken: Flag for token calculation
    
    Returns:
        Tuple of (response_text, status_code, token_dict)
    """
    status_code = response.status_code
    token_dict = None
    
    try:
        response_text = response.text
        response_json = json.loads(response_text)
        # OpenAI-style: extract from choices[0].message.content
        response_text = response_json['choices'][0]['message']['content']
        
        # Extract token information if requested
        if iscaltoken:
            usage = response_json.get("usage", {})
            # Map prompt_tokens -> input_tokens, completion_tokens -> output_tokens
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total = usage.get("total_tokens", input_tokens + output_tokens)
            token_dict = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total": total}
    except Exception as e:
        response_text = "Error: " + str(e)
    
    return response_text, status_code, token_dict


def _process_stream_response(url: str, body: dict, headers: dict, iscaltoken: bool, timeout: int = 1000) -> Tuple[str, int, Optional[Dict]]:
    """
    Send streaming request and process SSE stream response.
    
    Args:
        url: API endpoint URL
        body: Request body dictionary
        headers: Request headers dictionary
        iscaltoken: Flag for token calculation
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (response_text, status_code, token_dict)
    """
    # Add reasoning field for reasoning models (if not already present)
    model = body.get("model")
    if model in REASONING_MODELS and "reasoning" not in body:
        body["reasoning"] = {
            "exclude": True
        }
    
    # Add Accept header for streaming
    stream_headers = headers.copy()
    stream_headers["Accept"] = "text/event-stream"
    
    try:
        with requests.post(url, json=body, headers=stream_headers, stream=True, timeout=timeout) as r:
            status_code = r.status_code
            
            # Ensure UTF-8 decoding to avoid encoding issues
            r.encoding = "utf-8"
            
            full_text = []
            input_tokens = None
            output_tokens = None
            
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    # Empty line indicates end of SSE event
                    continue
                
                # Line format: "event: xxx" or "data: {...}"
                if line.startswith("event:"):
                    continue
                
                if line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    
                    # Some implementations send "[DONE]" marker
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        payload = json.loads(data_str)
                    except json.JSONDecodeError:
                        # Skip if data is not JSON
                        continue
                    
                    event_type = payload.get("type")
                    
                    # 1) message_start: usage is in payload["message"]["usage"]
                    if event_type == "message_start":
                        msg = payload.get("message", {})
                        u = msg.get("usage", {})
                        if "input_tokens" in u:
                            input_tokens = u["input_tokens"]
                    
                    # 2) Incremental text: content_block_delta -> delta.text
                    if event_type == "content_block_delta":
                        delta = payload.get("delta", {})
                        if delta.get("type") == "text_delta":
                            chunk = delta.get("text", "")
                            if chunk:
                                full_text.append(chunk)
                    
                    # 3) message_delta / message_stop: final output_tokens usually here
                    if event_type in ("message_delta", "message_stop"):
                        u = payload.get("usage", {})
                        if "output_tokens" in u:
                            # Overwrite each time, last one is final value
                            output_tokens = u["output_tokens"]
                    
                    # 4) End condition (should be after reading usage)
                    if event_type == "message_stop":
                        break
            
            response_text = "".join(full_text)
            
            # Build token_dict if requested
            token_dict = None
            if iscaltoken:
                input_tokens = input_tokens or 0
                output_tokens = output_tokens or 0
                total = input_tokens + output_tokens
                token_dict = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total": total}
            
            return response_text, status_code, token_dict
            
    except Exception as e:
        error_text = f"Error: {str(e)}"
        return error_text, 500, None


def _process_deepseek_stream_response(url: str, body: dict, headers: dict, iscaltoken: bool, timeout: int = 1000) -> Tuple[str, int, Optional[Dict]]:
    """
    Send streaming request and process OpenAI-style SSE stream response for Deepseek.
    
    Args:
        url: API endpoint URL
        body: Request body dictionary
        headers: Request headers dictionary
        iscaltoken: Flag for token calculation
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (response_text, status_code, token_dict)
    """
    # Add Accept header for streaming
    stream_headers = headers.copy()
    stream_headers["Accept"] = "text/event-stream"
    
    full_text = []
    input_tokens = None
    output_tokens = None
    status_code = 500
    
    try:
        with requests.post(url, json=body, headers=stream_headers, stream=True, timeout=timeout) as r:
            status_code = r.status_code
            
            # Check status code before processing stream
            if status_code != 200:
                error_text = r.text or f"HTTP {status_code} error"
                return error_text, status_code, None
            
            # Ensure UTF-8 decoding to avoid encoding issues
            r.encoding = "utf-8"
            
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    # Empty line indicates end of SSE event
                    continue
                
                # Line format: "data: {...}"
                if line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    
                    # Handle "[DONE]" marker
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        payload = json.loads(data_str)
                    except json.JSONDecodeError:
                        # Skip if data is not JSON
                        continue
                    
                    # OpenAI-style: extract content from choices[0].delta.content
                    choices = payload.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_text.append(content)
                    
                    # Extract token information from usage field
                    usage = payload.get("usage", {})
                    if usage:
                        if "prompt_tokens" in usage:
                            input_tokens = usage.get("prompt_tokens", 0)
                        if "completion_tokens" in usage:
                            output_tokens = usage.get("completion_tokens", 0)
            
            response_text = "".join(full_text)
            
            # Handle empty response
            if not response_text:
                error_text = f"Streaming response ended prematurely: no data received (status: {status_code})"
                return error_text, status_code, None
            
            # Build token_dict if requested
            token_dict = None
            if iscaltoken:
                input_tokens = input_tokens or 0
                output_tokens = output_tokens or 0
                total = input_tokens + output_tokens
                token_dict = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total": total}
            
            return response_text, status_code, token_dict
            
    except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        error_text = f"Streaming error: {str(e)}"
        response_text = "".join(full_text)
        if response_text:
            token_dict = None
            if iscaltoken:
                input_tokens = input_tokens or 0
                output_tokens = output_tokens or 0
                total = input_tokens + output_tokens
                token_dict = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total": total}
            return response_text, status_code, token_dict
        else:
            return error_text, status_code, None
    except requests.exceptions.RequestException as e:
        error_text = f"Request error: {str(e)}"
        return error_text, 500, None
    except Exception as e:
        error_text = f"Error: {str(e)}"
        return error_text, 500, None


def _process_openai_chatcompletions_stream_response(
    url: str,
    body: dict,
    headers: dict,
    iscaltoken: bool,
    timeout: int = 1000,
) -> Tuple[str, int, Optional[Dict]]:
    """
    Send streaming request and process OpenAI-style SSE stream response for /v1/chat/completions.

    Notes:
      - Parses incremental content from choices[0].delta.content
      - Optionally prints chunks in real time when env var YIDONG_STREAM_PRINT=1
    """
    # Add Accept header for streaming
    stream_headers = headers.copy()
    stream_headers["Accept"] = "text/event-stream"

    # Real-time printing is OFF by default to avoid affecting existing usage patterns.
    # Enable by: export/set YIDONG_STREAM_PRINT=1
    # print_stream = True
    print_stream = os.getenv("YIDONG_STREAM_PRINT", "0") in ("1", "true", "True", "YES", "yes")

    full_text: List[str] = []
    input_tokens = None
    output_tokens = None
    status_code = 500

    try:
        with requests.post(url, json=body, headers=stream_headers, stream=True, timeout=timeout) as r:
            status_code = r.status_code

            # Check status code before processing stream
            if status_code != 200:
                error_text = r.text or f"HTTP {status_code} error"
                return error_text, status_code, None

            # Ensure UTF-8 decoding to avoid encoding issues
            r.encoding = "utf-8"

            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()

                # Handle "[DONE]" marker
                if data_str == "[DONE]":
                    break

                try:
                    payload = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # OpenAI-style: extract content from choices[0].delta.content
                choices = payload.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {}) or {}
                    content = delta.get("content")
                    if content:
                        full_text.append(content)
                        if print_stream:
                            print(content, end="", flush=True)

                # Usage may appear in the final chunk, or not at all depending on the gateway
                usage = payload.get("usage")
                if isinstance(usage, dict):
                    # 1) Prefer OpenAI naming first
                    pt = usage.get("prompt_tokens")
                    ct = usage.get("completion_tokens")

                    if isinstance(pt, int) and pt >= 0:
                        input_tokens = pt
                    if isinstance(ct, int) and ct >= 0:
                        output_tokens = ct

                    # 2) Fallback only if OpenAI naming not present
                    if pt is None:
                        it2 = usage.get("input_tokens")
                        if isinstance(it2, int) and it2 > 0:
                            input_tokens = it2

                    if ct is None:
                        ot2 = usage.get("output_tokens")
                        if isinstance(ot2, int) and ot2 > 0:
                            output_tokens = ot2

            response_text = "".join(full_text)

            # Build token_dict if requested
            token_dict = None
            if iscaltoken:
                it = int(input_tokens or 0)
                ot = int(output_tokens or 0)
                token_dict = {"input_tokens": it, "output_tokens": ot, "total": it + ot}

            return response_text, status_code, token_dict

    except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        error_text = f"Streaming error: {str(e)}"
        response_text = "".join(full_text)
        if response_text:
            token_dict = None
            if iscaltoken:
                it = int(input_tokens or 0)
                ot = int(output_tokens or 0)
                token_dict = {"input_tokens": it, "output_tokens": ot, "total": it + ot}
            return response_text, status_code, token_dict
        return error_text, status_code, None
    except requests.exceptions.RequestException as e:
        return f"Request error: {str(e)}", 500, None
    except Exception as e:
        return f"Error: {str(e)}", 500, None


# send chat to yidong
def send_chat_diverse_model(user_model: str, role_prompt: str, user_prompt: str, temperature: float = None, top_p: float = None, max_tokens: int = None, iscaltoken: bool = False, isstream: bool = True) -> Tuple[str, int, Optional[Dict]]:
    """
    Send chat to diverse models via yidong API.
    
    Args:
        user_model: Model name
        role_prompt: System/role prompt
        user_prompt: User prompt
        temperature: Temperature parameter
        top_p: Top-p parameter
        max_tokens: Maximum tokens
        iscaltoken: Flag to calculate and return token statistics
        isstream: Flag to use streaming mode (default: True)
    
    Returns:
        Tuple of (response_text, status_code, token_dict)
        When iscaltoken=False, token_dict is None.
    """
    # prepare max_tokens
    if max_tokens is None:
        if user_model not in MODEL_MAX_TOKENS:
            raise ValueError(f"Model {user_model} not found in MODEL_MAX_TOKENS")
        max_tokens = MODEL_MAX_TOKENS[user_model]

    if user_model in NON_STREAM_MODELS:
        isstream = False

    # gemini models
    if user_model in ["gemini-3-flash-preview", "gemini-3-flash-preview-thinking", "gemini-3-pro-preview", "gemini-2.5-pro-preview-06-05", "gemini-2.5-flash-nothinking", "gemini-2.5-flash-lite-thinking"]:
        url = "https://yinli.one/v1/chat/completions"
        # url = "http://38.247.13.232:3000/v1/chat/completions"
        body = {
            "model": user_model,
            "messages": [
                {
                    "role": "user",
                    "content": f"{role_prompt}\n\n{user_prompt}"
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "timeout": 1000,
            "stream": isstream
        }
        body = {k: v for k, v in body.items() if v is not None}
        if user_model in MODEL_THINKING_BUDGET:
            body["generationConfig"] = {
                "thinkingConfig": {
                    "thinkingLevel": MODEL_THINKING_BUDGET[user_model]
                }
            }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{YIDONG_API_KEY_1}"
        }

        if isstream:
            return _process_openai_chatcompletions_stream_response(url, body, headers, iscaltoken, timeout=1000)
        else:
            response = requests.post(url=url, json=body, headers=headers)
            return _parse_deepseek_non_stream_response(response, iscaltoken)

    # openai models
    elif user_model in ["gpt-5.1-chat", "gpt-5.1-2025-11-13", "gpt-5.2", "gpt-4o-2024-11-20"]:
        url = "https://yinli.one/v1/chat/completions"
        body = {
            "model": user_model,
            "messages": [
                {
                    "role": "user",
                    "content": f"{role_prompt}\n\n{user_prompt}"
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": isstream,
        }
        # Remove None values to avoid gateway issues
        body = {k: v for k, v in body.items() if v is not None}

        if isstream and iscaltoken:
            body["stream_options"] = {"include_usage": True}

        headers = {
            "Content-Type": "application/json",
            "Authorization": YIDONG_API_KEY_2,
        }

        if isstream:
            return _process_openai_chatcompletions_stream_response(url, body, headers, iscaltoken, timeout=1000)
        else:
            response = requests.post(url=url, json=body, headers=headers)
            return _parse_deepseek_non_stream_response(response, iscaltoken)


    # claude models
    elif user_model in ["claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101", "claude-3-haiku-20240307","claude-3-5-sonnet-20241022","claude-sonnet-4-20250514"]:
        url = "https://yinli.one/v1/messages"
        body = {
            "model": user_model,
            "messages": [
                {
                    "role": "user",
                    "content": f"{role_prompt}\n\n{user_prompt}"
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": isstream
        }
        headers = {
            "anthropic-version": "2023-06-01",
            "Authorization": YIDONG_API_KEY_3
        }
        
        if isstream:
            return _process_stream_response(url, body, headers, iscaltoken, timeout=1000)
        else:
            response = requests.post(url=url, json=body, headers=headers)
            return _parse_non_stream_response(response, iscaltoken)


    # grok models
    elif user_model in ["grok-4-fast-reasoning", "grok-4-fast-non-reasoning"]:
        url = "https://yinli.one/v1/messages"
        body = {
            "model": user_model,
            "messages": [
                {
                    "role": "user",
                    "content": f"{role_prompt}\n\n{user_prompt}"
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": isstream
        }
        headers = {
            "Content-Type": "application/json", 
            "anthropic-version": "2023-06-01", 
            "Authorization": YIDONG_API_KEY_2
        }
        
        if isstream:
            return _process_stream_response(url, body, headers, iscaltoken, timeout=1000)
        else:
            response = requests.request("POST", url, data=json.dumps(body), headers=headers)
            return _parse_non_stream_response(response, iscaltoken)

    # deepseek models
    elif user_model in ["deepseek-r1", "deepseek-v3"]:
        url = "https://yinli.one/v1/chat/completions"
        body = {
            "model": user_model,
            "messages": [
                {
                    "role": "user",
                    "content": f"{role_prompt}\n\n{user_prompt}"
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": isstream,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": YIDONG_API_KEY_2
        }

        if isstream:
            return _process_deepseek_stream_response(url, body, headers, iscaltoken, timeout=1000)
        else:
            response = requests.post(url, json=body, headers=headers)
            return _parse_deepseek_non_stream_response(response, iscaltoken)

    # qwen (dashscope compatible-mode) models
    elif user_model in ["qwen3-30b-a3b-instruct-2507", "qwen3-next-80b-a3b-instruct", "qwen-plus", "qwen3.5-plus", "qwen3-max"]:
        # 选择地域
        # 北京: https://dashscope.aliyuncs.com/compatible-mode/v1
        # 美国: https://dashscope-us.aliyuncs.com/compatible-mode/v1
        # 新加坡: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        url = f"{base_url}/chat/completions"

        body = {
            "model": user_model,
            "messages": [
                {
                    "role": "user",
                    "content": f"{role_prompt}\n\n{user_prompt}"
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": isstream,
        }

        body = {k: v for k, v in body.items() if v is not None}

        if isstream and iscaltoken:
            body["stream_options"] = {"include_usage": True}

        headers = {
            "Content-Type": "application/json",
            "Authorization": YIDONG_API_KEY_4,
        }

        if isstream:
            return _process_openai_chatcompletions_stream_response(url, body, headers, iscaltoken, timeout=1000)
        else:
            response = requests.post(url=url, json=body, headers=headers, timeout=1000)
            return _parse_deepseek_non_stream_response(response, iscaltoken)