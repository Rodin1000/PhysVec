"""OpenRouter adapter for text-only chat completions."""

from typing import Dict, Optional, Tuple

from openai import OpenAI

from config import OPENROUTER_API_KEY


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def send_chat_openrouter(
    user_model: str,
    role_prompt: str,
    user_prompt: str,
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
    iscaltoken: bool = False,
) -> Tuple[str, int, Optional[Dict]]:
    """Send a text-only chat request through OpenRouter.

    The return shape matches ``send_chat_yidong.send_chat_diverse_model``:
    ``(response_text, status_code, token_dict)``.
    """
    if not OPENROUTER_API_KEY:
        return (
            "OpenRouter API key is not set. Please set OPENROUTER_API_KEY in the environment or project .env file.",
            401,
            None,
        )

    request_params = {
        "model": user_model,
        "messages": [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    request_params = {key: value for key, value in request_params.items() if value is not None}

    try:
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
        response = client.chat.completions.create(**request_params)

        content = response.choices[0].message.content
        response_text = content.strip() if isinstance(content, str) else ""

        token_dict = None
        if iscaltoken:
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            reported_total = getattr(usage, "total_tokens", None)
            total_tokens = int(reported_total if reported_total is not None else input_tokens + output_tokens)
            token_dict = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total": total_tokens,
            }

        return response_text, 200, token_dict
    except Exception as exc:
        status_code = getattr(exc, "status_code", 500)
        if not isinstance(status_code, int):
            status_code = 500
        return f"OpenRouter request failed: {exc}", status_code, None
