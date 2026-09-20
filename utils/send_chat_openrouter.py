"""Non-streaming OpenRouter adapter for the DFT agents."""
from typing import Dict, Optional, Tuple

from openai import OpenAI

from config import OPENROUTER_API_KEY


def send_chat_openrouter(
    user_model: str,
    role_prompt: str,
    user_prompt: str,
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None,
    iscaltoken: bool = False,
) -> Tuple[str, int, Optional[Dict]]:
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key is not set. Configure OPENROUTER_API_KEY in the root .env file.", 401, None

    params = {
        name: value
        for name, value in {
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        }.items()
        if value is not None
    }
    try:
        with OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY) as client:
            response = client.chat.completions.create(
                model=user_model,
                messages=[
                    {"role": "system", "content": role_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **params,
            )
        response_text = (response.choices[0].message.content or "").strip()
        token_dict = None
        if iscaltoken:
            usage = response.usage
            token_dict = {
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total": getattr(usage, "total_tokens", 0) or 0,
            }
        return response_text, 200, token_dict
    except Exception as exc:
        status = getattr(exc, "status_code", None) or 500
        # Do not echo provider bodies, request headers, or credentials in errors.
        return f"OpenRouter request failed ({type(exc).__name__}, status {status}).", status, None
