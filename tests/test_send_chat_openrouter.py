"""Real OpenRouter connectivity test.

Run from the project root:

    source .venv/bin/activate
    python tests/test_send_chat_openrouter.py

Optionally override the test model with ``OPENROUTER_TEST_MODEL``.
This script sends one real, low-token API request and never prints the API key.
"""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.send_chat_openrouter import send_chat_openrouter


DEFAULT_TEST_MODEL = "qwen/qwen3-max"


def main() -> int:
    model = os.getenv("OPENROUTER_TEST_MODEL", DEFAULT_TEST_MODEL)
    print(f"Testing OpenRouter model: {model}")

    response, status_code, token_dict = send_chat_openrouter(
        user_model=model,
        role_prompt="You are a concise API connectivity test assistant.",
        user_prompt="Who are you",
        temperature=0.0,
        max_tokens=32,
        iscaltoken=True,
    )

    if status_code != 200:
        print(f"OpenRouter test failed (status {status_code}): {response}", file=sys.stderr)
        return 1

    if not response.strip():
        print("OpenRouter test failed: the API returned an empty response.", file=sys.stderr)
        return 1

    print(f"Response: {response}")
    print(f"Token usage: {token_dict}")
    print("OpenRouter connectivity test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
