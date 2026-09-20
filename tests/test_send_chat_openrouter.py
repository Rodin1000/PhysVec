"""Explicit, real API smoke test: python tests/test_send_chat_openrouter.py."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.send_chat_openrouter import send_chat_openrouter

DEFAULT_TEST_MODEL = "qwen/qwen3-max"


def main() -> int:
    model = os.getenv("OPENROUTER_TEST_MODEL", DEFAULT_TEST_MODEL)
    response, status, usage = send_chat_openrouter(
        user_model=model,
        role_prompt="You are a helpful assistant.",
        user_prompt="Who are you",
        max_tokens=32,
        iscaltoken=True,
    )
    print(f"Model: {model}")
    print(f"Status: {status}")
    print(f"Response: {response}")
    print(f"Token usage: {usage}")
    return 0 if status == 200 and response.strip() else 1


if __name__ == "__main__":
    raise SystemExit(main())
