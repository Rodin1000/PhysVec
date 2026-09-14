# -*- coding: utf-8 -*-
"""
Stream test for OpenAI-compatible /v1/chat/completions endpoint.
Adds token usage accounting (input/output/total) with stream + fallback.
"""

import os
import json
import httpx

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode"
CHAT_URL = f"{BASE_URL}/v1/chat/completions"
MODEL = "qwen3-max"


def _extract_usage(event: dict):
    """
    Try to extract OpenAI-style usage from either:
      - event["usage"]  (some providers)
      - event["choices"][0]["usage"] (rare)
    Returns dict or None.
    """
    if not isinstance(event, dict):
        return None
    usage = event.get("usage")
    if isinstance(usage, dict):
        return usage
    choices = event.get("choices")
    if isinstance(choices, list) and choices:
        u = choices[0].get("usage")
        if isinstance(u, dict):
            return u
    return None


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _request_usage_nonstream(client: httpx.Client, headers: dict, payload: dict):
    """
    Fallback: send a non-stream request to get usage.
    Many OpenAI-compatible APIs return usage only in non-stream responses.
    """
    p2 = dict(payload)
    p2["stream"] = False

    r = client.post(CHAT_URL, headers=headers, json=p2)
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage") or {}
    return usage


def main():
    # ✅ 建议：用环境变量
    api_key = "sk-c5a058ca63844c2ebc8d26aad66e84ef"  # PowerShell: $env:API_KEY="sk-xxx"
    if not api_key:
        raise RuntimeError("Missing API_KEY env var. Please set API_KEY first.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "you are a helpful assistant. who are you"},
        ],
        "stream": True,
        # 有些兼容实现支持这个：在 stream 内包含 usage（不保证）
        # "stream_options": {"include_usage": True},
    }

    timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)

    # 统计变量
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    got_usage_from_stream = False

    print(f"==> POST {CHAT_URL}")
    print(f"==> model: {MODEL}")
    print("==> streaming...\n")

    with httpx.Client(timeout=timeout) as client:
        try:
            with client.stream("POST", CHAT_URL, headers=headers, json=payload) as resp:
                print(f"[HTTP {resp.status_code}]")
                ctype = resp.headers.get("content-type", "")
                print(f"[content-type] {ctype}\n")

                if resp.status_code != 200:
                    err_text = resp.read().decode("utf-8", errors="replace")
                    print("Request failed. Body:\n", err_text)
                    return

                for line in resp.iter_lines():
                    if not line:
                        continue

                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()

                        if data == "[DONE]":
                            print("\n\n==> [DONE]")
                            break

                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            print(f"\n[non-json data] {data}")
                            continue

                        # 1) 尝试从 stream 事件里抓 usage
                        usage = _extract_usage(event)
                        if usage:
                            got_usage_from_stream = True
                            pt = _safe_int(usage.get("prompt_tokens"))
                            ct = _safe_int(usage.get("completion_tokens"))
                            tt = _safe_int(usage.get("total_tokens"))
                            # 有的服务会多次发送 usage，这里以“最新一次”为准
                            if pt is not None:
                                prompt_tokens = pt
                            if ct is not None:
                                completion_tokens = ct
                            if tt is not None:
                                total_tokens = tt

                        # 2) 正常打印 token 流内容
                        try:
                            choices = event.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                print(content, end="", flush=True)
                        except Exception as e:
                            print(f"\n[parse error] {e}\nraw={event}")

            # stream 结束后：如果没拿到 usage，就 fallback 非流请求拿 usage
            if not got_usage_from_stream:
                try:
                    usage2 = _request_usage_nonstream(client, headers, payload)
                    prompt_tokens = _safe_int(usage2.get("prompt_tokens"))
                    completion_tokens = _safe_int(usage2.get("completion_tokens"))
                    total_tokens = _safe_int(usage2.get("total_tokens"))
                except Exception as e:
                    print("\n[usage fallback failed]", repr(e))

            print("\n\n==> token usage")
            print(f"prompt_tokens     : {prompt_tokens}")
            print(f"completion_tokens : {completion_tokens}")
            print(f"total_tokens      : {total_tokens}")

        except httpx.ConnectError as e:
            print("ConnectError:", e)
        except httpx.ReadError as e:
            print("ReadError:", e)
        except Exception as e:
            print("Unexpected error:", repr(e))


if __name__ == "__main__":
    main()