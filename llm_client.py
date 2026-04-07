"""
llm_client.py
Direct HTTP call to Nautilus ellm endpoint — bypasses OpenAI client parsing.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from typing import List, Union

import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

BASE_URL = "https://ellm.nrp-nautilus.io/v1"
MODEL    = "gpt-oss"


@dataclasses.dataclass
class Message:
    role:    str
    content: str


def _get_client():
    """Keep for backward compat — not used in hot path."""
    from openai import OpenAI
    return OpenAI(
        base_url=BASE_URL,
        api_key=os.environ["OPENAI_API_KEY"],
    )


def _raw_chat(messages: list, temperature: float = 0.0) -> str:
    """
    POST directly to the ellm endpoint and return content string.
    Logs the full raw JSON on first call so we can see exactly what's returned.
    """
    payload = {
        "model":       MODEL,
        "messages":    messages,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type":  "application/json",
    }
    r = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    r.raise_for_status()

    body = r.json()

    # Log raw response once for diagnosis
    if os.environ.get("LLM_DEBUG", "0") == "1":
        print(f"  [LLM raw] {json.dumps(body)[:400]}", file=sys.stderr, flush=True)

    # Standard path
    choices = body.get("choices", [])
    if not choices:
        print(f"  [LLM] no choices in response: {json.dumps(body)[:200]}", file=sys.stderr, flush=True)
        return ""

    choice  = choices[0]
    message = choice.get("message", {})
    content = message.get("content") or ""

    if content:
        return content

    # content empty — check other fields
    # 1. tool_calls (model routing to function calling)
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        tc = tool_calls[0]
        fn = tc.get("function", {})
        raw = fn.get("arguments", "")
        print(f"  [LLM] content empty, tool_call: {fn.get('name')} args={raw[:80]!r}", file=sys.stderr, flush=True)
        return raw

    # 2. Log full message so we know where the tokens went
    print(
        f"  [LLM] empty content. finish={choice.get('finish_reason')!r} "
        f"tokens={body.get('usage',{}).get('completion_tokens')} "
        f"message_keys={list(message.keys())}",
        file=sys.stderr, flush=True,
    )
    # Print full raw for diagnosis on first empty
    print(f"  [LLM raw full] {json.dumps(body)[:600]}", file=sys.stderr, flush=True)
    return ""


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def gpt_chat(
    model:       str,
    messages:    List[Message],
    max_tokens:  int   = 1024,
    temperature: float = 0.0,
    num_comps:   int   = 1,
) -> Union[List[str], str]:
    formatted = [dataclasses.asdict(m) for m in messages]
    try:
        if num_comps == 1:
            return _raw_chat(formatted, temperature=0.0)
        else:
            return [_raw_chat(formatted, temperature=0.2) for _ in range(num_comps)]
    except Exception as e:
        print(f"gpt_chat error: {e}", file=sys.stderr)
        return "" if num_comps == 1 else [""] * num_comps


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user",   content=user_prompt),
    ]
    result = gpt_chat(model=MODEL, messages=messages, temperature=temperature)
    assert isinstance(result, str)
    return result
