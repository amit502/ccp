"""
llm_client.py — Nautilus ellm endpoint, raw HTTP + JSON extraction.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
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
    from openai import OpenAI
    return OpenAI(base_url=BASE_URL, api_key=os.environ["OPENAI_API_KEY"])


def _extract_text(response_body: str) -> str:
    """
    Extract the LLM's text from the raw HTTP response body.
    Tries multiple locations in order:
      1. choices[0].message.content  (standard)
      2. choices[0].message.tool_calls[0].function.arguments  (function-call mode)
      3. Any JSON object found anywhere in the body
    Logs the raw body if all paths return empty.
    """
    try:
        body = json.loads(response_body)
    except json.JSONDecodeError:
        # Not JSON at all — return raw text if it looks useful
        if len(response_body.strip()) > 0:
            return response_body.strip()
        return ""

    choices = body.get("choices", [])
    if not choices:
        print(f"  [LLM] no choices. body={response_body[:300]}", file=sys.stderr, flush=True)
        return ""

    msg = choices[0].get("message", {})

    # 1. Standard content field
    content = msg.get("content") or ""
    if content:
        return content

    # 2. Function/tool call arguments
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        args = (tool_calls[0].get("function") or {}).get("arguments", "")
        if args:
            print(f"  [LLM] content in tool_calls.arguments", file=sys.stderr, flush=True)
            return args

    # 3. Reasoning model — content is in "reasoning" field, need a follow-up call
    reasoning = msg.get("reasoning") or ""
    if reasoning:
        print(f"  [LLM] reasoning model detected, follow-up call", file=sys.stderr, flush=True)
        return _REASONING_SENTINEL + reasoning

    # 4. Log full body
    print(
        f"  [LLM] empty. finish={choices[0].get('finish_reason')!r} "
        f"tokens={body.get('usage', {}).get('completion_tokens')} "
        f"msg_keys={list(msg.keys())} "
        f"raw={response_body[:400]}",
        file=sys.stderr, flush=True,
    )
    return ""


_REASONING_SENTINEL = "__REASONING__:"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def _call_raw(messages: list, temperature: float = 0.0) -> str:
    r = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL, "messages": messages, "temperature": temperature},
        timeout=120,
    )
    r.raise_for_status()
    return _extract_text(r.text)


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
            return _call_raw(formatted, temperature=0.0)
        return [_call_raw(formatted, temperature=0.2) for _ in range(num_comps)]
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

    # Reasoning model returned thinking but no action — do a follow-up call
    # to extract the actual JSON action from the reasoning
    if result.startswith(_REASONING_SENTINEL):
        reasoning = result[len(_REASONING_SENTINEL):]
        followup_messages = [
            Message(role="system", content=system_prompt),
            Message(role="user",   content=user_prompt),
            Message(role="assistant", content=f"<thinking>{reasoning}</thinking>"),
            Message(role="user",   content=(
                "Based on your reasoning above, output ONLY the JSON action now. "
                "No thinking tags, no explanation — just the JSON."
            )),
        ]
        result = gpt_chat(model=MODEL, messages=followup_messages, temperature=temperature)
        assert isinstance(result, str)
        print(f"  [LLM] follow-up result: {result[:100]!r}", file=sys.stderr, flush=True)

    return result
