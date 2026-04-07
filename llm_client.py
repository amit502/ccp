"""
llm_client.py
Nautilus-hosted LLM client using OpenAI-compatible endpoint.
"""

from __future__ import annotations

import dataclasses
import os
from typing import List, Union

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

MODEL = "gpt-oss"


@dataclasses.dataclass
class Message:
    role:    str
    content: str


def _get_client() -> OpenAI:
    return OpenAI(
        base_url="https://ellm.nrp-nautilus.io/v1",
        api_key=os.environ["OPENAI_API_KEY"],
    )


def _chat_once(client, model: str, messages: list, temperature: float = 0.0) -> str:
    """
    Single chat completion — tries streaming first, falls back to non-streaming.
    Also checks tool_calls field in case the model routes output there.
    """
    import json as _json, sys

    # Try streaming first
    try:
        chunks = []
        tool_chunks = []
        with client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        ) as stream:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    chunks.append(delta.content)
                # Some endpoints put JSON in tool_calls even without tools defined
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        if hasattr(tc, "function"):
                            if tc.function.name:
                                tool_chunks.append(tc.function.name)
                            if tc.function.arguments:
                                tool_chunks.append(tc.function.arguments)

        result = "".join(chunks)
        if result:
            return result
        # Content empty but tool_call had data — reconstruct as JSON
        if tool_chunks:
            raw = "".join(tool_chunks)
            print(f"  [LLM] content empty, got tool_call data: {raw[:100]!r}", file=sys.stderr)
            return raw
    except Exception as e:
        print(f"  [LLM] streaming error: {e}", file=sys.stderr)

    # Fallback: non-streaming
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        choice = completion.choices[0]
        if choice.message.content:
            return choice.message.content
        # Check tool_calls on non-streaming response too
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            raw = tc.function.arguments or ""
            print(f"  [LLM] non-stream tool_call: {tc.function.name} args={raw[:80]!r}", file=sys.stderr)
            return raw
        # Log full response for diagnosis
        print(f"  [LLM] both empty. choice={choice!r}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"  [LLM] non-streaming error: {e}", file=sys.stderr)
        return ""


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def gpt_chat(
    model:       str,
    messages:    List[Message],
    max_tokens:  int   = 1024,
    temperature: float = 0.0,
    num_comps:   int   = 1,
) -> Union[List[str], str]:
    client    = _get_client()
    formatted = [dataclasses.asdict(m) for m in messages]
    try:
        if num_comps == 1:
            return _chat_once(client, model, formatted, temperature=0.0)
        else:
            return [_chat_once(client, model, formatted, temperature=0.2)
                    for _ in range(num_comps)]
    except Exception as e:
        print(f"gpt_chat error: {e}")
        return "" if num_comps == 1 else [""] * num_comps


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user",   content=user_prompt),
    ]
    result = gpt_chat(model=MODEL, messages=messages, temperature=temperature)
    assert isinstance(result, str)
    return result
