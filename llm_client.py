"""
llm_client.py
Nautilus-hosted LLM client (OpenAI-compatible endpoint).
Uses OPENAI_API_KEY (same secret as existing Nautilus jobs).
"""

from __future__ import annotations

import dataclasses
import os
from typing import List, Union

from tenacity import retry, stop_after_attempt, wait_random_exponential

OPENAI_BASE_URL = "https://llm.nrp-nautilus.io/"
MODEL = "gpt-oss"

@dataclasses.dataclass
class Message:
    role: str
    content: str

_client = None

def _get_client():
    global _client
    if _client is None:
        import openai
        _client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=OPENAI_BASE_URL,
        )
    return _client

@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def gpt_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps: int = 1,
) -> Union[List[str], str]:
    client = _get_client()
    formatted = [dataclasses.asdict(m) for m in messages]
    try:
        if num_comps == 1:
            completion = client.chat.completions.create(
                model=model, messages=formatted, temperature=temperature,
            )
            return completion.choices[0].message.content or ""
        else:
            results = []
            for _ in range(num_comps):
                completion = client.chat.completions.create(
                    model=model, messages=formatted, temperature=0.2,
                )
                results.append(completion.choices[0].message.content or "")
            return results
    except Exception as e:
        print(f"[gpt_chat] error: {e}")
        return "" if num_comps == 1 else [""] * num_comps

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    messages = [Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt)]
    result = gpt_chat(model=MODEL, messages=messages, temperature=temperature)
    assert isinstance(result, str)
    return result
