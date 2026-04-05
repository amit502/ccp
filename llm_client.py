"""
llm_client.py
Nautilus LLM — called directly via requests to avoid openai client version issues.
"""

from __future__ import annotations

import dataclasses
import os
from typing import List, Union

import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential

LLM_BASE_URL = "https://llm.nrp-nautilus.io"
MODEL        = "gpt-oss"


@dataclasses.dataclass
class Message:
    role:    str
    content: str


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type":  "application/json",
    }


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def gpt_chat(
    model:       str,
    messages:    List[Message],
    temperature: float = 0.0,
    num_comps:   int   = 1,
) -> Union[List[str], str]:
    payload = {
        "model":       model,
        "messages":    [dataclasses.asdict(m) for m in messages],
        "temperature": temperature,
    }
    if num_comps == 1:
        r = requests.post(
            f"{LLM_BASE_URL}/v1/chat/completions",
            headers=_headers(),
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
    else:
        results = []
        for _ in range(num_comps):
            r = requests.post(
                f"{LLM_BASE_URL}/v1/chat/completions",
                headers=_headers(),
                json={**payload, "temperature": 0.2},
                timeout=120,
            )
            r.raise_for_status()
            results.append(r.json()["choices"][0]["message"]["content"] or "")
        return results


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user",   content=user_prompt),
    ]
    result = gpt_chat(model=MODEL, messages=messages, temperature=temperature)
    assert isinstance(result, str)
    return result
