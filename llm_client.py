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
            completion = client.chat.completions.create(
                model=model,
                messages=formatted,
                temperature=0.0,
                # max_tokens=max_tokens,
            )
            return completion.choices[0].message.content or ""
        else:
            results = []
            for _ in range(num_comps):
                completion = client.chat.completions.create(
                    model=model,
                    messages=formatted,
                    temperature=0.2,
                    # max_tokens=max_tokens,
                )
                results.append(completion.choices[0].message.content or "")
            return results
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
