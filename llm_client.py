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
    Single chat completion call using streaming to work around the ellm endpoint
    returning empty content on non-streaming requests.
    """
    chunks = []
    with client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                chunks.append(delta)
    return "".join(chunks)


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
