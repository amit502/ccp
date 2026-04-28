"""
nq_mcp_server.py

Real MCP server for Natural Questions retrieval (Multi-objective QA benchmark).

Exposes search and lookup tools backed by:
  1. Local NQ JSONL file  (MULTIQA_DATA_FILE env var)  — fastest
  2. HuggingFace NQ dataset (auto-downloaded)          — requires internet
  3. DuckDuckGo web search fallback                    — always available

Started as a subprocess by MultiObjQAMCPRunner.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types as mcp_types

# ---------------------------------------------------------------------------
# NQ knowledge base — loaded once at startup
# ---------------------------------------------------------------------------

MULTIQA_DATA_FILE = os.environ.get("MULTIQA_DATA_FILE", "")
# JSON cache so repeated per-call subprocess restarts load in milliseconds
# instead of re-fetching 2500+ items from HuggingFace every time.
_NQ_KB_CACHE_FILE = os.environ.get("NQ_KB_CACHE_FILE", "/tmp/nq_kb_cache.json")
_NQ_KB: Dict[str, str] = {}   # question (normalised) → short answer


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def _load_nq_kb(path: str, max_items: int = 10000) -> Dict[str, str]:
    """Load question→answer pairs from a local NQ JSONL(.gz) file."""
    kb     = {}
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    q    = item.get("question_text") or item.get("question", "")
                    # NQ annotations: take the first short answer if available
                    ans  = ""
                    for ann in item.get("annotations", []):
                        sa = ann.get("short_answers", [])
                        if sa:
                            start = sa[0].get("start_token", 0)
                            end   = sa[0].get("end_token", 0)
                            tokens = item.get("document", {}).get("tokens", [])
                            if tokens:
                                ans = " ".join(
                                    t["token"] for t in tokens[start:end]
                                    if not t.get("is_html", False)
                                )
                            break
                    if q and ans:
                        kb[_normalise(q)] = ans
                    if len(kb) >= max_items:
                        break
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[NQ MCP] Loaded {len(kb)} QA pairs from {path}", file=__import__("sys").stderr)
    except Exception as e:
        print(f"[NQ MCP] Failed to load {path}: {e}", file=__import__("sys").stderr)
    return kb


def _load_hf_kb(max_items: int = 5000) -> Dict[str, str]:
    """Load from HuggingFace Hub (google-research-datasets/natural_questions)."""
    try:
        from datasets import load_dataset
        kb = {}
        ds = load_dataset(
            "google-research-datasets/natural_questions",
            split="validation",
            streaming=True,
        )
        for item in ds:
            q   = (item.get("question") or {}).get("text", "")
            ans_list = item.get("annotations", {}).get("short_answers", [[]])
            ans = ""
            if ans_list and ans_list[0]:
                first = ans_list[0]
                if isinstance(first, list):
                    ans = first[0] if first else ""
                elif isinstance(first, dict):
                    # HF NQ format: {"start_token": [...], "text": ["George Washington"]}
                    texts = first.get("text", [])
                    ans = texts[0] if texts else ""
                else:
                    ans = str(first)
            if q and ans:
                kb[_normalise(q)] = str(ans)
            if len(kb) >= max_items:
                break
        print(f"[NQ MCP] Loaded {len(kb)} QA pairs from HuggingFace",
              file=__import__("sys").stderr)
        return kb
    except Exception as e:
        print(f"[NQ MCP] HuggingFace load failed: {e}", file=__import__("sys").stderr)
        return {}


def _init_kb() -> Dict[str, str]:
    import sys

    # Fast path: JSON cache written by a previous startup in this pod run.
    # With per-call subprocess restarts, this cuts load time from ~100s to <1s.
    if _NQ_KB_CACHE_FILE and Path(_NQ_KB_CACHE_FILE).exists():
        try:
            kb = json.loads(Path(_NQ_KB_CACHE_FILE).read_text())
            if kb:
                print(f"[NQ MCP] Loaded {len(kb)} QA pairs from cache {_NQ_KB_CACHE_FILE}",
                      file=sys.stderr)
                return kb
        except Exception as e:
            print(f"[NQ MCP] Cache read failed ({e}) — reloading from source", file=sys.stderr)

    # Slow path: load from local file or HuggingFace
    kb: Dict[str, str] = {}
    if MULTIQA_DATA_FILE and Path(MULTIQA_DATA_FILE).exists():
        kb = _load_nq_kb(MULTIQA_DATA_FILE)
    if not kb:
        kb = _load_hf_kb()
    if not kb:
        # Minimal fallback — enough to test the pipeline
        print("[NQ MCP] No NQ data found. Using minimal built-in KB.", file=sys.stderr)
        kb = {
            "who was the first president of the united states":  "George Washington",
            "when was the eiffel tower completed":               "1889",
            "what is the capital of australia":                  "Canberra",
            "who wrote 1984":                                    "George Orwell",
            "what is the chemical symbol for gold":              "Au",
            "what is the largest planet in the solar system":    "Jupiter",
            "who painted the mona lisa":                         "Leonardo da Vinci",
            "when did world war ii end":                         "1945",
            "what is the speed of light in kilometers per second": "299792",
            "who discovered penicillin":                         "Alexander Fleming",
            "what is the longest river in the world":            "Nile River",
            "what is the currency of japan":                     "Japanese yen",
        }

    # Write cache for subsequent startups within this pod run
    if _NQ_KB_CACHE_FILE and kb:
        try:
            Path(_NQ_KB_CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
            Path(_NQ_KB_CACHE_FILE).write_text(json.dumps(kb))
            print(f"[NQ MCP] Cached {len(kb)} QA pairs to {_NQ_KB_CACHE_FILE}", file=sys.stderr)
        except Exception as e:
            print(f"[NQ MCP] Cache write failed ({e})", file=sys.stderr)

    return kb


def _fuzzy_lookup(query: str, kb: Dict[str, str]) -> str:
    """Return best matching answer from KB using word overlap."""
    q_words = set(_normalise(query).split())
    best_score = 0
    best_ans   = "No answer found in knowledge base."

    for key, val in kb.items():
        k_words = set(key.split())
        overlap  = len(q_words & k_words) / (len(q_words | k_words) + 1e-9)
        if overlap > best_score:
            best_score = overlap
            best_ans   = val
    return best_ans


def _web_search_fallback(query: str) -> str:
    """DuckDuckGo instant answer API — no API key required."""
    try:
        import requests
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=5,
        )
        data = r.json()
        if data.get("AbstractText"):
            return data["AbstractText"]
        if data.get("Answer"):
            return data["Answer"]
        related = data.get("RelatedTopics", [])
        if related and related[0].get("Text"):
            return related[0]["Text"]
    except Exception:
        pass
    return "No result found."


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

class NQMCPServer:
    def __init__(self):
        self.server = Server("nq-retrieval")
        self._kb    = _init_kb()
        self._register_handlers()

    def _register_handlers(self):

        @self.server.list_tools()
        async def list_tools() -> List[mcp_types.Tool]:
            return [
                mcp_types.Tool(
                    name="search",
                    description="Search the knowledge base for an answer to a question.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The question to answer"},
                            "top_k": {"type": "integer", "description": "Number of results (default 3)"},
                        },
                        "required": ["query"],
                    },
                ),
                mcp_types.Tool(
                    name="lookup_fact",
                    description="Look up a specific fact about an entity.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity":    {"type": "string"},
                            "attribute": {"type": "string"},
                        },
                        "required": ["entity", "attribute"],
                    },
                ),
                mcp_types.Tool(
                    name="web_search",
                    description="Search the web for current information (DuckDuckGo).",
                    inputSchema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
            if name == "search":
                query  = arguments.get("query", "")
                top_k  = int(arguments.get("top_k", 3))
                answer = _fuzzy_lookup(query, self._kb)
                # Return top_k-style results for consistency
                results = [
                    {"rank": 1, "query": query, "answer": answer, "source": "NQ"},
                ]
                if answer == "No answer found in knowledge base.":
                    # Fallback to web
                    web_ans = _web_search_fallback(query)
                    results.append({"rank": 2, "query": query, "answer": web_ans, "source": "web"})
                output = json.dumps({"results": results, "total": len(results)})

            elif name == "lookup_fact":
                entity    = arguments.get("entity", "")
                attribute = arguments.get("attribute", "")
                query     = f"{entity} {attribute}"
                answer    = _fuzzy_lookup(query, self._kb)
                output    = json.dumps({"entity": entity, "attribute": attribute, "value": answer})

            elif name == "web_search":
                query  = arguments.get("query", "")
                answer = _web_search_fallback(query)
                output = json.dumps({"query": query, "result": answer})

            else:
                output = json.dumps({"error": f"Unknown tool: {name}"})

            return [mcp_types.TextContent(type="text", text=output)]

    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


if __name__ == "__main__":
    server = NQMCPServer()
    asyncio.run(server.run())
