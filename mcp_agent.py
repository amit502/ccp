"""
mcp_agent.py

LangGraph agent with REAL MCP integration.

Key fix from v3: the ToolCallInterceptor now wraps ANY context manager
(CCPContextManager, FIFOManager, ACONContextManager, NoCompression, etc.)
via their shared add_observation() interface. Every baseline runs through
the same MCP path — no special-casing.

Architecture:

  Task Goal
      │
      ▼
  LangGraph Agent ◄── Compressed Context C*_t  (via manager.get_compressed_context())
      │
      │ tool_call (JSON-RPC over stdio)
      ▼
  MultiServerMCPClient
      │
      ▼
  MCP Server subprocess
      │  raw tool result
      ▼
  GenericToolCallInterceptor.add_observation(tool, input, raw_output)
      │                          ↑
      │            any context manager: CCP | FIFO | ACON | NoCompression | …
      ▼
  Agent context (compressed according to manager's policy)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager  # kept for potential future use
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# ---------------------------------------------------------------------------
# Generic interceptor — works with every context manager
# ---------------------------------------------------------------------------

class GenericToolCallInterceptor:
    """
    MCP ToolCallInterceptor that feeds every tool response through
    whichever context manager is in use (CCP, FIFO, ACON, NoCompression …).

    All context managers share the same add_observation(tool, input, output, status)
    interface, so the interceptor is completely manager-agnostic.

    The raw MCP tool output is intercepted here — BEFORE it enters the agent
    context — and the manager decides whether to keep, compress, or discard it.
    The agent then receives manager.get_compressed_context() at each step.
    """

    def __init__(self, manager: Any):
        """
        Args:
            manager: Any object with add_observation(tool_name, tool_input,
                     tool_output, status) — CCPContextManager, FIFOManager,
                     ACONContextManager, NoCompression, RetrievalBasedManager.
        """
        self.manager = manager

    async def __call__(
        self,
        request:  Any,                              # MCPToolCallRequest
        handler:  Callable[..., Awaitable[Any]],    # executes the actual tool
    ) -> Any:
        # 1. Execute tool on the real MCP server
        result = await handler(request)

        # 2. Extract raw text output
        raw_output = ""
        if hasattr(result, "content"):
            for block in result.content:
                if hasattr(block, "text"):
                    raw_output += block.text

        # 3. Feed through the context manager (CCP scores; FIFO tracks; etc.)
        element = self.manager.add_observation(
            tool_name=request.name,
            tool_input=dict(request.args or {}),
            tool_output=raw_output,
            status="ok",
        )

        # 4. If the manager compressed the output, return compressed version
        compressed = getattr(element, "compressed_output", None)
        if compressed is not None and compressed != raw_output:
            from mcp import types as mcp_types
            result = type(result)(
                content=[mcp_types.TextContent(type="text", text=compressed)]
            )

        return result


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class MCPAgentState(TypedDict):
    messages:     List[Any]
    goal:         str
    step:         int
    max_steps:    int
    done:         bool
    final_answer: Optional[str]


_SYSTEM_PROMPT = """\
You are a task-completion agent. Complete the given goal by calling tools.

To call a tool respond with ONLY this JSON (nothing else before or after):
{"action": "tool_call", "tool": "<exact_tool_name>", "input": {<params>}}

When the task is fully complete respond with ONLY:
FINAL ANSWER: <your answer>

Rules:
- Respond with JSON tool call OR FINAL ANSWER only — no prose, no explanation.
- Study the available tools and their descriptions carefully.
- Use tool results to make progress step by step.
"""


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

async def _agent_node(state: MCPAgentState, tools: List[Any]) -> MCPAgentState:
    """
    Agent node that calls the Nautilus LLM and parses the response.
    Prompts LLM to return JSON; falls back to regex parsing.
    Falls back to regex parsing if JSON is malformed.
    """
    import json, re
    # Build tool list for the prompt
    tool_list = "\n".join(
        f"  {t.name}: {t.description}" for t in tools
    )

    # Tool list (trimmed)
    tool_summary = "\n".join(
        f"  {t.name}: {t.description.split('.')[0]}" for t in tools[:60]
    )
    if len(tools) > 60:
        tool_summary += f"\n  ... and {len(tools)-60} more tools"

    system_text = f"""You are an autonomous agent completing tasks by calling real API tools.

RULES:
1. You are in a REAL environment. All tools work. Call them.
2. NEVER say "I cannot", "unable to locate", or ask the user for info.
3. ALWAYS follow this sequence:
   a. Call supervisor__show_active_task_active_task_get to get task details + credentials
   b. Authenticate with each required app using its __authenticate tool and the supervisor credentials
   c. Call tools to complete the task
   d. Only finish after ALL actions are verified done
4. Respond ONLY with a JSON object — no prose.

TOOL CALL: {{"action":"tool_call","tool":"<exact_tool_name>","input":{{<params>}}}}
FINISH:    {{"action":"finish","answer":"<what you did>"}}

Available tools:
{tool_summary}

Goal: {state["goal"]}"""

    # Build proper messages array — each tool call and result as its own turn
    # This is what the reasoning model needs to track conversation state
    llm_messages = [{"role": "system", "content": system_text}]

    if not state["messages"]:
        llm_messages.append({"role": "user", "content": "Begin. Call the first tool now."})
    else:
        for m in state["messages"]:
            if isinstance(m, SystemMessage):
                continue
            elif isinstance(m, HumanMessage):
                llm_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                # assistant turn — show what it called
                parts = []
                if m.content:
                    parts.append(m.content)
                for tc in (getattr(m, "tool_calls", None) or []):
                    parts.append(json.dumps({
                        "action": "tool_call",
                        "tool":   tc.get("name", ""),
                        "input":  tc.get("args", {}),
                    }))
                llm_messages.append({
                    "role":    "assistant",
                    "content": "\n".join(parts) or "(called tool)",
                })
            else:
                # ToolMessage — show result as a user turn
                content = getattr(m, "content", "") or ""
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                content = str(content)[:800]
                if content:
                    llm_messages.append({
                        "role":    "user",
                        "content": f"Tool result: {content}",
                    })

    # Call LLM with full conversation history
    try:
        from llm_client import _call_raw as _llm_raw, _REASONING_SENTINEL
        formatted = [{"role": m["role"], "content": m["content"]} for m in llm_messages]
        raw = await asyncio.get_event_loop().run_in_executor(
            None, _llm_raw, formatted, 0.0
        )
        if not isinstance(raw, str):
            raw = ""

        # Reasoning model — do follow-up to get actual JSON action
        if raw.startswith(_REASONING_SENTINEL):
            reasoning = raw[len(_REASONING_SENTINEL):]
            follow_msgs = formatted + [
                {"role": "assistant", "content": f"<thinking>{reasoning}</thinking>"},
                {"role": "user", "content":
                    "Based on your reasoning above, output ONLY the JSON action now. "
                    "No thinking tags — just the JSON object."},
            ]
            raw = await asyncio.get_event_loop().run_in_executor(
                None, _llm_raw, follow_msgs, 0.0
            )
            if not isinstance(raw, str):
                raw = ""
            # Strip any remaining sentinel
            if raw.startswith(_REASONING_SENTINEL):
                raw = ""
    except Exception as e:
        import traceback
        print(f"  [LLM] error: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raw = ""

    # Always log LLM output so we can see what's happening
    print(f"  [LLM s={state['step']}] {raw[:200]!r}", flush=True)

    # Parse JSON response — handle multiple common formats
    done = False
    final_answer = None
    tool_calls = []

    try:
        clean  = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)
        action = parsed.get("action", "")

        # Detect finish
        if (action == "finish"
                or parsed.get("answer") is not None
                or "FINAL ANSWER" in str(parsed.get("answer", ""))):
            done         = True
            final_answer = str(parsed.get("answer", parsed.get("result", "")))

        # Detect tool call — handle multiple naming conventions
        elif action in ("tool_call", "call_tool", "use_tool") or              "tool" in parsed or "function" in parsed or "name" in parsed:
            # Extract tool name
            tool_name = (parsed.get("tool")
                         or parsed.get("function")
                         or parsed.get("name")
                         or "")
            # Extract args
            tool_args = (parsed.get("input")
                         or parsed.get("args")
                         or parsed.get("arguments")
                         or parsed.get("parameters")
                         or {})
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    tool_args = {}
            if tool_name:
                tool_calls = [{
                    "name": tool_name,
                    "args": tool_args,
                    "id":   f"call_{state['step']}",
                    "type": "tool_call",
                }]
            else:
                print(f"  [LLM] no tool name found in: {parsed}", flush=True)
        else:
            print(f"  [LLM] unrecognized JSON: {parsed}", flush=True)

    except (json.JSONDecodeError, TypeError):
        # Plain-text fallback
        if "FINAL ANSWER:" in raw:
            done         = True
            final_answer = raw.split("FINAL ANSWER:", 1)[1].strip()
        else:
            print(f"  [LLM] not JSON: {raw[:150]!r}", flush=True)

    messages = state["messages"] or [
        SystemMessage(content=system_text),
        HumanMessage(content='Begin. Call the first tool now.'),
    ]
    response = AIMessage(content=raw, tool_calls=tool_calls)

    return {
        **state,
        "messages":     messages + [response],
        "step":         state["step"] + 1,
        "done":         done,
        "final_answer": final_answer,
    }


def _route(state: MCPAgentState) -> str:
    if state["done"] or state["step"] >= state["max_steps"]:
        return "end"
    messages = state["messages"]
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# Main entry point — accepts ANY context manager
# ---------------------------------------------------------------------------



class MutableInterceptor:
    """
    ToolCallInterceptor whose manager can be swapped between tasks.
    One MultiServerMCPClient + one subprocess serves all tasks per method run.
    """
    def __init__(self):
        self.manager: Optional[Any] = None

    def set_manager(self, manager: Any) -> None:
        self.manager = manager

    async def __call__(self, request: Any, handler: Callable[..., Awaitable[Any]]) -> Any:
        result = await handler(request)
        if self.manager is None:
            return result

        raw_output = ""
        if hasattr(result, "content"):
            for block in result.content:
                if hasattr(block, "text"):
                    raw_output += block.text

        element = self.manager.add_observation(
            tool_name=request.name,
            tool_input=dict(request.args or {}),
            tool_output=raw_output,
            status="ok",
        )
        compressed = getattr(element, "compressed_output", None)
        if compressed is not None and compressed != raw_output:
            from mcp import types as mcp_types
            result = type(result)(
                content=[mcp_types.TextContent(type="text", text=compressed)]
            )
        return result


async def build_shared_client(server_configs: Dict[str, Any]) -> tuple:
    """
    Create one MultiServerMCPClient for all tasks in a method run.
    Returns (client, tools, interceptor).
    Call interceptor.set_manager(manager) before each task.
    """
    interceptor = MutableInterceptor()
    client = MultiServerMCPClient(
        connections=server_configs,
        tool_interceptors=[interceptor],
    )
    tools = await client.get_tools()
    return client, tools, interceptor


async def run_agent_with_tools(
    goal: str, tools: List[Any], max_steps: int,
) -> Dict[str, Any]:
    """Run agent with pre-built tools — no new MCP client created."""
    graph = StateGraph(MCPAgentState)

    async def agent_node(state):
        return await _agent_node(state, tools)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _route, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    compiled = graph.compile()

    return await compiled.ainvoke({
        "messages": [], "goal": goal, "step": 0,
        "max_steps": max_steps, "done": False, "final_answer": None,
    })


async def build_mcp_agent(
    goal:            str,
    manager:         Any,
    server_configs:  Dict[str, Any],
    max_steps:       int = 40,
) -> tuple:
    """
    Build a compiled LangGraph agent connected to real MCP servers.
    Uses langchain-mcp-adapters>=0.1.0 API (no async with context manager).
    Each tool call starts its own MCP session automatically.
    """
    manager.set_goal(goal)
    interceptor = GenericToolCallInterceptor(manager=manager)

    client = MultiServerMCPClient(
        connections=server_configs,
        tool_interceptors=[interceptor],
    )
    tools = await client.get_tools()

    graph = StateGraph(MCPAgentState)

    async def agent_node(state):
        return await _agent_node(state, tools)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _route, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    compiled = graph.compile()

    initial_state: MCPAgentState = {
        "messages":     [],
        "goal":         goal,
        "step":         0,
        "max_steps":    max_steps,
        "done":         False,
        "final_answer": None,
    }

    return compiled, initial_state
