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
    from llm_client import _get_client, MODEL

    # Build tool list for the prompt
    tool_list = "\n".join(
        f"  {t.name}: {t.description}" for t in tools
    )

    # Build conversation history — include tool results for context
    history_parts = []
    for m in state["messages"]:
        if isinstance(m, SystemMessage):
            pass
        elif isinstance(m, HumanMessage):
            history_parts.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            if m.content:
                history_parts.append(f"Assistant: {m.content}")
            # Show tool calls made
            for tc in (getattr(m, "tool_calls", None) or []):
                history_parts.append(
                    f"Tool called: {tc.get('name','?')} with {tc.get('args',{})}"
                )
        else:
            # ToolMessage — show the result
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            if content:
                history_parts.append(f"Tool result: {str(content)[:500]}")
    history = "\n".join(history_parts)

    # Show only tool names + first sentence of description to save context
    tool_summary = "\n".join(
        f"  {t.name}: {t.description.split('.')[0]}" for t in tools[:60]
    )
    if len(tools) > 60:
        tool_summary += f"\n  ... and {len(tools)-60} more tools"

    system = f"""You are an autonomous agent completing tasks by calling real API tools.

RULES — follow these exactly:
1. You are in a REAL environment. Tools work. Call them.
2. NEVER say "I cannot", "tool execution not possible", or ask the user for info.
3. Start by calling supervisor__show_active_task to get full task details.
4. Call tools step by step until the task is fully done.
5. Only call {{"action":"finish",...}} AFTER you have actually completed all actions.
6. Respond ONLY with JSON — no prose, no explanation.

TOOL CALL format:
{{"action": "tool_call", "tool": "<exact_tool_name>", "input": {{"<param>": "<value>"}}}}

FINISH format (only after completing ALL required actions):
{{"action": "finish", "answer": "<what you did>"}}

Available tools:
{tool_summary}

Goal: {{goal}}"""

    # Inject goal into system prompt
    system = system.replace("{goal}", state["goal"])

    # User message: history of tool calls so far, or initial prompt
    user = history.strip() if history.strip() else "Begin. Call the first tool now."

    # Call LLM with JSON mode
    try:
        client = _get_client()
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.0,
        )
        raw = completion.choices[0].message.content or ""
        if not raw:
            print(
                f"  [LLM] empty content. "
                f"finish_reason={completion.choices[0].finish_reason!r} "
                f"usage={completion.usage}",
                flush=True,
            )
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
        SystemMessage(content=system),
        HumanMessage(content=user),
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
