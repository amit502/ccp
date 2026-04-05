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
from contextlib import asynccontextmanager
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
            tool_input=dict(request.arguments or {}),
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
You are a task-completion agent. Use the available tools to complete the goal.
When the task is done, respond with exactly:
  FINAL ANSWER: <your answer>
Otherwise call the most appropriate tool.
"""


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

async def _agent_node(state: MCPAgentState, tools: List[Any]) -> MCPAgentState:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "gpt-oss"),
        base_url=os.environ.get("NAUTILUS_BASE_URL", ""),
        api_key=os.environ.get("NAUTILUS_API_KEY", ""),
        temperature=0.0,
    ).bind_tools(tools)

    messages = state["messages"] or [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=state["goal"]),
    ]

    response = await llm.ainvoke(messages)

    done = False
    final_answer = None
    if isinstance(response, AIMessage) and response.content:
        content = str(response.content)
        if "FINAL ANSWER:" in content:
            done = True
            final_answer = content.split("FINAL ANSWER:", 1)[1].strip()

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

@asynccontextmanager
async def build_mcp_agent(
    goal:            str,
    manager:         Any,           # CCPContextManager | FIFOManager | ACON | …
    server_configs:  Dict[str, Any],
    max_steps:       int = 40,
):
    """
    Build and yield a compiled LangGraph agent connected to real MCP servers,
    with the provided context manager wired as a GenericToolCallInterceptor.

    Usage:
        async with build_mcp_agent(goal, manager, server_configs) as (graph, state):
            final = await graph.ainvoke(state)
    """
    manager.set_goal(goal)
    interceptor = GenericToolCallInterceptor(manager=manager)

    async with MultiServerMCPClient(
        connections=server_configs,
        tool_interceptors=[interceptor],
    ) as mcp_client:

        tools = mcp_client.get_tools()

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

        yield compiled, initial_state
