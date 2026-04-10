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

    # Group tools by app — show ALL names so agent can pick correct ones
    from collections import defaultdict
    app_tools = defaultdict(list)
    for t in tools:
        app = t.name.split("__")[0] if "__" in t.name else "other"
        app_tools[app].append(t.name)

    tool_summary_parts = []
    for app, tnames in sorted(app_tools.items()):
        tool_summary_parts.append(f"  [{app}]: {', '.join(tnames)}")
    tool_summary = "\n".join(tool_summary_parts)

    system_text = f"""You are an autonomous agent. Complete the task by calling tools in sequence.

STRICT SEQUENCE:
1. Call supervisor__show_active_task_active_task_get → read the "instruction" field
2. Call supervisor__show_account_passwords_account_passwords_get → get credentials
3. Login to required app: use the EXACT field names the API expects:
   - For Venmo: venmo__login_auth_token_post with {{"username": "<value>", "password": "<value>"}}
   - The "account_name" from passwords = the "username" for login
   - NEVER use placeholders like {{{{email}}}} — use the REAL values from step 2
4. Execute the required actions (send payment, request money, etc.)
5. Call finish ONLY after completing all actions

TOOL CALL: {{"action":"tool_call","tool":"<exact_tool_name>","input":{{<real_params>}}}}
FINISH:    {{"action":"finish","answer":"<what you did>"}}

RULES:
- Use REAL credential values, never template placeholders
- If login fails with 422, check the exact field names the tool expects
- Respond with a JSON OBJECT (not array), no prose

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
                # ToolMessage — extract text from content blocks if needed
                raw_content = getattr(m, "content", "") or ""
                if isinstance(raw_content, list):
                    parts = []
                    for block in raw_content:
                        if isinstance(block, dict) and "text" in block:
                            # Try to parse nested JSON for readability
                            try:
                                inner = json.loads(block["text"])
                                parts.append(json.dumps(inner))
                            except Exception:
                                parts.append(block["text"])
                        elif isinstance(block, str):
                            parts.append(block)
                    content = "\n".join(parts)
                else:
                    content = str(raw_content)
                content = content[:800]
                if content:
                    print(f"  [tool result] {content[:150]}", flush=True)
                    llm_messages.append({
                        "role":    "user",
                        "content": f"Tool result: {content}\nIf this shows an error or validation failure, check the field names and try again with correct values.",
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
            # If on step 0 or 1, prevent premature finish — must call tools first
            step_warning = ""
            if state["step"] <= 1:
                step_warning = (
                    " IMPORTANT: Do NOT output finish yet — "
                    "you have only just started. Call the next tool."
                )
            follow_msgs = formatted + [
                {"role": "assistant", "content": f"<thinking>{reasoning}</thinking>"},
                {"role": "user", "content":
                    f"Based on your reasoning, output ONLY the JSON action now. "
                    f"No thinking tags — just the JSON object.{step_warning}"},
            ]
            raw = await asyncio.get_event_loop().run_in_executor(
                None, _llm_raw, follow_msgs, 0.0
            )
            if not isinstance(raw, str):
                raw = ""
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
        # LLM sometimes wraps in an array — unwrap it
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        action = parsed.get("action", "")

        # Detect finish — but not if we've done fewer than 3 tool calls
        # (prevents finishing after just reading the task)
        if (action == "finish"
                or parsed.get("answer") is not None
                or "FINAL ANSWER" in str(parsed.get("answer", ""))):
            answer = str(parsed.get("answer", parsed.get("result", "")))
            trivial = any(x in answer.lower() for x in [
                "retrieved", "read", "obtained", "got the task",
                "task details", "active task",
            ])
            if trivial and state["step"] <= 3:
                # Force continuation — agent hasn't done real work yet
                print(f"  [agent] premature finish blocked at step {state['step']}: {answer[:80]!r}", flush=True)
                tool_calls = [{
                    "name": "supervisor__show_active_task_active_task_get",
                    "args": {},
                    "id":   f"call_{state['step']}_retry",
                    "type": "tool_call",
                }]
            else:
                done         = True
                final_answer = answer

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

        print(f"  [interceptor] tool={request.name} output={raw_output[:80]!r}", flush=True)

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
    """
    Run agent with pre-built tools.
    Uses a custom tool executor instead of ToolNode to avoid
    ToolNode starting a fresh MCP subprocess per call.
    """
    venmo_tools = [t.name for t in tools if "venmo" in t.name.lower()]
    print(f"  [tools] total={len(tools)} venmo={venmo_tools[:5]}", flush=True)

    # Build tool lookup by name
    tool_map = {t.name: t for t in tools}

    async def execute_tool_calls(messages: list) -> list:
        """Execute tool calls from the last AIMessage and return ToolMessages."""
        from langchain_core.messages import ToolMessage
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage):
            return []
        results = []
        for tc in (getattr(last, "tool_calls", None) or []):
            name = tc.get("name", "")
            args = tc.get("args", {})
            call_id = tc.get("id", f"call_{name}")
            tool = tool_map.get(name)
            if tool is None:
                content = f"Error: {name} is not a valid tool. Available: {list(tool_map.keys())[:10]}"
            else:
                try:
                    raw_result = await tool.ainvoke(args)
                    # Extract text from content block format: [{"type":"text","text":"..."}]
                    if isinstance(raw_result, list):
                        parts = []
                        for block in raw_result:
                            if isinstance(block, dict) and "text" in block:
                                parts.append(block["text"])
                            elif isinstance(block, str):
                                parts.append(block)
                            else:
                                parts.append(json.dumps(block))
                        content = "\n".join(parts)
                    elif isinstance(raw_result, str):
                        content = raw_result
                    else:
                        content = json.dumps(raw_result)
                except Exception as e:
                    content = f"Tool error: {e}"
            print(f"  [exec] {name}({list(args.keys())}) → {content[:120]}", flush=True)
            results.append(ToolMessage(content=content, tool_call_id=call_id))
        return results

    state = {
        "messages": [], "goal": goal, "step": 0,
        "max_steps": max_steps, "done": False, "final_answer": None,
    }

    for _ in range(max_steps):
        state = await _agent_node(state, tools)
        if state["done"]:
            break
        # Execute tool calls if any
        tool_msgs = await execute_tool_calls(state["messages"])
        if tool_msgs:
            state = {**state, "messages": state["messages"] + tool_msgs}
        elif not state["done"]:
            # No tool calls and not done — stuck, break
            break

    return state


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
