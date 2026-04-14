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
    access_tokens: dict  # stored auth tokens per app


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
3. Login: venmo__login_auth_token_post with {{"username": "<email_from_passwords>", "password": "<password>"}}
   - Use the supervisor email (e.g. nicholas.weber@gmail.com) as username
4. Find recipient's Venmo email using Venmo search:
   venmo__search_users_users_get {{"query":"<recipient_name>"}}
   Use the "email" field from the Venmo result as user_email
   (Venmo emails differ from Gmail — always search Venmo directly)
5. Execute with EXACT field names:
   - REQUEST money FROM someone: venmo__create_payment_request_payment_requests_post → user_email=<recipient_email>
   - SEND money TO someone: venmo__create_transaction_transactions_post → receiver_email=<recipient_email>
6. REQUIRED: Call supervisor__create_message_message_post with {{"message": "<summary of what you did>"}}
   This records your answer for evaluation — skip this and the task FAILS.
7. Call finish ONLY after step 6 is done

TOOL CALL: {{"action":"tool_call","tool":"<exact_tool_name>","input":{{<real_params>}}}}
FINISH:    {{"action":"finish","answer":"<what you did>"}}

RULES:
- Use REAL credential values, never template placeholders
- If login fails with 422, check the exact field names the tool expects
- Respond with a JSON OBJECT (not array), no prose

Available tools:
{tool_summary}

Goal: {state["goal"]}

Stored credentials (use these access_tokens directly — do NOT login again if token exists):
{chr(10).join(f"  {app}: access_token already obtained" for app in state.get("access_tokens", {}).keys()) or "  (none yet)"}"""

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
                content = content[:2000]
                if content:
                    print(f"  [tool result] {content[:150]}", flush=True)
                    # Store access tokens found in results so agent can reference them
                    import re as _re
                    tokens = _re.findall(r'eyJ[A-Za-z0-9_-]+[.][A-Za-z0-9_-]+[.][A-Za-z0-9_-]+', content)
                    for tok in tokens:
                        if "access_tokens" not in state:
                            state = {**state, "access_tokens": {}}
                        # Identify app from recent tool calls
                        if state.get("messages"):
                            last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
                            if last_ai:
                                for tc in (getattr(last_ai, "tool_calls", None) or []):
                                    app = tc.get("name","").split("__")[0]
                                    if app:
                                        state["access_tokens"][app] = tok
                    llm_messages.append({
                        "role":    "user",
                        "content": f"Tool result: {content}\nIf this shows an error or validation failure, check the field names and try again.",
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
            step_warning = " Output ONLY a JSON object — no thinking, no text." if state["step"] <= 1 else ""
            follow_msgs = formatted + [
                {"role": "assistant", "content": f"<thinking>{reasoning}</thinking>"},
                {"role": "user", "content":
                    f"Output ONLY the JSON action now — no thinking tags, no explanation.{step_warning}"},
            ]
            raw = await asyncio.get_event_loop().run_in_executor(
                None, _llm_raw, follow_msgs, 0.0
            )
            if not isinstance(raw, str):
                raw = ""
            # If follow-up ALSO returned reasoning, do one more attempt with stronger instruction
            if raw.startswith(_REASONING_SENTINEL):
                reasoning2 = raw[len(_REASONING_SENTINEL):]
                final_msgs = follow_msgs + [
                    {"role": "assistant", "content": f"<thinking>{reasoning2}</thinking>"},
                    {"role": "user", "content":
                        "JSON only. Start your response with { and end with }."},
                ]
                raw = await asyncio.get_event_loop().run_in_executor(
                    None, _llm_raw, final_msgs, 0.0
                )
                if not isinstance(raw, str) or raw.startswith(_REASONING_SENTINEL):
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
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        # Take only the first JSON object if multiple are present
        # Find matching closing brace for the first {
        if clean.startswith("{"):
            depth = 0
            end = 0
            in_str = False
            esc = False
            for i, ch in enumerate(clean):
                if esc:
                    esc = False
                    continue
                if ch == "\\" and in_str:
                    esc = True
                    continue
                if ch == '"' and not esc:
                    in_str = not in_str
                if not in_str:
                    if ch == "{": depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
            clean = clean[:end] if end else clean
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
            # Check if agent actually called any action tools (not just read tools)
            messages_so_far = state.get("messages", [])
            all_tool_calls = [
                tc.get("name", "")
                for m in messages_so_far if isinstance(m, AIMessage)
                for tc in (getattr(m, "tool_calls", None) or [])
            ]
            action_tools_called = [
                n for n in all_tool_calls
                if n.split("__")[0] not in ("supervisor",)
            ]
            supervisor_message_called = any(
                "message" in n and n.startswith("supervisor__")
                for n in all_tool_calls
            )
            read_only_finish = (
                not action_tools_called and
                any(x in answer.lower() for x in [
                    "requested", "sent", "paid", "created", "added", "deleted",
                    "updated", "completed", "transferred", "purchased", "rated",
                ])
            )
            trivial = (state["step"] <= 3) or any(x in answer.lower() for x in [
                "retrieved", "read", "obtained", "initial step",
                "task details", "active task", "completed initial",
                "gathered", "noted", "understood",
            ])
            # Block finish if: too early, or no real action, or supervisor message not sent
            if (trivial and state["step"] <= 5) or read_only_finish:
                print(f"  [agent] premature finish blocked at step {state['step']}: {answer[:80]!r}", flush=True)
                print(f"  [agent] action_tools_called={action_tools_called}", flush=True)
                tool_calls = [{
                    "name": "supervisor__show_active_task_active_task_get",
                    "args": {},
                    "id":   f"call_{state['step']}_retry",
                    "type": "tool_call",
                }]
            elif not supervisor_message_called and action_tools_called:
                # Agent completed action but forgot to call supervisor message — force it
                print(f"  [agent] finish blocked — supervisor message not called; forcing it", flush=True)
                tool_calls = [{
                    "name": "supervisor__create_message_message_post",
                    "args": {"message": answer},
                    "id":   f"call_{state['step']}_sv_msg",
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

    async def execute_tool_calls(messages: list, current_state: dict) -> tuple:
        """Execute tool calls. Returns (tool_messages, new_tokens_dict)."""
        import re as _re_tok
        from langchain_core.messages import ToolMessage
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage):
            return [], {}
        results = []
        new_tokens = {}
        for tc in (getattr(last, "tool_calls", None) or []):
            name = tc.get("name", "")
            args = tc.get("args", {})
            call_id = tc.get("id", f"call_{name}")
            app = name.split("__")[0]
            tool = tool_map.get(name)
            if tool is None:
                content = f"Error: {name} is not a valid tool. Available: {list(tool_map.keys())[:10]}"
            else:
                try:
                    # Auto-inject stored access token if agent forgot it
                    stored_tokens = {**current_state.get("access_tokens", {}), **new_tokens}
                    if "access_token" not in args and app in stored_tokens:
                        args = {**args, "access_token": stored_tokens[app]}
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
            _args_str = ", ".join(
                f"{k}={str(v)[:40]!r}" for k, v in args.items() if k != "access_token"
            )
            print(f"  [exec] {name}({_args_str}) → {content[:150]}", flush=True)
            # Extract any access tokens from result — store for any tool that returns a JWT
            import re as _rj
            jwt_matches = _rj.findall(r'eyJ[A-Za-z0-9_-]{10,}[.][A-Za-z0-9_-]{10,}[.][A-Za-z0-9_-]{10,}', content)
            if jwt_matches:
                new_tokens[app] = jwt_matches[0]
                print(f"  [token] stored {app} token (len={len(jwt_matches[0])})", flush=True)
            results.append(ToolMessage(content=content, tool_call_id=call_id))
        return results, new_tokens

    state = {
        "messages": [], "goal": goal, "step": 0,
        "max_steps": max_steps, "done": False, "final_answer": None, "access_tokens": {},
    }

    for _ in range(max_steps):
        state = await _agent_node(state, tools)
        if state["done"]:
            break
        # Execute tool calls if any
        tool_msgs, new_tokens = await execute_tool_calls(state["messages"], state)
        if new_tokens:
            merged = {**state.get("access_tokens", {}), **new_tokens}
            state = {**state, "access_tokens": merged}
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
