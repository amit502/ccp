"""
agent.py

LangGraph-based agent with CCP (Causal Context Pruning) integrated.

Architecture (matches Figure 1 in the proposal):

    Task Goal g
        |
        v
    LangGraph Agent ← Compressed Context C*_t
        |                       ^
        | tool call             |
        v                       |
    MCP Tool Layer   ──raw observation──> CCP Module (causal scorer)

The CCP module is implemented as a LangGraph node that intercepts tool
responses BEFORE they enter the agent's context.

For AppWorld: AppWorld's 457 APIs are accessed via a Python client; in a
full MCP deployment they would be wrapped as MCP servers.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .context_manager import CCPContextManager
from .llm_client import Message, call_llm, gpt_chat
from .models import AgentContext, ContextElement

# ---------------------------------------------------------------------------
# Agent state (the object flowing through LangGraph nodes)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    goal:          str
    step:          int
    max_steps:     int
    done:          bool
    final_answer:  Optional[str]
    # CCP manages context internally; we pass the manager via state
    ccp_manager:   CCPContextManager


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_AGENT_SYSTEM = """\
You are a task-completion agent with access to tools (APIs).
At each step you MUST respond with a JSON object in one of two formats:

1. To call a tool:
{
  "action": "tool_call",
  "tool":   "<tool_name>",
  "input":  {<tool parameters as a JSON object>}
}

2. When the task is complete:
{
  "action": "finish",
  "answer": "<your final answer>"
}

Respond with JSON only. No markdown, no preamble.
"""

def _build_agent_prompt(context: AgentContext) -> str:
    return (
        f"Current context:\n{context.to_prompt_str()}\n\n"
        "What is your next action?"
    )


# ---------------------------------------------------------------------------
# Node: agent_think
# Calls the LLM to decide the next action given compressed context
# ---------------------------------------------------------------------------

def agent_think(state: AgentState) -> AgentState:
    manager: CCPContextManager = state["ccp_manager"]
    context = manager.get_compressed_context()

    system = _AGENT_SYSTEM
    user   = _build_agent_prompt(context)

    raw = call_llm(system_prompt=system, user_prompt=user)

    # Parse the agent's decision
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        decision = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        # If LLM returns malformed JSON, treat as a no-op and continue
        print(f"[Agent] Malformed decision at step {state['step']}: {raw[:200]}")
        decision = {"action": "tool_call", "tool": "noop", "input": {}}

    state["_decision"] = decision  # type: ignore[index]
    state["step"] = state["step"] + 1
    return state


# ---------------------------------------------------------------------------
# Node: execute_tool
# Executes the tool call and feeds the result through CCP
# ---------------------------------------------------------------------------

# The tool registry maps tool names to callables.
# In the real system, these are replaced by AppWorld / MCP server calls.
_TOOL_REGISTRY: Dict[str, Any] = {}

def register_tool(name: str, fn) -> None:
    """Register an AppWorld / MCP tool so the agent can call it."""
    _TOOL_REGISTRY[name] = fn


def execute_tool(state: AgentState) -> AgentState:
    decision = state.get("_decision", {})  # type: ignore[call-overload]

    if decision.get("action") == "finish":
        state["done"] = True
        state["final_answer"] = decision.get("answer", "")
        return state

    tool_name  = decision.get("tool", "unknown")
    tool_input = decision.get("input", {})

    # ── Call the actual tool ────────────────────────────────────────────
    tool_fn = _TOOL_REGISTRY.get(tool_name)
    if tool_fn is not None:
        try:
            raw_output = tool_fn(**tool_input)
            status     = "ok"
        except Exception as exc:
            raw_output = str(exc)
            status     = "error"
    else:
        raw_output = f"[Tool '{tool_name}' not registered]"
        status     = "error"

    # ── CCP intercepts the raw observation ─────────────────────────────
    manager: CCPContextManager = state["ccp_manager"]
    manager.add_observation(
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=str(raw_output),
        status=status,
    )

    return state


# ---------------------------------------------------------------------------
# Routing: should the agent continue or stop?
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    if state["done"]:
        return "end"
    if state["step"] >= state["max_steps"]:
        return "end"
    return "think"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_ccp_agent(
    goal:        str,
    max_steps:   int = 40,
    tau_high:    float = 0.6,
    tau_low:     float = 0.3,
    token_threshold: int = 4000,
    use_heuristics: bool = True,
) -> tuple[Any, AgentState]:
    """
    Build and return a compiled LangGraph agent with CCP enabled.

    Returns:
        (compiled_graph, initial_state)
    """
    manager = CCPContextManager(
        tau_high=tau_high,
        tau_low=tau_low,
        token_threshold=token_threshold,
        use_heuristics=use_heuristics,
    )
    manager.set_goal(goal)

    initial_state: AgentState = {
        "goal":        goal,
        "step":        0,
        "max_steps":   max_steps,
        "done":        False,
        "final_answer": None,
        "ccp_manager": manager,
    }

    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("think",        agent_think)
    graph.add_node("execute_tool", execute_tool)

    graph.set_entry_point("think")
    graph.add_edge("think", "execute_tool")
    graph.add_conditional_edges(
        "execute_tool",
        should_continue,
        {"think": "think", "end": END},
    )

    compiled = graph.compile()
    return compiled, initial_state


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_agent(
    goal: str,
    max_steps: int = 40,
    tau_high: float = 0.6,
    tau_low:  float = 0.3,
    token_threshold: int = 4000,
    use_heuristics: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the CCP agent on a single task.

    Returns a result dict with final_answer, stats, and step count.
    """
    compiled, initial_state = build_ccp_agent(
        goal=goal,
        max_steps=max_steps,
        tau_high=tau_high,
        tau_low=tau_low,
        token_threshold=token_threshold,
        use_heuristics=use_heuristics,
    )

    final_state = compiled.invoke(initial_state)

    manager: CCPContextManager = final_state["ccp_manager"]
    stats_log = manager.get_stats_log()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {goal}")
        print(f"Steps: {final_state['step']}")
        print(f"Done: {final_state['done']}")
        print(f"Answer: {final_state['final_answer']}")
        if stats_log:
            total_reduction = sum(s.token_reduction_pct for s in stats_log)
            print(f"Avg token reduction per compression: {total_reduction/len(stats_log):.1f}%")
        print(f"{'='*60}\n")

    return {
        "goal":         goal,
        "final_answer": final_state["final_answer"],
        "steps":        final_state["step"],
        "success":      final_state["done"],
        "stats_log":    stats_log,
    }
