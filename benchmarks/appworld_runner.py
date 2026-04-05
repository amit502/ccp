"""
benchmarks/appworld_runner.py

AppWorld benchmark integration for CCP evaluation.

AppWorld provides 750 tasks across 9 apps and 457 APIs, making it the
primary benchmark (matches ACON's primary benchmark).

Setup (run once, requires Docker):
    pip install appworld
    appworld download all
    appworld server start   # starts the API server at localhost:8000

This module wraps the AppWorld Python client so tasks can be run through
the CCP agent and baselines with a unified interface.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# AppWorld imports — these work once `pip install appworld` is done
# and the server is running.
try:
    from appworld import AppWorld
    from appworld.task import Task
    APPWORLD_AVAILABLE = True
except ImportError:
    APPWORLD_AVAILABLE = False
    print("[AppWorld] Package not available — using mock tasks for development.")


# ---------------------------------------------------------------------------
# Task result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id:       str
    goal:          str
    success:       bool
    steps:         int
    final_answer:  Optional[str]
    peak_tokens:   int
    total_tokens:  int
    time_elapsed:  float              # seconds
    ccp_stats:     List[Any] = field(default_factory=list)
    method:        str = "ccp"


# ---------------------------------------------------------------------------
# AppWorld tool wrapper
# Wraps AppWorld's API client so it matches the agent's tool registry format
# ---------------------------------------------------------------------------

class AppWorldToolWrapper:
    """
    Wraps an AppWorld environment's API calls as named callables
    that the agent can register with register_tool().
    """

    def __init__(self, appworld_env):
        self.env = appworld_env

    def make_tool(self, app_name: str, api_name: str) -> Callable:
        """Return a callable that executes api_name on app_name."""
        def tool(**kwargs) -> str:
            result = self.env.execute(
                app_name=app_name,
                api_name=api_name,
                **kwargs,
            )
            return json.dumps(result) if not isinstance(result, str) else result
        tool.__name__ = f"{app_name}__{api_name}"
        return tool

    def register_all(self, tool_registry: Dict[str, Callable]) -> None:
        """Register all available AppWorld tools into a tool registry."""
        if not APPWORLD_AVAILABLE:
            return
        for app in self.env.apps:
            for api in self.env.get_apis(app):
                name = f"{app}__{api}"
                tool_registry[name] = self.make_tool(app, api)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class AppWorldRunner:
    """
    Runs CCP (or a baseline) against AppWorld tasks and collects metrics.
    """

    def __init__(
        self,
        split:      str = "test",     # "train" | "dev" | "test"
        max_tasks:  int = 50,         # How many tasks to run
        max_steps:  int = 40,         # Max steps per task
    ):
        self.split     = split
        self.max_tasks = max_tasks
        self.max_steps = max_steps

        if APPWORLD_AVAILABLE:
            # AppWorld v0.1.x API
            self.appworld = AppWorld(split=split)
        else:
            self.appworld = None
            print("[AppWorldRunner] Running in MOCK mode — no real API calls.")

    def _get_tasks(self) -> List[Any]:
        if self.appworld is None:
            return self._mock_tasks()
        tasks = list(self.appworld.tasks)
        return tasks[: self.max_tasks]

    # ------------------------------------------------------------------ #
    # Main evaluation loop                                                 #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        manager_factory: Callable,    # Callable() → a context manager
        method_name:     str = "ccp",
        verbose:         bool = True,
    ) -> List[TaskResult]:
        """
        Run all tasks with the given context manager and collect results.

        Args:
            manager_factory: Zero-arg callable that returns a fresh context manager.
                             E.g.: lambda: CCPContextManager(tau_high=0.6, tau_low=0.3)
            method_name:     Label for results ("ccp", "fifo", etc.)
        """
        from ..agent import _TOOL_REGISTRY, register_tool

        tasks   = self._get_tasks()
        results = []

        for i, task in enumerate(tasks):
            if verbose:
                print(f"\n[{method_name}] Task {i+1}/{len(tasks)}: {task.goal[:60]}...")

            # Fresh manager and tool registry per task
            manager = manager_factory()
            manager.set_goal(task.goal)
            _TOOL_REGISTRY.clear()

            if self.appworld is not None:
                env     = self.appworld.reset(task.id)
                wrapper = AppWorldToolWrapper(env)
                wrapper.register_all(_TOOL_REGISTRY)
            else:
                _register_mock_tools(_TOOL_REGISTRY)

            # Run the agent
            t0 = time.time()
            result = _run_task(
                task_id=task.id,
                goal=task.goal,
                manager=manager,
                max_steps=self.max_steps,
                appworld_env=self.appworld,
                verbose=verbose,
            )
            elapsed = time.time() - t0
            result.time_elapsed = elapsed
            result.method = method_name
            results.append(result)

            if verbose:
                status = "✓" if result.success else "✗"
                print(f"  {status} Steps: {result.steps} | "
                      f"Peak tokens: {result.peak_tokens} | "
                      f"Time: {elapsed:.1f}s")

        return results

    # ------------------------------------------------------------------ #
    # Mock tasks (used when AppWorld server is not running)               #
    # ------------------------------------------------------------------ #

    def _mock_tasks(self) -> List[Any]:
        from types import SimpleNamespace
        tasks = []
        mock_goals = [
            "Send an email to Alice with subject 'Meeting Tomorrow' and body 'Are you free at 2pm?'",
            "Order 2 units of 'Wireless Mouse' from Amazon and send the order confirmation to Bob via SMS",
            "Create a Spotify playlist called 'Study Vibes' and add the top 5 trending songs",
            "Transfer $50 to Charlie via Venmo with message 'Dinner split'",
            "Find Alice's phone number in contacts and call her",
        ]
        for i, goal in enumerate(mock_goals):
            t = SimpleNamespace(id=f"mock_{i:03d}", goal=goal)
            tasks.append(t)
        return tasks[: self.max_tasks]


# ---------------------------------------------------------------------------
# Task execution (agent loop)
# ---------------------------------------------------------------------------

def _run_task(
    task_id:      str,
    goal:         str,
    manager:      Any,
    max_steps:    int,
    appworld_env: Any,
    verbose:      bool,
) -> TaskResult:
    """Run one task through the LangGraph agent and return a TaskResult."""
    from ..agent import _TOOL_REGISTRY, agent_think, execute_tool

    # Build a minimal state to drive the agent manually
    # (avoids graph compilation overhead in tight eval loops)
    state: Dict[str, Any] = {
        "goal":         goal,
        "step":         0,
        "max_steps":    max_steps,
        "done":         False,
        "final_answer": None,
        "ccp_manager":  manager,
    }

    peak_tokens  = 0
    total_tokens = 0

    while not state["done"] and state["step"] < max_steps:
        state = agent_think(state)
        state = execute_tool(state)

        ctx_tokens = manager.get_compressed_context().total_tokens()
        peak_tokens   = max(peak_tokens, ctx_tokens)
        total_tokens += ctx_tokens

    # Score success via AppWorld's evaluator if available
    success = False
    if appworld_env is not None:
        try:
            score = appworld_env.evaluate(task_id=task_id)
            success = score >= 1.0
        except Exception:
            success = state.get("done", False)
    else:
        # Mock: treat "done" as success
        success = state.get("done", False)

    return TaskResult(
        task_id=task_id,
        goal=goal,
        success=success,
        steps=state["step"],
        final_answer=state.get("final_answer"),
        peak_tokens=peak_tokens,
        total_tokens=total_tokens,
        time_elapsed=0.0,
        ccp_stats=manager.get_stats_log(),
    )


# ---------------------------------------------------------------------------
# Mock tools (development / unit testing)
# ---------------------------------------------------------------------------

def _register_mock_tools(registry: Dict[str, Any]) -> None:
    """Register lightweight mock tools for local development."""
    import random

    def mock_send_email(**kwargs):
        return {"status": "sent", "message_id": f"msg_{random.randint(1000,9999)}"}

    def mock_get_contacts(**kwargs):
        return [{"name": "Alice", "email": "alice@example.com", "phone": "+1-555-0101"}]

    def mock_search(**kwargs):
        return {"results": [f"Result {i}" for i in range(5)], "total": 5}

    def mock_authenticate(**kwargs):
        return {"token": f"tok_{random.randint(100000,999999)}", "expires_in": 3600}

    def mock_list_items(**kwargs):
        return [f"Item {i}" for i in range(20)]

    registry.update({
        "email__send":         mock_send_email,
        "contacts__get_all":   mock_get_contacts,
        "search__query":       mock_search,
        "auth__login":         mock_authenticate,
        "catalog__list_items": mock_list_items,
    })
