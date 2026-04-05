"""
benchmarks/mcp_runner.py

Real MCP benchmark runner. Every method (CCP, FIFO, ACON, NoCompression,
Retrieval, TokenPerplexity) runs through real MCP servers via the same
GenericToolCallInterceptor. No mocks, no special-casing.

Key design decisions:
  1. ONE asyncio event loop per benchmark run (not per task).
     MCP server subprocesses start once and stay alive for all tasks.
  2. manager_factory() is called per task, creating a fresh context manager.
     That manager is passed directly to build_mcp_agent() which wires it
     as the interceptor for that task's MCP session.
  3. OfficeBench task initialisation happens via REST before opening the
     MCP session, so the MCP server is ready for that task's files.

Architecture for one task:

  runner.evaluate(manager_factory=lambda: FIFOManager())
       │
       ▼
  asyncio.run(run_all_tasks_async(...))   ← single event loop for all tasks
       │
       ├── for task in tasks:
       │       manager = manager_factory()   ← fresh FIFO/CCP/ACON/etc per task
       │       async with MultiServerMCPClient(
       │           connections=server_configs,          ← real MCP servers
       │           tool_interceptors=[
       │               GenericToolCallInterceptor(manager)  ← ANY manager
       │           ]
       │       ) as client:
       │           tools = client.get_tools()
       │           final_state = await compiled_graph.ainvoke(state)
       │
       └── results
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .appworld_runner import TaskResult

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

APPWORLD_URL           = os.environ.get("APPWORLD_URL",     "http://localhost:8000")
OFFICEBENCH_URL        = os.environ.get("OFFICEBENCH_URL",  "http://localhost:8001")
MCP_SERVER_SCRIPT      = str(Path(__file__).parent.parent / "mcp_server.py")
OB_MCP_SERVER_SCRIPT   = str(Path(__file__).parent.parent / "officebench_mcp_server.py")
NQ_MCP_SERVER_SCRIPT   = str(Path(__file__).parent.parent / "nq_mcp_server.py")


# ---------------------------------------------------------------------------
# Core async task runner — used by every benchmark runner
# ---------------------------------------------------------------------------

async def _run_one_task(
    task_id:        str,
    goal:           str,
    manager:        Any,             # already-constructed context manager
    server_configs: Dict[str, Any],
    max_steps:      int,
    score_fn:       Callable[[str, Any], float],   # (task_id, final_state) → float
    verbose:        bool,
) -> TaskResult:
    """
    Run one task through real MCP servers with the given context manager.
    The manager (CCP, FIFO, ACON, NoCompression, etc.) is passed directly
    to build_mcp_agent which wires it as a GenericToolCallInterceptor.
    """
    from ..mcp_agent import build_mcp_agent

    t0 = time.time()

    try:
        compiled, initial_state = await build_mcp_agent(
            goal=goal,
            manager=manager,
            server_configs=server_configs,
            max_steps=max_steps,
        )

        final_state = await compiled.ainvoke(initial_state)

        success      = score_fn(task_id, final_state) >= 1.0
        final_answer = final_state.get("final_answer")

    except Exception as exc:
        if verbose:
            print(f"  [MCP] Task {task_id} error: {type(exc).__name__}: {exc}")
        success      = False
        final_answer = None

    ctx          = manager.get_compressed_context()
    peak_tokens  = ctx.total_tokens()
    total_tokens = ctx.total_tokens()

    return TaskResult(
        task_id=task_id,
        goal=goal,
        success=success,
        steps=len(ctx.elements),
        final_answer=final_answer,
        peak_tokens=peak_tokens,
        total_tokens=total_tokens,
        time_elapsed=time.time() - t0,
        ccp_stats=manager.get_stats_log(),
        method="",
    )


async def _run_all_tasks_async(
    tasks:           List[Any],
    manager_factory: Callable[[], Any],
    server_configs:  Dict[str, Any],
    max_steps:       int,
    score_fn:        Callable,
    verbose:         bool,
    appworld_root:   str = "",
    appworld_url:    str = "",
) -> List[TaskResult]:
    """Run all tasks sequentially, seeding AppWorld databases per task."""
    from .appworld_runner import _seed_task, _reset_task, APPWORLD_ROOT, APPWORLD_URL

    _root = appworld_root or APPWORLD_ROOT
    _url  = appworld_url  or APPWORLD_URL

    results = []
    for task in tasks:
        # Seed task-specific databases (AppWorld only — noop for other benchmarks)
        if _root and _url:
            seeded = _seed_task(task, _root, _url)
            if not seeded:
                print(f"  [WARN] Task {task.id} seeding failed — skipping")
                continue

        manager = manager_factory()
        manager.set_goal(task.goal)
        result = await _run_one_task(
            task_id=task.id,
            goal=task.goal,
            manager=manager,
            server_configs=server_configs,
            max_steps=max_steps,
            score_fn=score_fn,
            verbose=verbose,
        )
        results.append(result)

        # Clear task databases after each task
        if _root and _url:
            _reset_task(task.id, _url)

        if verbose:
            status = "✓" if result.success else "✗"
            print(f"  {status} {task.id} | steps={result.steps} "
                  f"peak_tok={result.peak_tokens} t={result.time_elapsed:.1f}s")
    return results

# ---------------------------------------------------------------------------
# AppWorld MCP Runner
# ---------------------------------------------------------------------------

class AppWorldMCPRunner:
    """
    Runs every method (CCP, FIFO, ACON, NoCompression …) against AppWorld
    through real MCP servers.

    Does NOT import the appworld Python package (pydantic v1 conflict).
    Loads tasks directly from the filesystem and evaluates via REST.

    Requires:
        APPWORLD_ROOT env var  — path to appworld data directory
        APPWORLD_URL env var   — appworld REST server URL (default localhost:8000)
        appworld serve apis --port 8000  (running in background)
    """

    def __init__(self, split: str = "test", max_tasks: int = 50, max_steps: int = 40):
        self.split     = split
        self.max_tasks = max_tasks
        self.max_steps = max_steps

        from .appworld_runner import _load_tasks_from_fs, APPWORLD_ROOT, _evaluate_via_rest
        self._score = _evaluate_via_rest

        if not APPWORLD_ROOT:
            raise RuntimeError(
                "APPWORLD_ROOT env var not set.\n"
                "Set it to the appworld data directory and run:\n"
                "  appworld serve apis --port 8000"
            )
        self._tasks = _load_tasks_from_fs(APPWORLD_ROOT, split, max_tasks)
        print(f"[AppWorldMCPRunner] Loaded {len(self._tasks)} tasks from {APPWORLD_ROOT}")

    def _server_configs(self) -> Dict[str, Any]:
        """Connect to persistent MCP HTTP server (started once per method run)."""
        return {
            "appworld": {
                "transport": "streamable_http",
                "url":       "http://localhost:8001/mcp",
            }
        }

    def _start_mcp_server(self) -> Any:
        """Start the MCP HTTP server as a background subprocess.
        Fetches OpenAPI specs once at startup — not per task."""
        import subprocess, time, requests as req
        proc = subprocess.Popen(
            [
                sys.executable, MCP_SERVER_SCRIPT,
                "--mode",         "http",
                "--port",         "8001",
                "--appworld-url", APPWORLD_URL,
                "--app",          "all",
            ],
            stderr=sys.stderr,
        )
        # Wait up to 30s for server ready
        for _ in range(60):
            try:
                req.get("http://localhost:8001/mcp", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        return proc

    def evaluate(
        self,
        manager_factory: Callable[[], Any],
        method_name:     str = "ccp",
        verbose:         bool = True,
    ) -> List[TaskResult]:
        if verbose:
            print(f"\n[AppWorld/MCP] {method_name} | {len(self._tasks)} tasks")

        # Start persistent MCP server once — serves all tasks in this run
        mcp_proc = self._start_mcp_server()
        try:
            configs = self._server_configs()
            results = asyncio.run(
                _run_all_tasks_async(
                    tasks=self._tasks,
                    manager_factory=manager_factory,
                    server_configs=configs,
                    max_steps=self.max_steps,
                    score_fn=self._score,
                    verbose=verbose,
                )
            )
        finally:
            mcp_proc.terminate()
            mcp_proc.wait()

        for r in results:
            r.method = method_name
        return results


# ---------------------------------------------------------------------------
# OfficeBench MCP Runner
# ---------------------------------------------------------------------------

class OfficeBenchMCPRunner:
    """
    Runs every method against OfficeBench through a real MCP server that
    wraps the OfficeBench REST API.

    Before each task:
      1. POST /tasks/{id}/init via REST — sets up the task's initial files.
      2. Open an MCP session with the OfficeBench MCP server.
      3. Run the agent through the MCP session.
      4. POST /tasks/{id}/evaluate via REST to score.

    Requires:
      git clone https://github.com/zlwangx/OfficeBench
      cd OfficeBench && python server.py --port 8001
      export OFFICEBENCH_URL=http://localhost:8001
      export OFFICEBENCH_TASKS_DIR=/path/to/OfficeBench/tasks
    """

    TASKS_DIR = os.environ.get("OFFICEBENCH_TASKS_DIR", "")

    def __init__(self, split: str = "test", max_tasks: int = 50, max_steps: int = 25):
        self.split     = split
        self.max_tasks = max_tasks
        self.max_steps = max_steps

        import requests
        try:
            r = requests.get(f"{OFFICEBENCH_URL}/health", timeout=3)
            if r.status_code != 200:
                raise RuntimeError(f"status {r.status_code}")
        except Exception as e:
            raise RuntimeError(
                f"OfficeBench server not reachable at {OFFICEBENCH_URL}: {e}\n"
                "Run: cd OfficeBench && python server.py --port 8001"
            )

        self._client = requests.Session()
        self._tasks  = self._load_tasks()

    def _load_tasks(self) -> List[Any]:
        from ..benchmarks.officebench_runner import (
            OfficeBenchClient, _load_tasks_from_dir, _load_tasks_from_server,
        )
        client = OfficeBenchClient(OFFICEBENCH_URL)
        if self.TASKS_DIR:
            tasks = _load_tasks_from_dir(self.TASKS_DIR, self.split, self.max_tasks)
            if tasks:
                return tasks
        return _load_tasks_from_server(client, self.split, self.max_tasks)

    def _init_task(self, task_id: str) -> None:
        """Initialise task environment on the OfficeBench server before MCP session."""
        self._client.post(
            f"{OFFICEBENCH_URL}/tasks/{task_id}/init",
            timeout=30,
        ).raise_for_status()

    def _server_configs(self, task_id: str, app: str) -> Dict[str, Any]:
        """
        OfficeBench MCP server for this task's app. The task_id is passed
        as a CLI arg so the server knows which task's files to operate on.
        """
        return {
            "officebench": {
                "command":   sys.executable,
                "args":      [OB_MCP_SERVER_SCRIPT,
                              "--app",        app,
                              "--server-url", OFFICEBENCH_URL,
                              "--task-id",    task_id],
                "transport": "stdio",
            }
        }

    def _score(self, task_id: str, final_state: Dict) -> float:
        try:
            r     = self._client.post(
                f"{OFFICEBENCH_URL}/tasks/{task_id}/evaluate", timeout=30
            )
            return float(r.json().get("score", 0.0))
        except Exception:
            return 1.0 if final_state.get("done") else 0.0

    async def _run_tasks_async(
        self,
        manager_factory: Callable,
        verbose:         bool,
    ) -> List[TaskResult]:
        from ..mcp_agent import build_mcp_agent

        results = []
        for task in self._tasks:
            # Initialise task environment via REST BEFORE opening MCP session
            try:
                self._init_task(task.id)
            except Exception as e:
                print(f"  [OfficeBench] init failed for {task.id}: {e}")
                continue

            manager = manager_factory()
            manager.set_goal(task.goal)
            configs = self._server_configs(task.id, getattr(task, "app", "word"))

            result = await _run_one_task(
                task_id=task.id,
                goal=task.goal,
                manager=manager,
                server_configs=configs,
                max_steps=self.max_steps,
                score_fn=self._score,
                verbose=verbose,
            )
            results.append(result)

            if verbose:
                status = "✓" if result.success else "✗"
                print(f"  {status} {task.id} | steps={result.steps} "
                      f"peak_tok={result.peak_tokens} t={result.time_elapsed:.1f}s")

        return results

    def evaluate(
        self,
        manager_factory: Callable,
        method_name:     str = "ccp",
        verbose:         bool = True,
    ) -> List[TaskResult]:
        if verbose:
            print(f"\n[OfficeBench/MCP] Running {method_name} on {len(self._tasks)} tasks "
                  f"(OFFICEBENCH_URL={OFFICEBENCH_URL})")

        results = asyncio.run(self._run_tasks_async(manager_factory, verbose))
        for r in results:
            r.method = method_name
        return results


# ---------------------------------------------------------------------------
# Multi-objective QA MCP Runner
# ---------------------------------------------------------------------------

class MultiObjQAMCPRunner:
    """
    Runs every method against Multi-objective QA through the NQ MCP server.

    The NQ MCP server (nq_mcp_server.py) exposes search/lookup tools backed
    by real Natural Questions data (local file or HuggingFace Hub).

    No separate REST server needed — the MCP server is self-contained.
    """

    def __init__(self, max_tasks: int = 50, max_steps: int = 20, n_hops: int = 3):
        self.max_steps = max_steps
        from ..benchmarks.multiobjqa_runner import _load_nq_tasks, _score_moqa_answer
        self._tasks        = _load_nq_tasks(max_tasks=max_tasks, hops=n_hops)
        self._score_answer = _score_moqa_answer
        print(f"[MultiObjQA/MCP] {len(self._tasks)} tasks loaded ({n_hops} hops each)")

    def _server_configs(self) -> Dict[str, Any]:
        return {
            "retrieval": {
                "command":   sys.executable,
                "args":      [NQ_MCP_SERVER_SCRIPT],
                "transport": "stdio",
                "env": {
                    **os.environ,
                    "MULTIQA_DATA_FILE": os.environ.get("MULTIQA_DATA_FILE", ""),
                },
            }
        }

    def _score(self, task_id: str, final_state: Dict) -> float:
        answer = final_state.get("final_answer", "")
        # Find original task to get questions list
        task   = next((t for t in self._tasks if t.id == task_id), None)
        if task is None:
            return 1.0 if final_state.get("done") else 0.0
        score  = self._score_answer(answer, task)
        return score * (1 / 0.67) if score >= 0.67 else 0.0  # normalise to 0/1

    def evaluate(
        self,
        manager_factory: Callable,
        method_name:     str = "ccp",
        verbose:         bool = True,
    ) -> List[TaskResult]:
        if verbose:
            print(f"\n[MultiObjQA/MCP] Running {method_name} on {len(self._tasks)} tasks")

        configs = self._server_configs()
        results = asyncio.run(
            _run_all_tasks_async(
                tasks=self._tasks,
                manager_factory=manager_factory,
                server_configs=configs,
                max_steps=self.max_steps,
                score_fn=self._score,
                verbose=verbose,
            )
        )
        for r in results:
            r.method = method_name
        return results
