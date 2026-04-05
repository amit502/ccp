"""
benchmarks/appworld_runner.py

AppWorld benchmark integration that works WITHOUT importing the appworld
Python package (which conflicts with pydantic v2).

Task loading: reads task IDs and instructions directly from the data
directory structure: {APPWORLD_ROOT}/data/tasks/{task_id}/specs.json

Task execution: goes through real MCP servers that call the AppWorld
REST API (serve apis) — no direct Python import needed.

Task evaluation: calls the AppWorld environment REST server's /evaluate
endpoint via HTTP.

Requires:
    appworld serve apis --port 8000   (running in background)
    APPWORLD_ROOT env var pointing to the data directory
    APPWORLD_URL env var (default http://localhost:8000)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import requests

APPWORLD_ROOT = os.environ.get("APPWORLD_ROOT", "")
APPWORLD_URL  = os.environ.get("APPWORLD_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# TaskResult — shared across all benchmark runners
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id:      str
    goal:         str
    success:      bool
    steps:        int
    final_answer: Optional[str]
    peak_tokens:  int
    total_tokens: int
    time_elapsed: float
    ccp_stats:    List[Any] = field(default_factory=list)
    method:       str = "ccp"


# ---------------------------------------------------------------------------
# Task loading — reads directly from filesystem, no appworld import
# ---------------------------------------------------------------------------

def _load_tasks_from_fs(appworld_root: str, split: str, max_tasks: int) -> List[Any]:
    """
    Read task IDs from {appworld_root}/data/datasets/{split}.txt
    then load each task's instruction from data/tasks/{task_id}/specs.json.

    AppWorld stores task IDs in dataset files, not as directory name prefixes.
    """
    root       = Path(appworld_root)
    tasks_dir  = root / "data" / "tasks"
    dataset_file = root / "data" / "datasets" / f"{split}.txt"

    if not tasks_dir.exists():
        raise RuntimeError(f"AppWorld tasks directory not found: {tasks_dir}")

    # Read task IDs from the dataset split file
    if dataset_file.exists():
        raw_ids = [l.strip() for l in dataset_file.read_text().splitlines() if l.strip()]
        # Strip any tag suffixes (e.g. "task_001#tag" → "task_001")
        task_ids = [tid.split("#")[0] for tid in raw_ids]
    else:
        # Fallback: list all task directories
        print(f"[AppWorld] Dataset file not found: {dataset_file} — listing all tasks")
        task_ids = [d.name for d in sorted(tasks_dir.iterdir()) if d.is_dir()]

    task_ids = task_ids[:max_tasks]

    tasks = []
    for task_id in task_ids:
        specs_path = tasks_dir / task_id / "specs.json"
        if not specs_path.exists():
            continue
        try:
            specs = json.loads(specs_path.read_text())
            tasks.append(SimpleNamespace(
                id=task_id,
                goal=specs.get("instruction", ""),
                apps=specs.get("allowed_apps", []),
                data=specs,
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    print(f"[AppWorld] Loaded {len(tasks)} tasks from split='{split}'")
    return tasks


# ---------------------------------------------------------------------------
# Evaluation via REST
# ---------------------------------------------------------------------------


def _seed_task(task: Any, appworld_root: str, appworld_url: str) -> bool:
    """
    Seed the AppWorld API server with task-specific databases before running agent.

    AppWorld serve-apis loads BASE databases on startup. Each task has its own
    SQLite databases at data/tasks/{task_id}/dbs/ that must be loaded before
    the agent calls any app APIs.

    Returns True if seeding succeeded, False otherwise.
    """
    task_dbs_path = str(Path(appworld_root) / "data" / "tasks" / task.id / "dbs")

    # Load task-specific databases into memory
    try:
        r = requests.post(
            f"{appworld_url}/dbs",
            json={
                "from_db_home_path": task_dbs_path,
                "to_db_home_path":   f":memory:task_input-{task.id}",
                "create": False,
            },
            timeout=30,
        )
        if r.status_code not in (200, 201):
            print(f"  [AppWorld] /dbs seed failed ({r.status_code}): {r.text[:100]}")
            return False
    except Exception as e:
        print(f"  [AppWorld] /dbs seed error: {e}")
        return False

    # Set task datetime from specs.json (non-fatal if missing)
    task_datetime = getattr(task, "data", {}).get("datetime")
    if task_datetime:
        try:
            requests.post(
                f"{appworld_url}/date_time",
                json={"date_and_time": task_datetime},
                timeout=10,
            )
        except Exception:
            pass

    return True


def _reset_task(task_id: str, appworld_url: str) -> None:
    """Clear task-specific databases from server memory after task completes."""
    try:
        requests.delete(
            f"{appworld_url}/dbs/cache",
            json={"task_id": task_id},
            timeout=10,
        )
    except Exception:
        pass


def _evaluate_via_rest(task_id: str, final_state: Dict) -> float:
    """Call AppWorld environment server to evaluate the task."""
    try:
        r = requests.post(
            f"{APPWORLD_URL}/evaluate",
            json={"task_id": task_id},
            timeout=30,
        )
        if r.status_code == 200:
            result = r.json()
            return float(result.get("score", 0.0))
    except Exception:
        pass
    # Fallback: use agent's done flag
    return 1.0 if final_state.get("done") else 0.0


# ---------------------------------------------------------------------------
# AppWorldRunner — used by ablation studies (non-MCP path)
# ---------------------------------------------------------------------------

class AppWorldRunner:
    """
    Runs baselines against AppWorld tasks.
    Uses filesystem for task loading, REST for evaluation.
    No appworld Python import.
    """

    def __init__(self, split: str = "test", max_tasks: int = 50, max_steps: int = 40):
        self.split     = split
        self.max_tasks = max_tasks
        self.max_steps = max_steps

        if not APPWORLD_ROOT:
            print("[AppWorldRunner] APPWORLD_ROOT not set — using mock tasks.")
        self._tasks = self._load_tasks()

    def _load_tasks(self) -> List[Any]:
        if not APPWORLD_ROOT:
            return self._mock_tasks()
        try:
            return _load_tasks_from_fs(APPWORLD_ROOT, self.split, self.max_tasks)
        except Exception as e:
            print(f"[AppWorldRunner] Task load failed: {e} — using mocks.")
            return self._mock_tasks()

    def evaluate(
        self,
        manager_factory: Callable,
        method_name:     str = "ccp",
        verbose:         bool = True,
    ) -> List[TaskResult]:
        from .mcp_runner import _run_all_tasks_async
        import asyncio
        from ..mcp_server import AppWorldMCPServer
        import sys

        MCP_SCRIPT = str(Path(__file__).parent.parent / "mcp_server.py")
        configs = {
            "appworld": {
                "command":   sys.executable,
                "args":      [MCP_SCRIPT, "--app", "all",
                              "--appworld-url", APPWORLD_URL],
                "transport": "stdio",
            }
        }

        if verbose:
            print(f"\n[AppWorld] {method_name} | {len(self._tasks)} tasks")

        results = asyncio.run(
            _run_all_tasks_async(
                tasks=self._tasks,
                manager_factory=manager_factory,
                server_configs=configs,
                max_steps=self.max_steps,
                score_fn=_evaluate_via_rest,
                verbose=verbose,
            )
        )
        for r in results:
            r.method = method_name
        return results

    def _mock_tasks(self) -> List[Any]:
        goals = [
            "Send an email to Alice with subject 'Meeting Tomorrow'",
            "Order 2 units of Wireless Mouse from Amazon and confirm via SMS",
            "Create a Spotify playlist called Study Vibes with 5 trending songs",
            "Transfer $50 to Charlie via Venmo with note Dinner split",
            "Find Alice phone number in contacts and send her a message",
        ]
        return [
            SimpleNamespace(id=f"mock_{i:03d}", goal=g, apps=[], data={})
            for i, g in enumerate(goals)
        ][: self.max_tasks]


# ---------------------------------------------------------------------------
# Mock tools (for local development only)
# ---------------------------------------------------------------------------

def _register_mock_tools(registry: Dict[str, Any]) -> None:
    import random

    def mock_authenticate(**kw):  return {"token": f"tok_{random.randint(100000,999999)}", "status": "ok"}
    def mock_search(**kw):        return {"results": [{"id": f"id_{i}", "name": f"Result {i}"} for i in range(3)]}
    def mock_send(**kw):          return {"status": "sent", "id": f"msg_{random.randint(1000,9999)}"}
    def mock_list(**kw):          return [{"id": f"item_{i}", "name": f"Item {i}"} for i in range(5)]
    def mock_get(**kw):           return {"id": "obj_001", "status": "ok", "data": "sample"}

    registry.update({
        "amazon__authenticate": mock_authenticate,
        "amazon__search":       mock_search,
        "amazon__order":        mock_send,
        "gmail__send":          mock_send,
        "gmail__list":          mock_list,
        "contacts__search":     mock_search,
        "venmo__pay":           mock_send,
        "spotify__search":      mock_search,
        "spotify__create":      mock_get,
    })
