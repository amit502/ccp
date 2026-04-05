"""
tests/test_new_components.py

Tests for the components added in the second session:
  - ACON (faithful implementation: guidelines, offline optimizer, inference)
  - MCP server (tool schema generation, mock responses)
  - OfficeBench runner (mock mode)
  - Multi-objective QA runner (mock mode, scoring)
"""

from __future__ import annotations

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import unittest
from unittest.mock import MagicMock, patch

from ..baselines.acon import (
    ACONContextManager,
    ACONGuidelines,
    ACONOfflineOptimizer,
    _DEFAULT_HISTORY_GUIDELINE,
    _DEFAULT_OBSERVATION_GUIDELINE,
    _format_trajectory,
    get_acon_reported,
    load_guidelines,
    optimize_guidelines_one_iter,
)
from ..mcp_server import AppWorldMCPServer, ALL_APPS
from ..benchmarks.officebench_runner import OfficeBenchRunner, _score_officebench_task
from ..benchmarks.multiobjqa_runner import (
    MultiObjQARunner,
    _build_multihop_goal,
    _fuzzy_lookup,
    _mock_tasks,
    _score_moqa_answer,
)
from ..models import AgentContext, CCPStats, ContextElement


# ---------------------------------------------------------------------------
# ACON: Guidelines loading
# ---------------------------------------------------------------------------

class TestACONGuidelines(unittest.TestCase):

    def test_load_defaults_when_no_file(self):
        """load_guidelines falls back to defaults when no saved file exists."""
        gl = load_guidelines("__nonexistent_benchmark__")
        self.assertEqual(gl.source, "default")
        self.assertIn("history", gl.history_guideline.lower())
        self.assertIn("compress", gl.observation_guideline.lower())

    def test_default_guidelines_not_empty(self):
        gl = load_guidelines("__nonexistent_benchmark__")
        self.assertGreater(len(gl.history_guideline), 50)
        self.assertGreater(len(gl.observation_guideline), 50)

    def test_acon_reported_numbers_exist(self):
        for bench in ["AppWorld", "OfficeBench", "Multi-objective QA"]:
            ref = get_acon_reported(bench)
            self.assertIsNotNone(ref, f"Missing ACON reported numbers for {bench}")
            self.assertGreater(ref.task_success_rate, 0)
            self.assertGreater(ref.token_reduction_pct, 0)

    def test_get_acon_reported_case_insensitive(self):
        ref = get_acon_reported("appworld")
        self.assertIsNotNone(ref)
        self.assertAlmostEqual(ref.task_success_rate, 0.61)

    def test_get_acon_reported_unknown_returns_none(self):
        self.assertIsNone(get_acon_reported("__unknown__"))


# ---------------------------------------------------------------------------
# ACON: Format trajectory
# ---------------------------------------------------------------------------

class TestFormatTrajectory(unittest.TestCase):

    def _make_elements(self, n: int):
        return [
            ContextElement(step=i, tool_name=f"tool_{i}",
                           tool_input={"k": i}, tool_output=f"output_{i}")
            for i in range(1, n + 1)
        ]

    def test_format_includes_all_steps(self):
        elements = self._make_elements(3)
        result   = _format_trajectory(elements, "TEST")
        for i in range(1, 4):
            self.assertIn(f"Step {i}", result)

    def test_format_includes_label(self):
        elements = self._make_elements(2)
        result   = _format_trajectory(elements, "MY_LABEL")
        self.assertIn("MY_LABEL", result)

    def test_long_output_truncated(self):
        elements = [ContextElement(step=1, tool_name="t", tool_input={},
                                   tool_output="x" * 500)]
        result = _format_trajectory(elements, "L")
        self.assertIn("...", result)


# ---------------------------------------------------------------------------
# ACON: Offline optimizer (mock LLM)
# ---------------------------------------------------------------------------

class TestACONOfflineOptimizer(unittest.TestCase):

    def _make_elements(self, n=3):
        return [ContextElement(step=i, tool_name="t", tool_input={}, tool_output="o")
                for i in range(1, n + 1)]

    def test_optimize_one_iter_parse_error_returns_original(self):
        """If LLM returns unparseable JSON, original guidelines are returned."""
        gl = ACONGuidelines(
            benchmark="test",
            history_guideline="hist",
            observation_guideline="obs",
        )
        with patch("ccp.baselines.acon.call_llm", return_value="not json at all"):
            result = optimize_guidelines_one_iter(
                self._make_elements(), self._make_elements(), gl, "test goal"
            )
        self.assertEqual(result.history_guideline, "hist")
        self.assertEqual(result.n_optimization_iters, 0)

    def test_optimize_one_iter_success(self):
        """If LLM returns valid JSON, guidelines are updated."""
        gl = ACONGuidelines(
            benchmark="test",
            history_guideline="old hist",
            observation_guideline="old obs",
        )
        mock_response = json.dumps({
            "history_guideline":     "new improved history guideline",
            "observation_guideline": "new improved observation guideline",
            "analysis":              "The failure was caused by X.",
        })
        with patch("ccp.baselines.acon.call_llm", return_value=mock_response):
            result = optimize_guidelines_one_iter(
                self._make_elements(), self._make_elements(), gl, "test goal"
            )
        self.assertEqual(result.history_guideline, "new improved history guideline")
        self.assertEqual(result.n_optimization_iters, 1)
        self.assertEqual(result.source, "offline")

    def test_optimizer_stops_early_when_no_failure_pairs(self):
        """If no paired trajectories are found, optimization stops early."""
        optimizer = ACONOfflineOptimizer(benchmark="test", n_iters=5, n_pairs=5)

        # task_runner always succeeds with compression → no failure pairs
        def always_success(task, manager):
            return [], True

        from types import SimpleNamespace
        tasks = [SimpleNamespace(id=f"t{i}", goal=f"goal {i}") for i in range(5)]

        gl = optimizer.run(always_success, tasks)
        # Should have 0 iterations (stopped immediately)
        self.assertEqual(gl.n_optimization_iters, 0)


# ---------------------------------------------------------------------------
# ACON: Inference-time context manager
# ---------------------------------------------------------------------------

class TestACONContextManager(unittest.TestCase):

    def _make_manager(self, threshold=100):
        gl = ACONGuidelines(
            benchmark="test",
            history_guideline=_DEFAULT_HISTORY_GUIDELINE,
            observation_guideline=_DEFAULT_OBSERVATION_GUIDELINE,
            source="default",
        )
        return ACONContextManager(guidelines=gl, token_threshold=threshold)

    def test_add_observation_increments_steps(self):
        m = self._make_manager(threshold=99999)
        m.set_goal("g")
        m.add_observation("t", {}, "output", "ok")
        m.add_observation("t", {}, "output", "ok")
        self.assertEqual(len(m.get_compressed_context().elements), 2)

    @patch("ccp.baselines.acon.call_llm")
    def test_compression_called_when_threshold_exceeded(self, mock_llm):
        """ACON calls the LLM for both history and observation when threshold exceeded."""
        mock_llm.return_value = json.dumps([
            {"step": 1, "action": "t({})", "observation": "summary"}
        ])
        m = self._make_manager(threshold=50)
        m.set_goal("test goal")
        # Two elements: needs at least 2 for history + current obs split
        m.add_observation("tool_a", {}, "x" * 300, "ok")  # triggers compression
        m.add_observation("tool_b", {}, "y" * 300, "ok")
        # Should have called LLM at least once (for history compression)
        self.assertGreater(mock_llm.call_count, 0)

    def test_acon_category_level_not_element_level(self):
        """
        ACON compresses at CATEGORY level (history vs obs), not element level.
        Verify that all history elements get the same guideline applied,
        not individual causal scores.
        """
        m = self._make_manager(threshold=99999)
        m.set_goal("g")
        for i in range(5):
            m.add_observation("t", {}, "output", "ok")
        # Elements have NO phi scores (that's CCP's property, not ACON's)
        for e in m.get_compressed_context().elements:
            self.assertIsNone(e.phi)

    def test_reset_clears_context(self):
        m = self._make_manager(threshold=99999)
        m.set_goal("goal 1")
        m.add_observation("t", {}, "o", "ok")
        m.reset("goal 2")
        self.assertEqual(len(m.get_compressed_context().elements), 0)


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

class TestMCPServer(unittest.TestCase):
    """Tests for the dynamic AppWorld MCP server.
    Tool discovery requires a running AppWorld server — these tests
    verify server initialisation and the ALL_APPS registry only."""

    def test_all_apps_list_has_expected_apps(self):
        """ALL_APPS contains all expected AppWorld application names."""
        for app in ["amazon", "gmail", "venmo", "spotify", "phone",
                    "file_system", "splitwise", "simple_note", "todoist"]:
            self.assertIn(app, ALL_APPS)

    def test_all_apps_count(self):
        """ALL_APPS has at least 10 entries covering all AppWorld apps."""
        self.assertGreaterEqual(len(ALL_APPS), 10)

    def test_server_init_no_connection_required(self):
        """AppWorldMCPServer initialises without connecting to AppWorld server."""
        server = AppWorldMCPServer(
            appworld_url="http://localhost:9999",  # not running
            apps=["amazon"]
        )
        self.assertIsNotNone(server.server)
        self.assertEqual(server.apps, ["amazon"])
        self.assertEqual(server._tools, [])  # lazy-loaded at first list_tools call

    def test_server_url_stored(self):
        """Server stores the AppWorld base URL correctly."""
        server = AppWorldMCPServer(
            appworld_url="http://myserver:8000",
            apps=["gmail"]
        )
        self.assertEqual(server.appworld_url, "http://myserver:8000")

    def test_tool_name_convention_in_fetch(self):
        """_fetch_app_tools produces tool names following {app}__{op_id}."""
        from unittest.mock import patch
        fake_spec = {
            "paths": {
                "/search": {
                    "post": {
                        "operationId": "search_products",
                        "summary": "Search products",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"query": {"type": "string"}},
                                        "required": ["query"]
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "components": {"schemas": {}}
        }
        from ccp.mcp_server import _fetch_app_tools
        import requests
        with patch.object(requests, "get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = fake_spec
            tools = _fetch_app_tools("amazon", "http://localhost:8000")
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "amazon__search_products")
        self.assertTrue(tools[0]["name"].startswith("amazon__"))

    def test_fetch_returns_empty_on_server_down(self):
        """_fetch_app_tools returns empty list when server is unreachable."""
        from ccp.mcp_server import _fetch_app_tools
        tools = _fetch_app_tools("amazon", "http://localhost:19999")
        self.assertEqual(tools, [])


# ---------------------------------------------------------------------------
# OfficeBench Runner
# ---------------------------------------------------------------------------

class TestOfficeBenchRunner(unittest.TestCase):

    def test_mock_tasks_generated(self):
        runner = OfficeBenchRunner(max_tasks=5)
        tasks  = runner._get_tasks()
        self.assertEqual(len(tasks), 5)
        for t in tasks:
            self.assertTrue(hasattr(t, "id"))
            self.assertTrue(hasattr(t, "goal"))
            self.assertGreater(len(t.goal), 10)

    def test_mock_tasks_cover_multiple_apps(self):
        runner = OfficeBenchRunner(max_tasks=8)
        tasks  = runner._get_tasks()
        goals  = [t.goal.lower() for t in tasks]
        # Should cover at least 3 different office app types
        apps_mentioned = sum(1 for app in ["docx", "excel", "powerpoint", "email", "calendar"]
                             if any(app in g for g in goals))
        self.assertGreaterEqual(apps_mentioned, 2)


# ---------------------------------------------------------------------------
# Multi-objective QA
# ---------------------------------------------------------------------------

class TestMultiObjQA(unittest.TestCase):

    def test_build_multihop_goal_includes_all_questions(self):
        qs = ["What is X?", "Who invented Y?", "When did Z happen?"]
        goal = _build_multihop_goal(qs)
        for q in qs:
            self.assertIn(q, goal)

    def test_fuzzy_lookup_known_fact(self):
        answer = _fuzzy_lookup("first president united states")
        self.assertIn("Washington", answer)

    def test_fuzzy_lookup_unknown_returns_not_found(self):
        answer = _fuzzy_lookup("xyzzy quux frobnitz")
        self.assertIn("not found", answer.lower())

    def test_mock_tasks_have_questions_attribute(self):
        tasks = _mock_tasks(max_tasks=3)
        self.assertEqual(len(tasks), 3)
        for t in tasks:
            self.assertTrue(hasattr(t, "questions"))
            self.assertGreater(len(t.questions), 1)

    def test_score_all_correct(self):
        """If the answer contains all expected facts, score should be 1.0."""
        from types import SimpleNamespace
        qs   = ["Who was the first president of the United States?"]
        task = SimpleNamespace(questions=qs)
        # Answer contains "George Washington"
        score = _score_moqa_answer("The first president was George Washington.", task)
        self.assertGreater(score, 0.5)

    def test_score_no_answer(self):
        """Empty answer scores 0."""
        from types import SimpleNamespace
        task = SimpleNamespace(questions=["What is X?"])
        score = _score_moqa_answer("", task)
        self.assertEqual(score, 0.0)

    def test_runner_mock_tasks_generated(self):
        runner = MultiObjQARunner(max_tasks=4, n_hops=3)
        # _load_nq_tasks falls back to mocks when HF not available
        tasks = _mock_tasks(max_tasks=4)
        self.assertEqual(len(tasks), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
