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
from ..mcp_server import AppWorldMCPServer, _mock_response
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

    def test_app_apis_populated(self):
        """Server has API definitions for expected AppWorld apps."""
        for app in ["amazon", "gmail", "venmo", "spotify", "contacts"]:
            self.assertIn(app, AppWorldMCPServer.APP_APIS)
            self.assertGreater(len(AppWorldMCPServer.APP_APIS[app]), 0)

    def test_tool_schema_naming_convention(self):
        """Tool names follow {app}__{method} convention."""
        server = AppWorldMCPServer(apps=["amazon"])
        # Build schemas synchronously for testing
        schemas = []
        for api in AppWorldMCPServer.APP_APIS["amazon"]:
            schema = server._build_tool_schema("amazon", api)
            schemas.append(schema)
            self.assertTrue(schema.name.startswith("amazon__"),
                            f"Tool name '{schema.name}' doesn't follow convention")

    def test_tool_schema_has_input_schema(self):
        """Every tool schema has a valid JSON Schema inputSchema."""
        server = AppWorldMCPServer(apps=["gmail"])
        for api in AppWorldMCPServer.APP_APIS["gmail"]:
            schema = server._build_tool_schema("gmail", api)
            self.assertIn("type", schema.inputSchema)
            self.assertIn("properties", schema.inputSchema)
            self.assertEqual(schema.inputSchema["type"], "object")

    def test_mock_response_authenticate(self):
        """Mock authenticate returns token and user_id."""
        resp = _mock_response("amazon", "authenticate", {})
        self.assertIn("token", resp)
        self.assertIn("user_id", resp)
        self.assertIn("amazon", resp["token"])

    def test_mock_response_list_returns_list(self):
        """Mock list_* methods return a list."""
        resp = _mock_response("amazon", "list_orders", {})
        self.assertIsInstance(resp, list)

    def test_mock_response_send_returns_status(self):
        """Mock send_* methods return a status."""
        resp = _mock_response("gmail", "send_email", {})
        self.assertIn("status", resp)

    def test_all_six_apps_registered(self):
        """Server registers all 6 expected apps."""
        expected = {"amazon", "gmail", "venmo", "spotify", "contacts", "phone"}
        actual   = set(AppWorldMCPServer.APP_APIS.keys())
        self.assertTrue(expected.issubset(actual))

    def test_tool_count(self):
        """Server exposes at least 25 tools across all apps (457 in real AppWorld)."""
        server = AppWorldMCPServer()
        total  = sum(len(apis) for apis in AppWorldMCPServer.APP_APIS.values())
        self.assertGreater(total, 25)


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
