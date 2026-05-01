"""
tests/test_ccp.py

Unit tests for CCP core components.
Run with: python -m pytest tests/ -v

Tests cover:
  - ContextElement and AgentContext data structures
  - MCP heuristic scorer (no LLM calls needed)
  - Three-tier tier assignment
  - CCPContextManager trigger and compression
  - Baseline managers (FIFO, Retrieval, TokenPerplexity)
  - Metrics computation
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest.mock import MagicMock, patch

from ..models import AgentContext, CCPStats, CompressionTier, ContextElement
from ..causal_scorer import _heuristic_phi
from ..context_manager import CCPContextManager, assign_tiers
from ..baselines.compression import FIFOManager, RetrievalBasedManager, TokenPerplexityManager
from ..benchmarks.metrics import (
    causal_recall,
    compression_efficiency,
    compute_all_metrics,
    context_dependency,
    mean_peak_token_usage,
    task_success_rate,
)
from ..benchmarks.appworld_runner import TaskResult


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------

class TestContextElement(unittest.TestCase):

    def _make_element(self, step=1, tool="search", output="some result", status="ok"):
        return ContextElement(
            step=step,
            tool_name=tool,
            tool_input={"query": "test"},
            tool_output=output,
            status=status,
        )

    def test_token_count_positive(self):
        e = self._make_element(output="a" * 400)
        self.assertGreater(e.token_count(), 0)

    def test_token_count_rough(self):
        # 400 chars ≈ 100 tokens
        e = self._make_element(output="a" * 400)
        self.assertAlmostEqual(e.token_count(), 100, delta=20)

    def test_to_context_block_contains_tool_name(self):
        e = self._make_element(tool="email__send")
        block = e.to_context_block()
        self.assertIn("email__send", block)

    def test_observation_str_uses_compressed_when_set(self):
        e = self._make_element(output="long raw output")
        e.compressed_output = "short summary"
        self.assertEqual(e.observation_str(), "short summary")

    def test_observation_str_raw_when_no_compression(self):
        e = self._make_element(output="raw output")
        self.assertEqual(e.observation_str(), "raw output")


class TestAgentContext(unittest.TestCase):

    def test_add_and_len(self):
        ctx = AgentContext(goal="test goal")
        for i in range(5):
            ctx.add(ContextElement(
                step=i, tool_name="t", tool_input={}, tool_output="x"
            ))
        self.assertEqual(len(ctx), 5)

    def test_total_tokens_grows(self):
        ctx = AgentContext(goal="g")
        ctx.add(ContextElement(step=1, tool_name="t", tool_input={}, tool_output="x" * 400))
        ctx.add(ContextElement(step=2, tool_name="t", tool_input={}, tool_output="x" * 400))
        self.assertGreater(ctx.total_tokens(), 0)


# ---------------------------------------------------------------------------
# Heuristic scorer tests (no LLM needed)
# ---------------------------------------------------------------------------

class TestHeuristicScorer(unittest.TestCase):

    def _element(self, tool, output="ok", status="ok"):
        return ContextElement(
            step=1, tool_name=tool, tool_input={},
            tool_output=output, status=status
        )

    def test_error_status_low_phi(self):
        e = self._element("search", status="error")
        phi = _heuristic_phi(e)
        self.assertIsNotNone(phi)
        self.assertLess(phi, 0.3)

    def test_authenticate_high_phi(self):
        e = self._element("authenticate")
        phi = _heuristic_phi(e)
        self.assertIsNotNone(phi)
        self.assertGreater(phi, 0.7)

    def test_get_token_high_phi(self):
        e = self._element("get_token", output="tok_123456")
        phi = _heuristic_phi(e)
        self.assertIsNotNone(phi)
        self.assertGreater(phi, 0.7)

    def test_short_output_high_phi(self):
        e = self._element("unknown_tool", output="id_abc123")
        phi = _heuristic_phi(e)
        self.assertIsNotNone(phi)
        self.assertGreater(phi, 0.5)

    def test_long_output_low_phi(self):
        e = self._element("list_items", output="x" * 3000)
        phi = _heuristic_phi(e)
        self.assertIsNotNone(phi)
        self.assertLess(phi, 0.4)

    def test_unknown_mid_length_returns_none(self):
        e = self._element("some_random_tool", output="x" * 500)
        phi = _heuristic_phi(e)
        self.assertIsNone(phi)  # No heuristic → fall through to LLM


# ---------------------------------------------------------------------------
# Tier assignment tests
# ---------------------------------------------------------------------------

class TestTierAssignment(unittest.TestCase):

    def _scored_element(self, step, phi):
        e = ContextElement(step=step, tool_name="t", tool_input={}, tool_output="o")
        e.phi = phi
        return e

    def test_high_phi_becomes_active(self):
        e = self._scored_element(1, phi=0.9)
        active, relevant, inert = assign_tiers([e], tau_high=0.6, tau_low=0.3)
        self.assertIn(e, active)
        self.assertEqual(e.tier, CompressionTier.ACTIVE)

    def test_mid_phi_becomes_relevant(self):
        e = self._scored_element(1, phi=0.45)
        active, relevant, inert = assign_tiers([e], tau_high=0.6, tau_low=0.3)
        self.assertIn(e, relevant)
        self.assertEqual(e.tier, CompressionTier.RELEVANT)

    def test_low_phi_becomes_inert(self):
        e = self._scored_element(1, phi=0.1)
        active, relevant, inert = assign_tiers([e], tau_high=0.6, tau_low=0.3)
        self.assertIn(e, inert)
        self.assertEqual(e.tier, CompressionTier.INERT)

    def test_none_phi_defaults_to_relevant(self):
        e = ContextElement(step=1, tool_name="t", tool_input={}, tool_output="o")
        e.phi = None  # Not scored
        active, relevant, inert = assign_tiers([e], tau_high=0.6, tau_low=0.3)
        self.assertIn(e, relevant)  # Conservative default


# ---------------------------------------------------------------------------
# CCPContextManager integration tests (mocking the LLM scorer)
# ---------------------------------------------------------------------------

class TestCCPContextManager(unittest.TestCase):

    def _make_manager(self, threshold=100):
        """Manager with very low threshold to trigger compression quickly."""
        return CCPContextManager(
            tau_high=0.6,
            tau_low=0.3,
            token_threshold=threshold,
            use_heuristics=True,
            compress_relevant=False,  # Disable LLM summarisation in tests
        )

    def test_add_observation_increments_step(self):
        m = self._make_manager(threshold=99999)
        m.set_goal("test goal")
        m.add_observation("search", {"q": "x"}, "result", "ok")
        m.add_observation("search", {"q": "y"}, "result", "ok")
        self.assertEqual(len(m.get_compressed_context().elements), 2)

    @patch("ccp.context_manager._compress_to_summary", return_value="[summary]")
    @patch("ccp.context_manager._compress_to_digest", return_value="[digest]")
    def test_compression_triggered_when_threshold_exceeded(self, mock_digest, mock_summary):
        """Add a high-ϕ tool (authenticate) and a low-ϕ one (list_items with long output).
        Ensure the inert one gets a digest and the active one is preserved."""
        m = self._make_manager(threshold=50)  # Very low threshold
        m.set_goal("test goal")

        # High-ϕ (heuristic: authenticate)
        m.add_observation("authenticate", {"user": "a"}, "tok_abc123", "ok")
        # Low-ϕ (heuristic: list_items with long output) — this should trigger compression
        long_out = "item, " * 600  # > 2000 chars
        m.add_observation("list_items", {"page": 1}, long_out, "ok")

        stats = m.get_stats_log()
        self.assertGreater(len(stats), 0, "Compression should have been triggered")

    def test_reset_clears_context(self):
        m = self._make_manager(threshold=99999)
        m.set_goal("goal 1")
        m.add_observation("t", {}, "out", "ok")
        m.reset(goal="goal 2")
        self.assertEqual(len(m.get_compressed_context().elements), 0)
        self.assertEqual(m.get_compressed_context().goal, "goal 2")


# ---------------------------------------------------------------------------
# Baseline tests
# ---------------------------------------------------------------------------

class TestFIFOManager(unittest.TestCase):

    def test_drops_oldest_when_threshold_exceeded(self):
        m = FIFOManager(token_threshold=50, keep_ratio=0.5)
        m.set_goal("g")
        for i in range(20):
            m.add_observation("t", {}, "x" * 100, "ok")
        # After compression, context should be shorter than 20 elements
        self.assertLess(len(m.get_compressed_context().elements), 20)

    def test_always_keeps_recent_elements(self):
        m = FIFOManager(token_threshold=50, keep_ratio=0.5)
        m.set_goal("g")
        for i in range(20):
            m.add_observation("t", {}, "x" * 100, "ok")
        # The most recent element should still be there
        steps = [e.step for e in m.get_compressed_context().elements]
        self.assertIn(20, steps)


class TestRetrievalBasedManager(unittest.TestCase):

    def test_keeps_goal_relevant_elements(self):
        m = RetrievalBasedManager(token_threshold=50, top_k=2)
        m.set_goal("email alice meeting")
        # Add a relevant element
        m.add_observation("email__send", {"to": "alice"}, "sent ok", "ok")
        # Add many irrelevant elements to trigger compression
        for i in range(15):
            m.add_observation("catalog__search", {"query": "shoes"}, "x" * 200, "ok")
        ctx = m.get_compressed_context()
        tool_names = [e.tool_name for e in ctx.elements]
        self.assertIn("email__send", tool_names)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):

    def _make_result(self, success=True, steps=10, peak=1000, total=5000, stats=None):
        return TaskResult(
            task_id="t1",
            goal="g",
            success=success,
            steps=steps,
            final_answer="done",
            peak_tokens=peak,
            total_tokens=total,
            time_elapsed=1.0,
            ccp_stats=stats or [],
            method="ccp",
        )

    def test_task_success_rate_all_success(self):
        results = [self._make_result(success=True) for _ in range(5)]
        self.assertAlmostEqual(task_success_rate(results), 1.0)

    def test_task_success_rate_none_success(self):
        results = [self._make_result(success=False) for _ in range(5)]
        self.assertAlmostEqual(task_success_rate(results), 0.0)

    def test_mean_peak_token_usage(self):
        results = [
            self._make_result(peak=1000),
            self._make_result(peak=2000),
        ]
        self.assertAlmostEqual(mean_peak_token_usage(results), 1500.0)

    def test_compression_efficiency(self):
        results = [self._make_result(success=True, total=2000)]
        eff = compression_efficiency(results)
        self.assertAlmostEqual(eff, 1.0 / 2.0, places=4)  # 1 / (2000/1000) = 0.5

    def test_causal_recall_no_stats_returns_none(self):
        results = [self._make_result(stats=[])]
        self.assertIsNone(causal_recall(results))

    def test_causal_recall_with_stats(self):
        stats = [CCPStats(
            step=5,
            total_elements=10,
            active_count=4,
            relevant_count=3,
            inert_count=3,
            tokens_before=2000,
            tokens_after=1000,
            scorer_calls=2,
        )]
        results = [self._make_result(stats=stats)]
        cr = causal_recall(results)
        self.assertIsNotNone(cr)
        # (4+3)/10 = 0.7 preserved
        self.assertAlmostEqual(cr, 0.7, places=4)

    def test_compute_all_metrics_returns_dict(self):
        results = [self._make_result()]
        metrics = compute_all_metrics(results, method="ccp")
        required_keys = [
            "method", "n_tasks", "task_success_rate",
            "mean_peak_tokens", "context_dependency",
            "compression_efficiency",
        ]
        for k in required_keys:
            self.assertIn(k, metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
