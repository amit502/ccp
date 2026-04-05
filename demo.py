"""
demo.py

End-to-end CCP demonstration — runs WITHOUT AppWorld or Docker.

Uses a self-contained mock environment that simulates a realistic
multi-step agentic task (online shopping + email notification workflow)
to demonstrate CCP's three-tier compression in action.

Run with:
    NAUTILUS_API_KEY=<key> NAUTILUS_BASE_URL=<url> python demo.py

Or in mock-LLM mode (no API key needed — uses scripted agent responses):
    python demo.py --mock-llm
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from .agent import _TOOL_REGISTRY, register_tool
from .baselines.acon import ACONContextManager
from .baselines.compression import FIFOManager, NoCompression, RetrievalBasedManager
from .benchmarks.metrics import compute_all_metrics, print_metrics_table
from .benchmarks.appworld_runner import TaskResult
from .context_manager import CCPContextManager
from .models import AgentContext, ContextElement

# ---------------------------------------------------------------------------
# Mock tool implementations
# These simulate AppWorld's APIs (Amazon, Gmail, Venmo, Contacts, etc.)
# ---------------------------------------------------------------------------

def _mock_authenticate(service: str = "amazon", username: str = "", password: str = "") -> dict:
    token = f"tok_{service}_{random.randint(100000, 999999)}"
    return {"status": "ok", "token": token, "user_id": f"usr_{random.randint(1000, 9999)}"}

def _mock_search_product(query: str = "", token: str = "") -> dict:
    products = [
        {"id": "PROD_001", "name": "Wireless Mouse", "price": 29.99, "rating": 4.5, "in_stock": True},
        {"id": "PROD_002", "name": "Wireless Mouse Pro", "price": 49.99, "rating": 4.8, "in_stock": True},
        {"id": "PROD_003", "name": "Basic Mouse", "price": 12.99, "rating": 3.9, "in_stock": False},
    ]
    return {"results": [p for p in products if query.lower() in p["name"].lower()], "total": 3}

def _mock_get_product_details(product_id: str = "", token: str = "") -> dict:
    return {
        "id": product_id,
        "name": "Wireless Mouse",
        "price": 29.99,
        "description": "Ergonomic wireless mouse with 2.4GHz connectivity. "
                       "Battery life: 12 months. DPI: 800-1600. Compatible with Windows/Mac.",
        "seller": "TechDeals Inc.",
        "shipping_days": 2,
        "return_policy": "30-day returns accepted.",
    }

def _mock_get_cart(token: str = "") -> dict:
    return {"items": [], "subtotal": 0.0, "estimated_tax": 0.0}

def _mock_add_to_cart(product_id: str = "", quantity: int = 1, token: str = "") -> dict:
    return {"status": "added", "cart_id": "cart_abc123", "item_count": quantity}

def _mock_place_order(cart_id: str = "", token: str = "", address: str = "") -> dict:
    order_id = f"ORD-{random.randint(10000, 99999)}"
    return {
        "status": "confirmed",
        "order_id": order_id,
        "estimated_delivery": "2026-04-07",
        "total_charged": 32.19,
    }

def _mock_get_contacts(name: str = "", token: str = "") -> dict:
    contacts = {
        "alice": {"name": "Alice Smith", "email": "alice@example.com", "phone": "+1-555-0101"},
        "bob":   {"name": "Bob Jones",   "email": "bob@example.com",   "phone": "+1-555-0102"},
    }
    key = name.lower()
    match = contacts.get(key)
    return {"results": [match] if match else [], "total": 1 if match else 0}

def _mock_send_email(to: str = "", subject: str = "", body: str = "", token: str = "") -> dict:
    return {"status": "sent", "message_id": f"msg_{random.randint(10000, 99999)}"}

def _mock_ping(service: str = "") -> dict:
    return {"status": "ok", "latency_ms": random.randint(10, 50)}

def _mock_list_recent_orders(token: str = "", limit: int = 10) -> list:
    # Simulate a long, verbose response (likely inert after the first look)
    return [
        {"order_id": f"ORD-{i}", "date": "2026-03-{:02d}".format(i),
         "status": "delivered", "total": round(random.uniform(10, 100), 2)}
        for i in range(1, limit + 1)
    ]

def _setup_mock_tools() -> None:
    """Register all mock tools into the agent's tool registry."""
    _TOOL_REGISTRY.clear()
    register_tool("amazon__authenticate",      _mock_authenticate)
    register_tool("amazon__search_product",    _mock_search_product)
    register_tool("amazon__get_product",       _mock_get_product_details)
    register_tool("amazon__get_cart",          _mock_get_cart)
    register_tool("amazon__add_to_cart",       _mock_add_to_cart)
    register_tool("amazon__place_order",       _mock_place_order)
    register_tool("amazon__list_recent_orders",_mock_list_recent_orders)
    register_tool("contacts__search",          _mock_get_contacts)
    register_tool("gmail__authenticate",       _mock_authenticate)
    register_tool("gmail__send_email",         _mock_send_email)
    register_tool("system__ping",              _mock_ping)


# ---------------------------------------------------------------------------
# Scripted agent responses for mock-LLM mode
# (Lets you demo the full pipeline without an API key)
# ---------------------------------------------------------------------------

_SCRIPTED_STEPS = [
    # Step 1: Authenticate with Amazon
    {"action": "tool_call", "tool": "amazon__authenticate",
     "input": {"service": "amazon", "username": "amit", "password": "pass123"}},
    # Step 2: Irrelevant health-check (should become INERT)
    {"action": "tool_call", "tool": "system__ping", "input": {"service": "amazon"}},
    # Step 3: List recent orders (verbose, likely INERT after review)
    {"action": "tool_call", "tool": "amazon__list_recent_orders",
     "input": {"token": "tok_amazon_123", "limit": 10}},
    # Step 4: Search for product
    {"action": "tool_call", "tool": "amazon__search_product",
     "input": {"query": "Wireless Mouse", "token": "tok_amazon_123"}},
    # Step 5: Get product details (RELEVANT → summarised)
    {"action": "tool_call", "tool": "amazon__get_product",
     "input": {"product_id": "PROD_001", "token": "tok_amazon_123"}},
    # Step 6: Add to cart (ACTIVE — cart_id needed for order)
    {"action": "tool_call", "tool": "amazon__add_to_cart",
     "input": {"product_id": "PROD_001", "quantity": 1, "token": "tok_amazon_123"}},
    # Step 7: Place order (ACTIVE — produces order_id needed for email)
    {"action": "tool_call", "tool": "amazon__place_order",
     "input": {"cart_id": "cart_abc123", "token": "tok_amazon_123",
               "address": "123 Main St, Yankton SD"}},
    # Step 8: Authenticate Gmail
    {"action": "tool_call", "tool": "gmail__authenticate",
     "input": {"service": "gmail", "username": "amit", "password": "pass123"}},
    # Step 9: Look up Alice's email
    {"action": "tool_call", "tool": "contacts__search",
     "input": {"name": "Alice", "token": "tok_gmail_123"}},
    # Step 10: Send confirmation email
    {"action": "tool_call", "tool": "gmail__send_email",
     "input": {"to": "alice@example.com",
               "subject": "Order Confirmed",
               "body": "Hi Alice, I've placed the order (ORD-XXXXX). Arriving Apr 7.",
               "token": "tok_gmail_123"}},
    # Step 11: Finish
    {"action": "finish", "answer": "Order placed and confirmation email sent to Alice."},
]


# ---------------------------------------------------------------------------
# Core demo runner
# ---------------------------------------------------------------------------

def run_demo(
    method:          str   = "ccp",
    mock_llm:        bool  = True,
    token_threshold: int   = 300,   # Low threshold to trigger compression during demo
    verbose:         bool  = True,
) -> TaskResult:
    """
    Run a single demo task end-to-end with the specified context manager.

    Returns a TaskResult for metrics computation.
    """
    GOAL = (
        "Order 1 unit of 'Wireless Mouse' from Amazon using my account, "
        "then send an email to Alice confirming the order with the order ID."
    )

    _setup_mock_tools()

    # Select context manager
    if method == "ccp":
        manager = CCPContextManager(
            tau_high=0.6, tau_low=0.3,
            token_threshold=token_threshold,
            use_heuristics=True,
            compress_relevant=False,    # Disable LLM summarisation for speed in demo
        )
    elif method == "acon":
        manager = ACONContextManager(token_threshold=token_threshold)
    elif method == "fifo":
        manager = FIFOManager(token_threshold=token_threshold)
    elif method == "retrieval":
        manager = RetrievalBasedManager(token_threshold=token_threshold, top_k=4)
    elif method == "no_compression":
        manager = NoCompression(token_threshold=token_threshold)
    else:
        raise ValueError(f"Unknown method: {method}")

    manager.set_goal(GOAL)

    step_idx   = 0
    peak_tokens  = 0
    total_tokens = 0
    done         = False
    final_answer = None

    if verbose:
        print(f"\n{'='*65}")
        print(f"  CCP Demo — Method: {method.upper()}")
        print(f"  Goal: {GOAL[:60]}...")
        print(f"{'='*65}")

    while step_idx < len(_SCRIPTED_STEPS):
        decision = _SCRIPTED_STEPS[step_idx]
        step_idx += 1

        if decision["action"] == "finish":
            done         = True
            final_answer = decision.get("answer", "")
            if verbose:
                print(f"\n  ✓ DONE: {final_answer}")
            break

        tool_name  = decision["tool"]
        tool_input = decision["input"]

        # Execute the tool
        tool_fn = _TOOL_REGISTRY.get(tool_name)
        if tool_fn:
            try:
                raw_output = tool_fn(**tool_input)
                status     = "ok"
            except Exception as exc:
                raw_output = str(exc)
                status     = "error"
        else:
            raw_output = "[not found]"
            status     = "error"

        # CCP intercepts the observation
        element = manager.add_observation(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=json.dumps(raw_output) if not isinstance(raw_output, str) else raw_output,
            status=status,
        )

        ctx_tokens    = manager.get_compressed_context().total_tokens()
        peak_tokens   = max(peak_tokens, ctx_tokens)
        total_tokens += ctx_tokens

        if verbose and element.phi is not None:
            tier_label = {
                "active":   "🔴 ACTIVE",
                "relevant": "🟡 RELEVANT",
                "inert":    "⚫ INERT",
            }.get(element.tier.value if element.tier else "", "  unscored")
            print(f"  Step {step_idx:2d} | {tool_name:<35} | ϕ={element.phi:.2f} | {tier_label}")
        elif verbose:
            print(f"  Step {step_idx:2d} | {tool_name:<35} | [no ϕ score]")

    # Print compression summary
    if verbose:
        stats_log = manager.get_stats_log()
        if stats_log:
            print(f"\n  ── Compression events: {len(stats_log)} ──")
            for s in stats_log:
                print(f"     Step {s.step}: {s.tokens_before}→{s.tokens_after} tokens "
                      f"(-{s.token_reduction_pct:.1f}%) | "
                      f"active={s.active_count} relevant={s.relevant_count} inert={s.inert_count}")
        print(f"  Peak tokens: {peak_tokens}")

    return TaskResult(
        task_id="demo_001",
        goal=GOAL,
        success=done,
        steps=step_idx,
        final_answer=final_answer,
        peak_tokens=peak_tokens,
        total_tokens=total_tokens,
        time_elapsed=0.0,
        ccp_stats=manager.get_stats_log(),
        method=method,
    )


# ---------------------------------------------------------------------------
# Full comparison demo: run all methods, print metrics table
# ---------------------------------------------------------------------------

def run_comparison_demo(verbose: bool = True) -> None:
    """Run all 5 methods on the same task and compare metrics."""
    methods = ["no_compression", "fifo", "retrieval", "ccp"]
    all_results = []

    for method in methods:
        print(f"\nRunning: {method}")
        result = run_demo(method=method, mock_llm=True, verbose=verbose)
        all_results.append(result)

    # Compute and display metrics
    from .benchmarks.metrics import compute_all_metrics, print_metrics_table
    all_metrics = [compute_all_metrics([r], method=r.method) for r in all_results]

    print(f"\n{'='*65}")
    print("  RESULTS SUMMARY")
    print(f"{'='*65}")
    print_metrics_table(all_metrics)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCP End-to-End Demo")
    parser.add_argument(
        "--method",
        default="all",
        choices=["all", "ccp", "fifo", "retrieval", "no_compression", "acon"],
        help="Which method to demo (default: all)",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        default=True,
        help="Use scripted agent responses (no API key needed)",
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if args.method == "all":
        run_comparison_demo(verbose=args.verbose)
    else:
        run_demo(method=args.method, mock_llm=args.mock_llm, verbose=args.verbose)
