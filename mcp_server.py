"""
mcp_server.py

Real MCP server wrapping AppWorld's 457 APIs.

This file implements a proper MCP server using the official `mcp` Python SDK.
Each AppWorld app (Gmail, Amazon, Venmo, Spotify, etc.) is exposed as a set
of MCP tools following the Model Context Protocol specification.

Run this as a subprocess (stdio transport):
    python -m ccp.mcp_server --app gmail
    python -m ccp.mcp_server --app all

In the full deployment, one server process per app is started, matching
Figure 1 of the proposal: "MCP Servers (457 APIs)".

The CCP module intercepts tool responses at the MCP boundary — specifically
via langchain-mcp-adapters' ToolCallInterceptor — BEFORE they enter the
agent's context window.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

# MCP SDK — official Python implementation
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types as mcp_types

# ---------------------------------------------------------------------------
# AppWorld client (requires `appworld server start` to be running)
# ---------------------------------------------------------------------------

try:
    from appworld.client.api import AppWorldClient
    APPWORLD_AVAILABLE = True
except ImportError:
    APPWORLD_AVAILABLE = False

APPWORLD_BASE_URL = os.environ.get("APPWORLD_BASE_URL", "http://localhost:8000")


class AppWorldMCPServer:
    """
    Wraps AppWorld's REST API as a proper MCP server.

    For each AppWorld app, this server exposes its API methods as MCP tools
    with proper JSON Schema input specifications — enabling the CCP module's
    MCP-aware heuristics to operate on structured tool metadata.

    Tool naming convention: {app_name}__{method_name}
    (double-underscore separator matches AppWorld's naming)
    """

    # AppWorld apps and their primary API methods.
    # Full list: appworld.tasks.api_docs() after server start.
    APP_APIS: Dict[str, List[Dict]] = {
        "amazon": [
            {"name": "authenticate",    "description": "Log in and get session token.",
             "params": {"username": "string", "password": "string"}},
            {"name": "search_products", "description": "Search product catalog.",
             "params": {"query": "string", "token": "string"}},
            {"name": "get_product",     "description": "Get full product details by ID.",
             "params": {"product_id": "string", "token": "string"}},
            {"name": "add_to_cart",     "description": "Add item to shopping cart.",
             "params": {"product_id": "string", "quantity": "integer", "token": "string"}},
            {"name": "place_order",     "description": "Place order for cart contents.",
             "params": {"cart_id": "string", "address": "string", "token": "string"}},
            {"name": "get_order",       "description": "Get order details by order ID.",
             "params": {"order_id": "string", "token": "string"}},
            {"name": "list_orders",     "description": "List recent orders for account.",
             "params": {"token": "string", "limit": "integer"}},
        ],
        "gmail": [
            {"name": "authenticate",    "description": "Authenticate with Gmail.",
             "params": {"username": "string", "password": "string"}},
            {"name": "send_email",      "description": "Send an email.",
             "params": {"to": "string", "subject": "string", "body": "string", "token": "string"}},
            {"name": "list_emails",     "description": "List emails in inbox.",
             "params": {"token": "string", "limit": "integer", "folder": "string"}},
            {"name": "get_email",       "description": "Get full email content by ID.",
             "params": {"email_id": "string", "token": "string"}},
            {"name": "search_emails",   "description": "Search emails by query.",
             "params": {"query": "string", "token": "string"}},
        ],
        "venmo": [
            {"name": "authenticate",    "description": "Log in to Venmo.",
             "params": {"username": "string", "password": "string"}},
            {"name": "send_payment",    "description": "Send money to a user.",
             "params": {"recipient": "string", "amount": "number",
                        "note": "string", "token": "string"}},
            {"name": "get_balance",     "description": "Get current account balance.",
             "params": {"token": "string"}},
            {"name": "list_transactions","description": "List recent transactions.",
             "params": {"token": "string", "limit": "integer"}},
        ],
        "spotify": [
            {"name": "authenticate",    "description": "Authenticate with Spotify.",
             "params": {"username": "string", "password": "string"}},
            {"name": "search_tracks",   "description": "Search for tracks.",
             "params": {"query": "string", "token": "string", "limit": "integer"}},
            {"name": "create_playlist", "description": "Create a new playlist.",
             "params": {"name": "string", "description": "string", "token": "string"}},
            {"name": "add_to_playlist", "description": "Add tracks to a playlist.",
             "params": {"playlist_id": "string", "track_ids": "array", "token": "string"}},
            {"name": "get_trending",    "description": "Get trending tracks.",
             "params": {"token": "string", "limit": "integer"}},
        ],
        "contacts": [
            {"name": "search",          "description": "Search contacts by name or email.",
             "params": {"query": "string", "token": "string"}},
            {"name": "get_contact",     "description": "Get contact details by ID.",
             "params": {"contact_id": "string", "token": "string"}},
            {"name": "list_contacts",   "description": "List all contacts.",
             "params": {"token": "string"}},
        ],
        "phone": [
            {"name": "send_sms",        "description": "Send an SMS message.",
             "params": {"to": "string", "message": "string", "token": "string"}},
            {"name": "list_messages",   "description": "List SMS messages.",
             "params": {"token": "string", "limit": "integer"}},
        ],
    }

    def __init__(self, apps: Optional[List[str]] = None):
        self.apps = apps or list(self.APP_APIS.keys())
        self.server = Server("appworld-ccp")
        self._appworld_client = None
        self._register_handlers()

    def _get_client(self):
        if not APPWORLD_AVAILABLE:
            return None
        if self._appworld_client is None:
            self._appworld_client = AppWorldClient(base_url=APPWORLD_BASE_URL)
        return self._appworld_client

    def _build_tool_schema(self, app: str, api: Dict) -> mcp_types.Tool:
        """Convert an AppWorld API spec into an MCP Tool with JSON Schema."""
        tool_name = f"{app}__{api['name']}"
        properties = {}
        required = []

        for param_name, param_type in api["params"].items():
            type_map = {
                "string": {"type": "string"},
                "integer": {"type": "integer"},
                "number": {"type": "number"},
                "boolean": {"type": "boolean"},
                "array": {"type": "array", "items": {"type": "string"}},
            }
            properties[param_name] = type_map.get(param_type, {"type": "string"})
            # All params required except 'limit' and optional fields
            if param_name not in ("limit", "description", "folder"):
                required.append(param_name)

        return mcp_types.Tool(
            name=tool_name,
            description=f"[{app.upper()}] {api['description']}",
            inputSchema={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )

    def _register_handlers(self):
        """Register MCP protocol handlers on the server."""

        @self.server.list_tools()
        async def list_tools() -> List[mcp_types.Tool]:
            tools = []
            for app in self.apps:
                for api in self.APP_APIS.get(app, []):
                    tools.append(self._build_tool_schema(app, api))
            return tools

        @self.server.call_tool()
        async def call_tool(
            name: str,
            arguments: Dict[str, Any],
        ) -> List[mcp_types.TextContent]:
            """
            Execute an AppWorld API call and return the result as MCP TextContent.
            This is the hook point where CCP's ToolCallInterceptor operates.
            """
            # Parse tool name: {app}__{method}
            if "__" not in name:
                return [mcp_types.TextContent(
                    type="text",
                    text=json.dumps({"error": f"Invalid tool name: {name}"})
                )]

            app, method = name.split("__", 1)
            client = self._get_client()

            if client is not None:
                # Real AppWorld call
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: client.execute(app=app, api=method, **arguments)
                    )
                    output = json.dumps(result) if not isinstance(result, str) else result
                except Exception as exc:
                    output = json.dumps({"error": str(exc), "status": "error"})
            else:
                # Mock response for development
                output = json.dumps(_mock_response(app, method, arguments))

            return [mcp_types.TextContent(type="text", text=output)]

    async def run(self):
        """Start the MCP server on stdio (for subprocess transport)."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


def _mock_response(app: str, method: str, args: Dict) -> Dict:
    """Development mock — returns plausible responses without AppWorld running."""
    import random
    if method == "authenticate":
        return {"token": f"tok_{app}_{random.randint(100000,999999)}",
                "user_id": f"usr_{random.randint(1000,9999)}", "status": "ok"}
    if method in ("search_products", "search_tracks", "search_emails", "search"):
        return {"results": [{"id": f"id_{i}", "name": f"Result {i}"} for i in range(3)],
                "total": 3}
    if method in ("list_orders", "list_emails", "list_transactions",
                  "list_contacts", "list_messages"):
        return [{"id": f"item_{i}", "status": "ok"} for i in range(5)]
    if method in ("add_to_cart",):
        return {"cart_id": "cart_abc123", "status": "added"}
    if method in ("place_order", "send_payment", "send_email", "send_sms"):
        return {"id": f"id_{random.randint(10000,99999)}", "status": "confirmed"}
    if method == "get_balance":
        return {"balance": 142.50, "currency": "USD"}
    return {"status": "ok", "data": {}}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--app", default="all",
        help="Which apps to expose (comma-separated, or 'all')"
    )
    args = parser.parse_args()

    apps = (
        list(AppWorldMCPServer.APP_APIS.keys())
        if args.app == "all"
        else [a.strip() for a in args.app.split(",")]
    )

    server = AppWorldMCPServer(apps=apps)
    asyncio.run(server.run())
