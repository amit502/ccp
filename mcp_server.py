"""
mcp_server.py

Real MCP server wrapping AppWorld's 457 APIs.

Discovers all APIs dynamically from the AppWorld REST server's OpenAPI spec
at startup — no hardcoding, no appworld Python import needed.

Each app is mounted at /{app_name}/ on the AppWorld server.
OpenAPI spec: GET http://localhost:8000/{app_name}/openapi.json

Tool naming: {app_name}__{operation_id}
Tool call:   POST http://localhost:8000/{app_name}/{path} with JSON body

Run as subprocess:
    python -m ccp.mcp_server --appworld-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

import requests

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types as mcp_types

APPWORLD_URL = os.environ.get("APPWORLD_URL", "http://localhost:8000")

# All AppWorld app names (from APP_TO_DESCRIPTION in appworld source)
ALL_APPS = [
    # Real user-facing apps — api_docs excluded (internal, has broken constants)
    # admin excluded (internal task management, not for agent tool calls)
    "supervisor",
    "amazon", "phone", "file_system", "spotify",
    "venmo", "gmail", "splitwise", "simple_note", "todoist",
]


# ---------------------------------------------------------------------------
# Dynamic API discovery from OpenAPI spec
# ---------------------------------------------------------------------------

def _fetch_app_tools(app: str, base_url: str) -> List[Dict]:
    """
    Fetch OpenAPI spec for one app and convert to tool definitions.
    Returns list of {name, description, path, method, properties, required}.
    """
    try:
        r = requests.get(f"{base_url}/{app}/openapi.json", timeout=5)
        if r.status_code != 200:
            return []
        spec = r.json()
    except Exception:
        return []

    tools = []
    paths = spec.get("paths", {})
    schemas = spec.get("components", {}).get("schemas", {})

    for path, path_item in paths.items():
        for http_method, operation in path_item.items():
            if http_method not in ("get", "post", "put", "delete", "patch"):
                continue
            if operation.get("summary", "").startswith("meta:"):
                continue  # skip internal AppWorld meta-endpoints

            op_id = operation.get("operationId", "")
            if not op_id:
                # Derive from path
                op_id = path.strip("/").replace("/", "_")

            tool_name = f"{app}__{op_id}"
            description = operation.get("summary") or operation.get("description") or op_id

            # Extract parameters from requestBody or parameters
            properties: Dict[str, Any] = {}
            required: List[str] = []

            # Query/path params
            for param in operation.get("parameters", []):
                pname = param.get("name", "")
                pschema = param.get("schema", {"type": "string"})
                properties[pname] = pschema
                if param.get("required", False):
                    required.append(pname)

            # Request body
            body = operation.get("requestBody", {})
            if body:
                content = body.get("content", {})
                json_content = content.get("application/json", {})
                body_schema = json_content.get("schema", {})

                # Resolve $ref if needed
                if "$ref" in body_schema:
                    ref_name = body_schema["$ref"].split("/")[-1]
                    body_schema = schemas.get(ref_name, {})

                body_props = body_schema.get("properties", {})
                body_req   = body_schema.get("required", [])

                # If body has a single embed field (FastAPI Body embed=True pattern)
                if len(body_props) == 1:
                    field_name = list(body_props.keys())[0]
                    field_schema = body_props[field_name]
                    if "$ref" in field_schema:
                        ref_name = field_schema["$ref"].split("/")[-1]
                        inner = schemas.get(ref_name, {})
                        properties.update(inner.get("properties", {}))
                        required.extend(inner.get("required", []))
                    else:
                        properties[field_name] = field_schema
                        if field_name in body_req:
                            required.append(field_name)
                else:
                    for pname, pschema in body_props.items():
                        if "$ref" in pschema:
                            ref_name = pschema["$ref"].split("/")[-1]
                            pschema = schemas.get(ref_name, {"type": "object"})
                        properties[pname] = pschema
                    required.extend(body_req)

            tool_entry = {
                "name":        tool_name,
                "description": f"[{app.upper()}] {description}",
                "path":        path,
                "http_method": http_method,
                "properties":  properties,
                "required":    list(set(required)),
            }
            tools.append(tool_entry)
            # Debug: log parameter names for payment_request creation
            if "payment_request" in tool_name and http_method == "post":
                print(f"[MCP] {tool_name} params={list(properties.keys())} "
                      f"required={tool_entry['required']}", file=sys.stderr)

    return tools


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

class AppWorldMCPServer:
    """
    Discovers all AppWorld APIs at startup via OpenAPI spec and exposes them
    as MCP tools. Tool calls are forwarded to the AppWorld REST server via HTTP.
    No appworld Python import required.
    """

    def __init__(self, appworld_url: str, apps: List[str]):
        self.appworld_url = appworld_url.rstrip("/")
        self.apps         = apps
        self.server       = Server("appworld-mcp")
        self._tools: List[Dict] = []
        self._session = requests.Session()
        self._register_handlers()

    def _load_tools(self) -> List[Dict]:
        """Fetch tool definitions from all app OpenAPI specs."""
        all_tools = []
        for app in self.apps:
            app_tools = _fetch_app_tools(app, self.appworld_url)
            all_tools.extend(app_tools)
            if app_tools:
                print(f"[MCP] {app}: {len(app_tools)} tools", file=sys.stderr)
        print(f"[MCP] Total: {len(all_tools)} tools loaded", file=sys.stderr)
        return all_tools

    def _register_handlers(self):

        @self.server.list_tools()
        async def list_tools() -> List[mcp_types.Tool]:
            if not self._tools:
                self._tools = self._load_tools()
            return [
                mcp_types.Tool(
                    name=t["name"],
                    description=t["description"],
                    inputSchema={
                        "type": "object",
                        "properties": t["properties"],
                        "required": t["required"],
                    },
                )
                for t in self._tools
            ]

        @self.server.call_tool()
        async def call_tool(
            name: str,
            arguments: Dict[str, Any],
        ) -> List[mcp_types.TextContent]:
            """Forward tool call to AppWorld REST server via HTTP."""

            if not self._tools:
                self._tools = self._load_tools()

            # Find matching tool definition
            tool_def = next((t for t in self._tools if t["name"] == name), None)

            if tool_def is None:
                return [mcp_types.TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}),
                )]

            app     = name.split("__")[0]
            path    = tool_def["path"]
            http_m  = tool_def["http_method"]
            url     = f"{self.appworld_url}/{app}{path}"

            try:
                fn = getattr(self._session, http_m)
                _args = arguments
                _url  = url

                # Always send access_token as query param (AppWorld API convention)
                _qp   = {}
                _body = dict(_args)
                if "access_token" in _body:
                    _qp["access_token"] = _body.pop("access_token")

                # Also send token in Authorization header (some endpoints check header)
                _headers = {}
                if _qp.get("access_token"):
                    _headers["Authorization"] = f"Bearer {_qp['access_token']}"

                async def _call():
                    if http_m in ("post", "put", "patch"):
                        if "auth/token" in _url:
                            # OAuth endpoints use form data; normalize field names
                            _form = dict(_body)
                            if "email" in _form and "username" not in _form:
                                _form["username"] = _form.pop("email")
                            if "login" in _form and "username" not in _form:
                                _form["username"] = _form.pop("login")
                            if "account_name" in _form and "username" not in _form:
                                _form["username"] = _form.pop("account_name")
                            _f = dict(_form)
                            return await asyncio.get_event_loop().run_in_executor(
                                None, lambda: fn(_url, data=_f, params=_qp, headers=_headers, timeout=30)
                            )
                        _b = dict(_body)
                        return await asyncio.get_event_loop().run_in_executor(
                            None, lambda: fn(_url, json=_b, params=_qp, headers=_headers, timeout=30)
                        )
                    else:
                        _merged = {**_body, **_qp}
                        return await asyncio.get_event_loop().run_in_executor(
                            None, lambda: fn(_url, params=_merged, headers=_headers, timeout=30)
                        )

                resp   = await _call()
                output = resp.text
            except Exception as exc:
                output = json.dumps({"error": str(exc), "status": "error"})

            return [mcp_types.TextContent(type="text", text=output)]

    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-url", default=APPWORLD_URL)
    parser.add_argument("--app", default="all",
                        help="Comma-separated apps or 'all'")
    args = parser.parse_args()

    apps = (ALL_APPS if args.app == "all"
            else [a.strip() for a in args.app.split(",")])

    server = AppWorldMCPServer(appworld_url=args.appworld_url, apps=apps)
    asyncio.run(server.run())
