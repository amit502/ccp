"""
officebench_mcp_server.py

Real MCP server wrapping the OfficeBench REST API.

Started as a subprocess by OfficeBenchMCPRunner:
    python officebench_mcp_server.py --app word --server-url http://localhost:8001

Exposes all office application tools as MCP tools with proper JSON Schema.
Tool calls are forwarded to the real OfficeBench server via HTTP.
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

OFFICEBENCH_URL = os.environ.get("OFFICEBENCH_URL", "http://localhost:8001")

# ---------------------------------------------------------------------------
# Tool definitions per app (match OfficeBench's actual tool interface)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: Dict[str, List[Dict]] = {
    "word": [
        {"name": "word__open_document",   "desc": "Open a Word document by path.",
         "params": {"path": "string"}},
        {"name": "word__read_content",    "desc": "Read full text content of open document.",
         "params": {"doc_id": "string"}},
        {"name": "word__insert_text",     "desc": "Insert text at a position in the document.",
         "params": {"doc_id": "string", "text": "string", "position": "string"}},
        {"name": "word__replace_text",    "desc": "Replace all occurrences of find_text with replace_text.",
         "params": {"doc_id": "string", "find_text": "string", "replace_text": "string"}},
        {"name": "word__add_heading",     "desc": "Add a heading to the document.",
         "params": {"doc_id": "string", "text": "string", "level": "integer"}},
        {"name": "word__add_table",       "desc": "Add a table with given rows and columns.",
         "params": {"doc_id": "string", "rows": "integer", "cols": "integer"}},
        {"name": "word__save_document",   "desc": "Save document to path.",
         "params": {"doc_id": "string", "path": "string"}},
        {"name": "word__close_document",  "desc": "Close an open document.",
         "params": {"doc_id": "string"}},
    ],
    "excel": [
        {"name": "excel__open_workbook",  "desc": "Open an Excel workbook by path.",
         "params": {"path": "string"}},
        {"name": "excel__read_cell",      "desc": "Read value from a cell.",
         "params": {"workbook_id": "string", "sheet": "string", "cell": "string"}},
        {"name": "excel__read_range",     "desc": "Read a range of cells.",
         "params": {"workbook_id": "string", "sheet": "string", "range": "string"}},
        {"name": "excel__write_cell",     "desc": "Write a value to a cell.",
         "params": {"workbook_id": "string", "sheet": "string", "cell": "string", "value": "string"}},
        {"name": "excel__apply_formula",  "desc": "Apply a formula to a cell.",
         "params": {"workbook_id": "string", "sheet": "string", "cell": "string", "formula": "string"}},
        {"name": "excel__create_chart",   "desc": "Create a chart from a data range.",
         "params": {"workbook_id": "string", "chart_type": "string", "data_range": "string", "title": "string"}},
        {"name": "excel__save_workbook",  "desc": "Save workbook.",
         "params": {"workbook_id": "string", "path": "string"}},
    ],
    "powerpoint": [
        {"name": "powerpoint__open_presentation",  "desc": "Open a PowerPoint file.",
         "params": {"path": "string"}},
        {"name": "powerpoint__add_slide",          "desc": "Add a new slide.",
         "params": {"pptx_id": "string", "layout": "string", "title": "string"}},
        {"name": "powerpoint__add_text_box",       "desc": "Add a text box to a slide.",
         "params": {"pptx_id": "string", "slide_num": "integer", "text": "string"}},
        {"name": "powerpoint__save_presentation",  "desc": "Save presentation.",
         "params": {"pptx_id": "string", "path": "string"}},
    ],
    "email": [
        {"name": "email__list_inbox",    "desc": "List emails in inbox.",
         "params": {"limit": "integer"}},
        {"name": "email__read_email",    "desc": "Read full content of an email by ID.",
         "params": {"email_id": "string"}},
        {"name": "email__reply_email",   "desc": "Reply to an email.",
         "params": {"email_id": "string", "body": "string"}},
        {"name": "email__send_email",    "desc": "Compose and send a new email.",
         "params": {"to": "string", "subject": "string", "body": "string"}},
        {"name": "email__search_emails", "desc": "Search emails by query string.",
         "params": {"query": "string"}},
    ],
    "calendar": [
        {"name": "calendar__list_events",   "desc": "List upcoming calendar events.",
         "params": {"days_ahead": "integer"}},
        {"name": "calendar__create_event",  "desc": "Create a calendar event.",
         "params": {"title": "string", "date": "string", "time": "string",
                    "duration_mins": "integer", "attendees": "string"}},
        {"name": "calendar__update_event",  "desc": "Update an existing event.",
         "params": {"event_id": "string", "title": "string", "date": "string"}},
        {"name": "calendar__delete_event",  "desc": "Delete a calendar event.",
         "params": {"event_id": "string"}},
    ],
    "file_manager": [
        {"name": "file_manager__list_directory", "desc": "List directory contents.",
         "params": {"path": "string"}},
        {"name": "file_manager__move_file",      "desc": "Move a file to a destination.",
         "params": {"src": "string", "dest": "string"}},
        {"name": "file_manager__copy_file",      "desc": "Copy a file.",
         "params": {"src": "string", "dest": "string"}},
        {"name": "file_manager__delete_file",    "desc": "Delete a file.",
         "params": {"path": "string"}},
        {"name": "file_manager__create_directory","desc": "Create a directory.",
         "params": {"path": "string"}},
        {"name": "file_manager__search_files",   "desc": "Search files matching a pattern.",
         "params": {"directory": "string", "pattern": "string"}},
    ],
}

TYPE_MAP = {
    "string":  {"type": "string"},
    "integer": {"type": "integer"},
    "number":  {"type": "number"},
    "boolean": {"type": "boolean"},
}


class OfficeBenchMCPServer:
    def __init__(self, apps: List[str], server_url: str):
        self.apps       = apps
        self.server_url = server_url.rstrip("/")
        self.server     = Server("officebench-mcp")
        self._session   = requests.Session()
        self._task_id: str = ""
        self._register_handlers()

    def set_task(self, task_id: str) -> None:
        self._task_id = task_id

    def _build_tool(self, schema: Dict) -> mcp_types.Tool:
        props    = {k: TYPE_MAP.get(v, {"type": "string"})
                    for k, v in schema["params"].items()}
        required = list(schema["params"].keys())
        return mcp_types.Tool(
            name=schema["name"],
            description=schema["desc"],
            inputSchema={"type": "object", "properties": props, "required": required},
        )

    def _register_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> List[mcp_types.Tool]:
            tools = []
            for app in self.apps:
                for schema in TOOL_SCHEMAS.get(app, []):
                    tools.append(self._build_tool(schema))
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[mcp_types.TextContent]:
            """Forward tool call to OfficeBench REST server."""
            try:
                payload  = {"tool": name, "params": arguments, "task_id": self._task_id}
                response = self._session.post(
                    f"{self.server_url}/tasks/{self._task_id}/execute",
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                output = json.dumps(result)
            except requests.HTTPError as e:
                output = json.dumps({"error": str(e), "status": "error"})
            except requests.Timeout:
                output = json.dumps({"error": "OfficeBench server timeout", "status": "error"})
            except Exception as e:
                output = json.dumps({"error": str(e), "status": "error"})

            return [mcp_types.TextContent(type="text", text=output)]

    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app",        default="word",
                        help="Comma-separated list of apps to expose")
    parser.add_argument("--server-url", default=OFFICEBENCH_URL,
                        help="OfficeBench REST server URL")
    parser.add_argument("--task-id",    default="",
                        help="Task ID to use for REST calls")
    args = parser.parse_args()

    apps   = [a.strip() for a in args.app.split(",")]
    server = OfficeBenchMCPServer(apps=apps, server_url=args.server_url)
    server.set_task(args.task_id)
    asyncio.run(server.run())
