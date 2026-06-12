"""Minimal raw MCP client for the UE toolset server on :8000.

The harness-level MCP session went stale (Invalid session id) and the harness
caches its session, so we drive the server directly: initialize -> load the
Programmatic toolset -> execute_tool_script. Each invocation creates a fresh
session, which the server accepts.

Usage:
  python tools/ue_mcp.py probe                 # level + settings sanity
  python tools/ue_mcp.py exec <script_file>    # run an execute_tool_script payload
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000/mcp"
EXEC_TOOL = "toolset_registry_toolsets_core_programmatic_ProgrammaticToolset_execute_tool_script"
LOAD_TOOL = "load_toolset"


class McpSession:
    def __init__(self):
        self.sid = None
        self._id = 0

    def post(self, method, params=None, is_notification=False):
        self._id += 1
        payload = {"jsonrpc": "2.0", "method": method}
        if not is_notification:
            payload["id"] = self._id
        if params is not None:
            payload["params"] = params
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream"}
        if self.sid:
            headers["mcp-session-id"] = self.sid
        req = urllib.request.Request(BASE, json.dumps(payload).encode(), headers)
        with urllib.request.urlopen(req, timeout=600) as r:
            sid = r.headers.get("mcp-session-id")
            if sid:
                self.sid = sid
            body = r.read().decode()
        data = None
        for line in body.splitlines():
            if line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                except Exception:
                    pass
        if data is None and body.strip():
            try:
                data = json.loads(body)
            except Exception:
                data = body[:300]
        return data

    def start(self):
        self.post("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "claude-bash", "version": "1.0"}})
        self.post("notifications/initialized", is_notification=True)
        self.post("tools/call", {
            "name": LOAD_TOOL,
            "arguments": {"toolset_name":
                          "toolset_registry.toolsets.core.programmatic.ProgrammaticToolset"}})

    def exec_script(self, script):
        r = self.post("tools/call", {"name": EXEC_TOOL, "arguments": {"script": script}})
        if isinstance(r, dict):
            res = r.get("result", {})
            content = res.get("content", [])
            for c in content:
                if c.get("type") == "text":
                    return c["text"]
            if res.get("structuredContent"):
                return json.dumps(res["structuredContent"], ensure_ascii=False)
            if r.get("error"):
                return "MCP_ERROR: " + json.dumps(r["error"], ensure_ascii=False)
        return str(r)[:500]


PROBE = '''
import json

def run():
    out = {}
    result, error = execute_tool(
        "toolset_registry.toolsets.core.scene.SceneTools", "get_current_level", "{}")
    out["level"] = json.loads(result)["returnValue"] if not error else "ERR: " + str(error)[:150]
    result, error = execute_tool(
        "toolset_registry.toolsets.core.object.ObjectTools", "set_properties",
        json.dumps({"instance": {"refPath": "/Script/PythonScriptPlugin.Default__PythonScriptPluginSettings"},
                    "values": json.dumps({"bRemoteExecution": True})}))
    out["remote_exec_on"] = json.loads(result)["returnValue"] if not error else "ERR: " + str(error)[:150]
    result, error = execute_tool(
        "toolset_registry.toolsets.core.object.ObjectTools", "set_properties",
        json.dumps({"instance": {"refPath": "/Script/UnrealEd.Default__EditorPerformanceSettings"},
                    "values": json.dumps({"bThrottleCPUWhenNotForeground": False})}))
    out["throttle_off"] = json.loads(result)["returnValue"] if not error else "ERR: " + str(error)[:150]
    return out
'''


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    s = McpSession()
    s.start()
    if mode == "probe":
        print(s.exec_script(PROBE))
    elif mode == "exec":
        script = open(sys.argv[2], encoding="utf-8").read()
        print(s.exec_script(script))


if __name__ == "__main__":
    main()
