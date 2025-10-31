# agent.py — LLM-driven agent that decides which MCP tool to call and executes it.
# Default LLM backend: Ollama (local, free). Swap call_llm() to use OpenAI/Claude if you want.

import asyncio
import json
import sys
from typing import Dict, Any
import requests  # used for Ollama; remove if you switch providers

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ---------- LLM BACKEND (Ollama example) ----------
# Runs a local model via Ollama's HTTP API (http://localhost:11434)
# Install: https://ollama.com  |  Run a model first: `ollama run llama3.1`

OLLAMA_MODEL = "llama3.1"  # change to what you have locally

def call_llm(system: str, user: str) -> str:
    """
    Returns the LLM's raw text output. We instruct it to ONLY return JSON.
    Replace this with your preferred provider if needed.
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Ollama returns concatenated message content
    return data.get("message", {}).get("content", "").strip()


# ---------- UTIL ----------
AGENT_SYSTEM_PROMPT = """You are a tool-using data analyst. You have access to tools via MCP.
You must respond with ONLY one JSON object on a single line, no prose.

JSON schema:
{
  "tool": "<tool_name>",
  "args": { ... JSON-safe arguments ... },
  "reason": "<brief why this tool & args>",
  "next": "<optional follow-up user question to ask after executing>"
}

Rules:
- Choose exactly ONE tool per turn from the available tool list I will give you.
- If a dataset must be loaded, call tool_load_data first with {"path":"<csv_path>"}.
- When summarizing data, use tool_summary with {"numeric_only": true|false}.
- For categories, tool_top_categories with {"columns":[...], "top_n": N}.
- For correlations, tool_correlations with {"columns":[...]} or omit to use all numeric.
- For outliers, tool_outliers with {"columns":[...], "z": 3.0}.
- For time trends, tool_time_trend with {"column":"...", "freq":"M"} or "Y".
- If the user asks a question that needs seeing rows, use tool_head with {"n": 5}.
- If you lack a dataset path and none is loaded, ask the user for a CSV path in "next".
- Output strictly valid JSON. Do not include backticks or comments.
"""

def dump_tool_result(r):
    """Print MCP result regardless of shape; keep concise."""
    if getattr(r, "structuredContent", None):
        print(json.dumps(r.structuredContent, indent=2))
        return
    parts = getattr(r, "content", []) or []
    if not parts:
        print("{}")
        return
    # Print first JSON part if present; otherwise show all
    printed = False
    for i, p in enumerate(parts):
        t = getattr(p, "type", None)
        if t == "json":
            print(json.dumps(p.data, indent=2))
            printed = True
            break
    if not printed:
        # Fallback: print all text parts
        for i, p in enumerate(parts):
            if getattr(p, "type", None) == "text":
                print(getattr(p, "text", ""))


async def run_agent():
    # 1) Start MCP server
    params = StdioServerParameters(command="python", args=["server.py"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 2) Discover tools and feed to LLM as context
            tools_resp = await session.list_tools()
            tool_names = [t.name for t in tools_resp.tools]

            print("✅ MCP connected. Tools available:")
            for n in tool_names:
                print(" -", n)
            print("\nType your request (or 'exit'):")

            while True:
                try:
                    user_msg = input("\nYou: ").strip()
                    if user_msg.lower() in ("exit", "quit"):
                        print("👋 Bye.")
                        break
                    if not user_msg:
                        continue

                    tool_list_str = ", ".join(tool_names)
                    prompt_user = (
                        f"Available tools: [{tool_list_str}].\n"
                        f"User request: {user_msg}\n"
                        f"Return only the JSON as specified."
                    )

                    # 3) Ask LLM to pick ONE tool + args as JSON
                    raw = call_llm(AGENT_SYSTEM_PROMPT, prompt_user)

                    # 4) Parse LLM JSON
                    try:
                        plan = json.loads(raw)
                    except json.JSONDecodeError:
                        print("LLM did not return valid JSON. Raw:\n", raw)
                        continue

                    tool = plan.get("tool")
                    args = plan.get("args", {}) or {}

                    if tool not in tool_names:
                        print(f"⚠️ LLM chose unknown tool: {tool}\nPlan:", plan)
                        continue

                    print(f"\n🤖 LLM plan: call {tool} with args {args}")
                    if plan.get("reason"):
                        print("Reason:", plan["reason"])

                    # 5) Execute tool
                    result = await session.call_tool(tool, arguments=args)

                    # 6) Print result
                    print("\n📦 Tool result:")
                    dump_tool_result(result)

                    # 7) Ask follow-up if LLM suggested one
                    nxt = plan.get("next")
                    if nxt:
                        print("\n🤖 Next question:", nxt)

                except KeyboardInterrupt:
                    print("\n👋 Bye.")
                    break


if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except RuntimeError as e:
        # If you're in Jupyter/IPython with a running loop:
        #   await run_agent()
        print("RuntimeError:", e, file=sys.stderr)
        raise
