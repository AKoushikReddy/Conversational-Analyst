# client.py — interactive MCP client (fixed graceful exit)

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

def dump_tool_result(r):
    if getattr(r, "structuredContent", None):
        try:
            print(json.dumps(r.structuredContent, indent=2))
            return
        except Exception:
            pass
    parts = getattr(r, "content", []) or []
    if not parts:
        print("{}")
        return
    for i, p in enumerate(parts):
        ptype = getattr(p, "type", None)
        if ptype == "json":
            try:
                print(f"[part {i} json]")
                print(json.dumps(p.data, indent=2))
            except Exception as e:
                print(f"[part {i} json](unprintable): {e}")
        elif ptype == "text":
            print(f"[part {i} text]")
            print(getattr(p, "text", ""))
        else:
            print(f"[part {i} {ptype}] {p}")

async def main():
    params = StdioServerParameters(command="python", args=["server.py"])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\n✅ Connected to MCP server.")
            tools = await session.list_tools()
            print("Available tools:")
            for t in tools.tools:
                print(" -", t.name)

            print("\n💬 Enter: <tool_name> <JSON-args>")
            print('   e.g., tool_load_data {"path": "netflix_titles.csv"}')
            print('         tool_summary {"numeric_only": true}')
            print("   Type 'exit' to quit.\n")

            try:
                while True:
                    line = input(">>> ").strip()
                    if not line:
                        continue
                    if line.lower() in ("exit", "quit"):
                        print("👋 Exiting client.")
                        break

                    if " " in line:
                        tool, json_part = line.split(" ", 1)
                        try:
                            args = json.loads(json_part)
                        except json.JSONDecodeError:
                            print("❌ Invalid JSON. Example: tool_summary {\"numeric_only\": true}")
                            continue
                    else:
                        tool, args = line, {}

                    print(f"\n🚀 Calling: {tool}  args={args}")
                    res = await session.call_tool(tool, arguments=args)
                    dump_tool_result(res)

            except KeyboardInterrupt:
                print("\n👋 Exiting client.")
            finally:
                # No session.shutdown() in this SDK; context manager will close it.
                try:
                    await session.close()  # some versions support this
                except AttributeError:
                    pass  # safe to ignore; exiting 'async with' handles cleanup

if __name__ == "__main__":
    asyncio.run(main())
