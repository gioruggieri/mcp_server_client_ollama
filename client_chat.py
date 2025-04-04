import asyncio
import sys
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import aiohttp

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        if not is_python:
            raise ValueError("Server script must be a .py file")

        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def call_ollama(self, query: str, available_tools: list):
        ollama_url = "http://localhost:11434/api/chat"
        headers = {"Content-Type": "application/json"}

        messages = [
            {"role": "system", "content": (
    "Puoi rispondere normalmente alle domande dell'utente. "
    "Se tra gli strumenti disponibili c'è qualcosa che può aiutarti a rispondere meglio, usalo. "
    "Altrimenti, non usare alcuno strumento."
)},


            {"role": "user", "content": query}
        ]

        data = {
            "model": "qwen2.5:7b",
            "messages": messages,
            "tools": available_tools,
            "tool_choice": "auto",
            "stream": False
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(ollama_url, headers=headers, json=data) as response:
                if response.status == 200:
                    response_data = await response.json()
                    message = response_data.get('message', {})
                    if 'tool_calls' in message:
                        return message['tool_calls']
                    else:
                        return message.get('content', '')
                else:
                    raise Exception(f"Errore Ollama: {response.status} - {await response.text()}")

    async def process_query(self, query: str):
        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema
                }
            } for tool in response.tools
        ]


        try:
            content = await self.call_ollama(query, available_tools)

            if isinstance(content, list):  # tool_calls
                for call in content:
                    name = call.get("function", {}).get("name") or call.get("name")
                    args_raw = call.get("function", {}).get("arguments") or call.get("arguments", {})
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    print(f"\n⚙️ Eseguo tool '{name}' con argomenti: {args}")
                    result = await self.session.call_tool(name, args)
                    result_text = getattr(result.content[0], 'text', result)
                    print(f"\n📦 Risultato da {name}:\n{result_text}")
            else:
                print("\n🤖 Modello:", content)

        except Exception as e:
            print(f"Errore durante la chiamata a Ollama: {str(e)}")

    async def chat_loop(self):
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break

                await self.process_query(query)

            except Exception as e:
                print(f"\nErrore: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python <path_to_client_script> <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
