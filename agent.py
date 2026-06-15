import subprocess
import json

import requests

proc = subprocess.Popen(
    ["python", "mcp_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1
)

def call_mcp(tool, arguments):
    request = {
        "tool": tool,
        "arguments": arguments
    }

    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()

    response_line = proc.stdout.readline()
    return json.loads(response_line)

def ask_llm(message):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "system",
                    "content": """
You are an agent.

Available tools:
- read_file(filepath)

If you use a tool, respond ONLY in JSON:
{"tool":"...","arguments":{...}}
"""
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            "stream": False
        }
    )
    print("waiting for response..")
    return response.json()["message"]["content"]

def run(user_input):
    llm_response = ask_llm(user_input)
    print(f"LLM: {llm_response}")

    try:
        data = json.loads(llm_response)
    except:
        print("\nANSWER:")
        print(llm_response)
        return

    if "tool" in data:
        tool = data["tool"]
        args = data.get("arguments", {})

        print(f"\nAGENT calling tool: {tool}")

        result = call_mcp(tool, args)

        print(f"MCP result: {result}")

        follow_up = ask_llm(
            f"""
User asked: {user_input}

Tool used: {tool}
Result: {result}

Now give final answer to user on the same language that user wrote input
"""
        )
        print("\nQWEN:")
        print(follow_up)

while True:
    user_input = input("\n> ")
    run(user_input)