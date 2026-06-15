import subprocess
import json
import datetime
import requests

from tools_inspector import get_tools_list

MAX_STEPS = 10

proc = subprocess.Popen(
    ["python", "mcp_server.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1
)

def render_tools():
    tools = get_tools_list()
    tools_text = ""

    for tool_name, tool_info in tools.items():
        args = tool_info.get("args", {})

        args_str = ", ".join(
            f"{name}: {type_}"
            for name, type_ in args.items()
        )

        tools_text += f"- {tool_name}({args_str})"

        if description := tool_info.get("description"):
            tools_text += f"\n  Description: {description}"

        if returns := tool_info.get("returns"):
            tools_text += f"\n  Returns: {returns}"

        tools_text += "\n\n"

    return tools_text

def call_mcp(tool, arguments):
    request = {
        "tool": tool,
        "arguments": arguments
    }

    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()

    response_line = proc.stdout.readline()
    return json.loads(response_line)

def ask_llm(messages):
    # print(f"request created: {datetime.datetime.now()}")
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3:8b",
            "messages": messages,
            "think": False,
            "stream": False
        }
    )
    # print(f"response time: {datetime.datetime.now()}")
    print("\nLLM:", response.json()["message"]["content"])
    return response.json()["message"]["content"]

def get_system_prompt():
    tools = render_tools()
    prompt = \
f"""
You are an agent.

You can use tools.

Available tools:
{tools}

If you want to use a tool, respond ONLY in JSON:
{{"tool":"...","arguments":{{...}}}}

You may call tools multiple times (currently 1 per response).
Use as many tool calls as needed to complete the task.
Only provide a final answer when you have enough information.
Tool results are authoritative.
Do not verify tool results by calling other tools unless absolutely necessary.
If you are done, respond in normal text.
"""
    return {"role": "system", "content": prompt}

def create_initial_messages(user_input):
    messages = [
        get_system_prompt(),
        {
            "role": "user",
            "content": user_input
        }
    ]
    return messages

def return_final_answer(response):
    # print(f"\nAnswer:\n\t{response}")
    return f"\nAnswer:\n\t{response}"

def parse_tool_call(response) -> dict:
    try:
        data = json.loads(response)
        if "tool" in data and "arguments" in data:
            return {"final": False, "data": data}
    except json.JSONDecodeError:
        final_response = return_final_answer(response)
        return {"final": True, "data": final_response}
    except Exception as e:
        raise e

def execute_tool(data):
    result = call_mcp(tool=data["tool"], arguments=data.get("arguments", {}))
    print(f"\t[AGENT] calling {data["tool"]}")
    # print(f"\t\t[MCP] result:\n\t\t{result["result"][:30]}...")
    return result

def update_messages(messages: list, response, result, tool):
    messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
    messages.append(
        {
            "role": "tool",
            "name": tool,
            "content": json.dumps(result),
        }
    )
    return messages

def run(user_input):
    executed = set()
    messages = create_initial_messages(user_input)
    while True:
        response = ask_llm(messages)

        tool_call: dict = parse_tool_call(response)

        if tool_call["final"]:
            return tool_call["data"]

        result = execute_tool(tool_call["data"])
        messages = update_messages(messages, response, result, tool_call["data"]["tool"])

#
# if __name__ == "__main__":
#     user = input(">")
#     print(run(user))