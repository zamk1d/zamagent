# zamagent

Your own terminal agent: reads, searches, edits and runs code in the current folder.
Works with **Groq** (fast cloud models) and local **Ollama**.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # put your GROQ_API_KEY there (https://console.groq.com/keys)
python /path/to/zamagent.py # run it inside the project you want to work on
```

No `GROQ_API_KEY`? It falls back to Ollama (`qwen3:8b`, or any tool-capable model).

```bash
python zamagent.py                              # interactive REPL
python zamagent.py "add type hints to utils.py" # one-shot
python zamagent.py -p groq -m fast "explain main.py"
python zamagent.py -p ollama -m qwen3:8b -y "run the tests and fix failures"
python zamagent.py -p groq --list-models
```

Groq aliases for `-m` / `/model`: `smart` (gpt-oss-120b), `oss-small` (gpt-oss-20b), `llama` (llama-3.3-70b-versatile), `fast` (llama-3.1-8b-instant).
Any other id works too; `/models` shows what your key can use right now.

## What it can do

* **Native tool calling** (no more "answer with a raw JSON array" prompt hacks) with streaming, parallel tool calls and argument coercion.
* **Memory** across messages in a session; automatic history compaction; `/save`, `/load`.
* **15 tools**: `tree`, `find_files`, `search_in_files` (grep), `read_file` (line ranges), `write_file`, `append_file`, `edit_file`, `delete_file`, `move_file`, `create_directory`, `run_python`, `run_command`, `fetch_url`, ...
  Add a function to `tools.py` (type hints + docstring with `:param x:` lines) and it becomes a tool automatically.
* **Safety**: everything is confined to the workspace (symlinks resolved); `run_command`, `run_python`, `delete_file`, `move_file` ask for approval (`y` / `n` / `a` = always). `-y` or `/yes` auto-approves. Edits are shown as diffs.
  The command blocklist is a seatbelt, not a sandbox - the approval prompt is the real guard.
* **Resilience**: retries with backoff on 429/5xx (honours `retry-after`), recovers from Groq `tool_use_failed`, shrinks the request on 413, loop detection, Ctrl+C never corrupts memory.
* **Project instructions**: drop a `ZAMAGENT.md` (or `AGENTS.md`) into the workspace and it is added to the system prompt.

## REPL commands

`/model`, `/provider`, `/models`, `/yes`, `/tools`, `/usage`, `/clear`, `/save`, `/load`, `/help`, `exit`.
Switching provider/model keeps the conversation.

## Layout

| file | role |
| --- | --- |
| `zamagent.py` | CLI / REPL (rich UI) |
| `agent.py` | agent loop, memory, approvals, tool process client |
| `providers.py` | Groq, Ollama, generic OpenAI-compatible providers |
| `tools.py` / `tools_inspector.py` | tools and automatic JSON-schema generation |
| `mcp_server.py` | separate process that executes tools |
| `tests/test_smoke.py` | offline tests with fake Groq/Ollama servers (`python -m unittest tests.test_smoke`) |

Env vars: see `.env.example`.
