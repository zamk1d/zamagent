"""
Offline smoke tests: fake Groq (OpenAI SSE) and fake Ollama servers drive the
real agent loop, tool process and file tools. Run:  python -m unittest tests.test_smoke -v
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TMP = tempfile.mkdtemp(prefix="zamagent-test-")
os.chdir(TMP)                       # WORK_DIR is fixed at import time

import providers                    # noqa: E402
from agent import Agent             # noqa: E402
from providers import ToolUseFailed  # noqa: E402

CODE = 'print("hi from mock")'


def _turn(messages):
    """number of tool results after the last real user message"""
    n = 0
    for m in reversed(messages):
        if m["role"] == "tool":
            n += 1
        elif m["role"] == "user" and not m["content"].startswith("Your last tool call"):
            break
    return n


class Fake(BaseHTTPRequestHandler):
    log_message = lambda *a, **k: None
    state = {"seen": [], "429": 0, "bad": 0}

    def _json(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._json(200, {"data": [{"id": "openai/gpt-oss-120b"}, {"id": "whisper-large-v3"}],
                         "models": [{"name": "qwen3:8b"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.state["seen"].append(req)
        s = Fake.state
        if self.path.endswith("/chat/completions"):
            for m in req["messages"]:
                assert not any(k.startswith("_") for k in m), "private key leaked"
            if s["429"]:
                s["429"] -= 1
                return self._json(429, {"error": {"message": "slow down"}}, {"retry-after": "0.1"})
            if s["bad"]:
                s["bad"] -= 1
                return self._json(400, {"error": {"message": "Failed to call a function",
                                                  "code": "tool_use_failed", "failed_generation": "<function=x>"}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self._sse(req)
        else:  # /api/chat
            self.send_response(200)
            self.end_headers()
            self._ollama(req)

    # --- scripted scenario: write_file -> run_python -> final answer -------- #
    def _sse(self, req):
        def send(delta, finish=None, **extra):
            chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        t = _turn(req["messages"])
        if t == 0:
            args = json.dumps({"filepath": "hello.py", "content": CODE})
            send({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                  "function": {"name": "write_file", "arguments": ""}}]})
            for i in range(0, len(args), 7):          # arguments arrive in fragments
                send({"tool_calls": [{"index": 0, "function": {"arguments": args[i:i + 7]}}]})
            send({}, "tool_calls")
        elif t == 1:
            send({"tool_calls": [{"index": 0, "id": "call_2", "type": "function",
                                  "function": {"name": "run_python", "arguments": '{"filepath": "hello.py"}'}}]})
            send({}, "tool_calls")
        else:
            last = [m for m in req["messages"] if m["role"] == "tool"][-1]["content"]
            send({"content": "done: "})
            send({"content": "hi from mock" if "hi from mock" in last else "MISSING"})
            send({}, "stop", x_groq={"usage": {"prompt_tokens": 11, "completion_tokens": 5}})
        self.wfile.write(b"data: [DONE]\n\n")

    def _ollama(self, req):
        assert req["tools"], "tools must be sent"
        for m in req["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                assert isinstance(m["tool_calls"][0]["function"]["arguments"], dict)
            if m["role"] == "tool":
                assert m["tool_name"]
        t = _turn(req["messages"])
        if t == 0:
            msg = {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "write_file", "arguments": {"filepath": "o.py", "content": CODE}}}]}
        elif t == 1:
            msg = {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "run_python", "arguments": {"filepath": "o.py"}}}]}
        else:
            msg = {"role": "assistant", "content": "ollama ok"}
        self.wfile.write((json.dumps({"message": msg, "done": False}) + "\n").encode())
        self.wfile.write((json.dumps({"done": True, "prompt_eval_count": 7, "eval_count": 3}) + "\n").encode())


class SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        Fake.state.update(seen=[], **{"429": 0, "bad": 0})

    def groq(self):
        return providers.GroqProvider("openai/gpt-oss-120b", base_url=f"http://127.0.0.1:{self.port}/openai/v1",
                                      api_key="test")

    def test_groq_full_loop(self):
        agent = Agent(self.groq(), auto_approve=True)
        calls = []
        agent.on_tool_call = lambda n, a: calls.append(n)
        tokens = []
        agent.on_token = tokens.append
        answer = agent.run("make hello.py and run it")
        self.assertEqual(answer, "done: hi from mock")
        self.assertEqual(calls, ["write_file", "run_python"])
        self.assertEqual(open("hello.py").read(), CODE)
        self.assertEqual("".join(tokens), "done: hi from mock")
        self.assertEqual(agent.usage["prompt"], 11)
        first = Fake.state["seen"][0]
        self.assertEqual(first["reasoning_effort"], "medium")
        self.assertEqual(first["tool_choice"], "auto")
        self.assertEqual(len(first["tools"]), 15)
        # memory: second turn keeps the first one
        n = len(agent.messages)
        agent.run("again")
        self.assertGreater(len(agent.messages), n)

    def test_confirmation_denied(self):
        agent = Agent(self.groq(), auto_approve=False)
        agent.confirm = lambda tool, args: "n"
        agent.run("go")
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        self.assertIn("denied", tool_msgs[1]["content"])

    def test_retry_on_429_and_tool_use_failed(self):
        Fake.state.update(**{"429": 2, "bad": 1})
        notes = []
        providers.notice = notes.append
        agent = Agent(self.groq(), auto_approve=True)
        self.assertEqual(agent.run("go"), "done: hi from mock")
        self.assertEqual(len(notes), 2)
        providers.notice = None

    def test_interrupt_rolls_back_memory(self):
        agent = Agent(self.groq(), auto_approve=True)
        agent.run("first")
        keep = list(agent.messages)
        agent.on_tool_call = lambda n, a: (_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            agent.run("second")
        self.assertEqual(agent.messages, keep)

    def test_ollama_native(self):
        p = providers.OllamaProvider("qwen3:8b", host=f"http://127.0.0.1:{self.port}")
        agent = Agent(p, auto_approve=True)
        self.assertEqual(agent.run("go"), "ollama ok")
        self.assertTrue(os.path.exists("o.py"))
        self.assertEqual(p.list_models(), ["qwen3:8b"])

    def test_list_models_filters_audio(self):
        self.assertEqual(self.groq().list_models(), ["openai/gpt-oss-120b"])

    def test_compaction_keeps_valid_history(self):
        agent = Agent(self.groq(), auto_approve=True)
        agent.provider.context_chars = 1500
        agent.run("one")
        agent.run("two")
        agent.run("three")
        self.assertEqual(agent.messages[0]["role"], "user")
        for i, m in enumerate(agent.messages):   # every tool result follows an assistant tool_call
            if m["role"] == "tool":
                self.assertTrue(any(x["role"] == "assistant" and x.get("tool_calls") for x in agent.messages[:i]))

    def test_tools_and_workspace_boundary(self):
        import tools
        from tool_helpers import _resolve_path
        os.makedirs("pkg/sub", exist_ok=True)
        open("pkg/sub/a.txt", "w").write("alpha\nBeta needle\n")
        self.assertIn("pkg/sub/a.txt:2: Beta needle", tools.search_in_files("needle")["result"])
        self.assertIn("pkg/sub/a.txt", tools.find_files("*.txt")["result"])
        self.assertIn("sub/", tools.tree("pkg")["result"])
        self.assertTrue(tools.read_file("pkg/sub/a.txt", 2, 2)["result"].endswith("Beta needle\n"))
        self.assertEqual(tools.edit_file("pkg/sub/a.txt", "alpha", "ALPHA")["status"], "ok")
        self.assertEqual(tools.move_file("pkg/sub/a.txt", "pkg/b.txt")["status"], "ok")
        self.assertEqual(tools.delete_file("pkg/b.txt")["status"], "ok")
        self.assertEqual(tools.run_command("sudo ls")["status"], "error")
        with self.assertRaises(Exception):
            _resolve_path("../outside.txt")
        with self.assertRaises(Exception):
            _resolve_path(os.path.abspath(os.sep))


if __name__ == "__main__":
    unittest.main()
