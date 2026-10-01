"""
LLM providers for zamagent.

* GroqProvider    - Groq Cloud (OpenAI-compatible, native tool calling, SSE streaming)
* NvidiaProvider  - NVIDIA NIM / build.nvidia.com (OpenAI-compatible, NVIDIA_API_KEY)
* GeminiProvider  - Google Gemini via its OpenAI-compatible endpoint (GEMINI_API_KEY)
* OllamaProvider  - local Ollama (/api/chat, native tool calling)
* OpenAICompatProvider - any other OpenAI-compatible endpoint (OPENAI_BASE_URL)

Internally the conversation always uses the OpenAI message format, so the
provider / model can be switched in the middle of a session.
Keys starting with "_" inside a message are private and never sent.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_RETRIES = 5

# UI hook: providers.notice = lambda msg: ...
notice: Callable[[str], None] | None = None


def _say(msg: str) -> None:
    if notice:
        notice(msg)


# --- debugging: AGENT_DEBUG=1 or --debug; details go to zamagent-debug.log ----
DEBUG = os.getenv("AGENT_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
DEBUG_LOG = os.path.join(BASE_DIR, "zamagent-debug.log")
READ_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "90") or 90)   # max silence from the server, seconds
MAX_TIMEOUT_RETRIES = 1


def set_debug(on: bool = True) -> None:
    global DEBUG
    DEBUG = on


def _dbg(msg: str, console: bool = True) -> None:
    """Debug line -> log file (always when DEBUG) and, optionally, the console."""
    if not DEBUG:
        return
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass
    if console:
        _say("debug: " + msg[:200].replace("[", "(").replace("]", ")"))


class _Heartbeat:
    """Tells the user what we are waiting for, so a slow server does not look like a hang."""

    def __init__(self, what: str, state=None):
        self.what, self.state = what, state
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        every = 10 if DEBUG else 15
        while not self._stop.wait(every):
            extra = f" ({self.state()})" if self.state else ""
            _say(f"still waiting for {self.what}: {time.monotonic() - self._t0:.0f}s{extra}")

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

ENV_SOURCES: dict[str, str] = {}   # KEY -> file it was loaded from (absent = real environment)


def _clean_value(value: str) -> str:
    value = value.strip()
    if value[:1] in ("'", '"'):                       # quoted: take everything up to the closing quote
        end = value.find(value[0], 1)
        return value[1:end] if end != -1 else value[1:]
    m = re.search(r"\s#", value)                      # unquoted: drop trailing "  # comment"
    return (value[:m.start()] if m else value).strip()


def load_env() -> None:
    """Tiny .env loader (no extra dependency). Non-empty real env vars always win."""
    for folder in (BASE_DIR, os.getcwd()):
        path = os.path.join(folder, ".env")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:]
                key, val = line.split("=", 1)
                key = key.strip()
                if not os.environ.get(key):
                    os.environ[key] = _clean_value(val)
                    ENV_SOURCES[key] = path


GROQ_ALIASES = {
    "smart": "openai/gpt-oss-120b",
    "oss": "openai/gpt-oss-120b",
    "oss-small": "openai/gpt-oss-20b",
    "llama": "qwen/qwen3.6-27b",        # llama-3.3-70b-versatile was retired 2026-08-16
    "qwen": "qwen/qwen3.6-27b",
    "fast": "openai/gpt-oss-20b",       # llama-3.1-8b-instant was retired 2026-08-16
}

# NVIDIA retires free-endpoint models constantly (HTTP 410), so no hardcoded aliases:
# the default model is auto-detected (see NvidiaProvider). Use -m <full id> to pin one.
NVIDIA_ALIASES: dict[str, str] = {}

_ALIASES = {"groq": GROQ_ALIASES, "nvidia": NVIDIA_ALIASES}


def resolve_model(provider: str, model: str | None) -> str | None:
    if model and provider in _ALIASES:
        return _ALIASES[provider].get(model.lower(), model)
    return model


# --------------------------------------------------------------------------- #
# Types                                                                        #
# --------------------------------------------------------------------------- #

class ProviderError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class ProviderTimeout(ProviderError):
    """The server did not answer (or went silent) for AGENT_TIMEOUT seconds."""


class ToolUseFailed(ProviderError):
    """The model produced a malformed tool call (Groq: code 'tool_use_failed')."""

    def __init__(self, message: str, failed_generation: str = ""):
        super().__init__(message, 400)
        self.failed_generation = failed_generation


@dataclass
class LLMResponse:
    content: str = ""
    # [{"id": str, "name": str, "arguments": dict | None, "raw": str}]
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)  # prompt_tokens / completion_tokens


def _parse_args(raw) -> dict | None:
    if isinstance(raw, dict):
        return raw
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _new_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()


def _clean(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if not k.startswith("_")}
        if m.get("role") == "assistant" and m.get("tool_calls") and not m.get("content"):
            m["content"] = None
        out.append(m)
    return out


def _error_info(resp: requests.Response) -> tuple[str, str, str]:
    """-> (message, code, failed_generation)"""
    try:
        err = resp.json().get("error", {})
        if isinstance(err, str):
            return err, "", ""
        return (
            err.get("message") or resp.text[:300],
            err.get("code") or "",
            err.get("failed_generation") or "",
        )
    except Exception:
        return resp.text[:300], "", ""


# --------------------------------------------------------------------------- #
# Base                                                                         #
# --------------------------------------------------------------------------- #

class Provider:
    name = "base"
    tool_result_limit = 8000      # chars of a single tool result kept in context
    context_chars = 200_000       # rough history budget before compaction

    def __init__(self, model: str):
        self.model = model

    @property
    def label(self) -> str:
        return f"{self.name}/{self.model}"

    def chat(self, messages: list[dict], tools: list[dict],
             on_token: Callable[[str], None] | None = None) -> LLMResponse:
        raise NotImplementedError

    def list_models(self) -> list[str]:
        return []


# --------------------------------------------------------------------------- #
# OpenAI-compatible (Groq is a subclass)                                       #
# --------------------------------------------------------------------------- #

def _stream_chunk(provider, data: str, parts: list, calls: dict, usage: dict,
                  on_token, stats: dict) -> bool:
    """Handle one SSE data payload. Mutates parts/calls/usage. Returns True to stop."""
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        _dbg(f"bad json chunk: {data[:200]!r}", console=False)
        return False

    if chunk.get("error"):
        err = chunk["error"]
        _dbg(f"error chunk: {err}", console=False)
        if isinstance(err, dict) and err.get("code") == "tool_use_failed":
            raise ToolUseFailed(err.get("message", ""), err.get("failed_generation", ""))
        raise ProviderError(str(err.get("message", err)) if isinstance(err, dict) else str(err))

    u = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")
    if u:
        usage.update(u)

    for choice in chunk.get("choices") or []:
        if choice.get("finish_reason"):
            stats["finish"] = choice["finish_reason"]
        delta = choice.get("delta") or {}
        text = delta.get("content")
        if text:
            if stats["first_text"] is None:
                stats["first_text"] = time.monotonic() - stats["t0"]
                _dbg(f"first text token after {stats['first_text']:.1f}s")
            parts.append(text)
            if on_token:
                on_token(text)
        for tc in delta.get("tool_calls") or []:
            fn = tc.get("function") or {}
            idx = tc.get("index")
            if idx is None:
                # Gemini's OpenAI layer may omit "index": a chunk that names a function is a
                # new call, a chunk with only argument pieces continues the last one.
                idx = len(calls) if (fn.get("name") or not calls) else max(calls)
            entry = calls.setdefault(idx, {"id": "", "name": "", "args": ""})
            if tc.get("id"):
                entry["id"] = tc["id"]
            if tc.get("extra_content"):          # Gemini 3 "thought_signature": must be echoed back
                entry["extra"] = tc["extra_content"]
            if fn.get("name") and not entry["name"]:
                entry["name"] = fn["name"]
            piece = fn.get("arguments")
            if isinstance(piece, dict):
                piece = json.dumps(piece)
            if piece:
                entry["args"] += piece
    return False


class OpenAICompatProvider(Provider):
    name = "openai"
    default_base_url = "https://api.openai.com/v1"
    env_key = "OPENAI_API_KEY"
    env_base_url = "OPENAI_BASE_URL"
    key_hint = ""

    def __init__(self, model: str, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.2):
        super().__init__(model)
        self.base_url = (base_url or os.getenv(self.env_base_url) or self.default_base_url).rstrip("/")
        self.api_key = (api_key or os.getenv(self.env_key, "")).strip()
        self.temperature = temperature
        self.context_chars = int(os.getenv("AGENT_CONTEXT_CHARS", self.context_chars))

    # -- hooks -------------------------------------------------------------- #
    def extra_params(self) -> dict:
        return {}

    def prepare_tools(self, tools: list[dict]) -> list[dict]:
        return tools

    def _headers(self) -> dict:
        if not self.api_key:
            raise ProviderError(
                f"{self.env_key} is not set. Put it in .env (see .env.example) or export it."
            )
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    # -- HTTP with retries -------------------------------------------------- #
    def _post(self, payload: dict) -> requests.Response:
        url = f"{self.base_url}/chat/completions"
        delay = 1.5
        timeouts = 0
        size = len(json.dumps(payload, ensure_ascii=False))
        for attempt in range(1, MAX_RETRIES + 1):
            t0 = time.monotonic()
            _dbg(f"POST {url} model={self.model} messages={len(payload['messages'])} "
                 f"tools={len(payload.get('tools') or [])} body={size} bytes "
                 f"timeout=10s connect/{READ_TIMEOUT:.0f}s read attempt={attempt}")
            try:
                with _Heartbeat(f"{self.model} to start answering"):
                    resp = requests.post(url, headers=self._headers(), json=payload,
                                         stream=True, timeout=(10, READ_TIMEOUT))
            except requests.Timeout as exc:
                kind = "connect" if isinstance(exc, requests.ConnectTimeout) else "read"
                _dbg(f"TIMEOUT ({kind}) after {time.monotonic() - t0:.1f}s: {exc}")
                timeouts += 1
                if timeouts > MAX_TIMEOUT_RETRIES or attempt == MAX_RETRIES:
                    raise ProviderTimeout(
                        f"no response from {self.model} for {READ_TIMEOUT:.0f}s ({kind} timeout). "
                        f"The server is overloaded or queueing the request. "
                        f"Try another model (-m), raise AGENT_TIMEOUT, or run with --debug.")
                _say(f"{self.model}: no answer in {READ_TIMEOUT:.0f}s, retry {timeouts}/{MAX_TIMEOUT_RETRIES}")
                continue
            except requests.ConnectionError as exc:
                _dbg(f"CONNECTION ERROR after {time.monotonic() - t0:.1f}s: {type(exc).__name__}: {exc}")
                if attempt == MAX_RETRIES:
                    raise ProviderError(f"connection failed: {exc}")
                _say(f"connection problem ({type(exc).__name__}), retry {attempt}/{MAX_RETRIES}")
                time.sleep(delay)
                delay *= 2
                continue

            interesting = {k: v for k, v in resp.headers.items()
                           if k.lower().startswith(("x-", "nvcf", "retry", "server", "cf-", "content-type"))}
            _dbg(f"HTTP {resp.status_code} after {time.monotonic() - t0:.1f}s headers={interesting}")
            if resp.status_code == 200:
                return resp

            message, code, failed = _error_info(resp)
            _dbg(f"error body: {message[:1500]}", console=False)
            status = resp.status_code
            resp.close()

            if code == "tool_use_failed":
                raise ToolUseFailed(message, failed)

            if status == 401:
                key = self.api_key
                src = ENV_SOURCES.get(self.env_key, "the shell environment (not .env!)")
                raise ProviderError(
                    f"401 {message}. {self.env_key} was taken from {src}: {len(key)} chars, "
                    f"starts with {key[:4]!r}, ends with {key[-2:]!r}. {self.key_hint} "
                    f"If it comes from the shell, remove it (bash: unset {self.env_key} / "
                    f"fish: set -e {self.env_key}) so .env is used.", 401)

            if status in (429, 500, 502, 503, 504):
                if status == 429 and re.search(r"per ?day|daily", message, re.I):
                    raise ProviderError(f"daily quota exhausted: {message}", status)
                try:
                    wait = float(resp.headers.get("retry-after", ""))
                except ValueError:
                    m = re.search(r"retry in ([\d.]+)s", message)      # Gemini puts it in the body
                    wait = float(m.group(1)) + 1 if m else delay
                if wait > 60:
                    raise ProviderError(
                        f"rate limit reached, the API asks to wait {wait:.0f}s: {message}", status)
                if attempt < MAX_RETRIES:
                    _say(f"{status} {message[:90]} - retry in {wait:.1f}s ({attempt}/{MAX_RETRIES})")
                    time.sleep(wait)
                    delay *= 2
                    continue
            raise ProviderError(f"{status}: {message}", status)
        raise ProviderError("request failed")

    # -- chat --------------------------------------------------------------- #
    def chat(self, messages, tools, on_token=None) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": _clean(messages),
            "stream": True,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = self.prepare_tools(tools)
            payload["tool_choice"] = "auto"
        payload.update(self.extra_params())

        resp = self._post(payload)
        parts: list[str] = []
        calls: dict[int, dict] = {}
        usage: dict = {}
        stats = {"t0": time.monotonic(), "chunks": 0, "keepalive": 0, "last": time.monotonic(),
                 "first_data": None, "first_text": None, "finish": None}

        def _state() -> str:
            return (f"{stats['chunks']} chunks, {len(parts)} text parts, "
                    f"last data {time.monotonic() - stats['last']:.0f}s ago")

        try:
            with _Heartbeat(f"{self.model} stream", _state):
              try:
                lines = resp.iter_lines()
                for raw in lines:
                    stats["last"] = time.monotonic()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", "replace")
                    if not line.startswith("data:"):
                        stats["keepalive"] += 1
                        _dbg(f"non-data line: {line[:120]!r}", console=False)
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    stats["chunks"] += 1
                    if stats["first_data"] is None:
                        stats["first_data"] = time.monotonic() - stats["t0"]
                        _dbg(f"first data chunk after {stats['first_data']:.1f}s (body read)")
                    if _stream_chunk(self, data, parts, calls, usage, on_token, stats):
                        break
              except requests.Timeout as exc:
                _dbg(f"STREAM STALLED: no data for {READ_TIMEOUT:.0f}s ({_state()}): {exc}")
                raise ProviderTimeout(
                    f"{self.model} went silent for {READ_TIMEOUT:.0f}s mid-answer ({_state()}).")
              except requests.ConnectionError as exc:
                _dbg(f"STREAM CONNECTION ERROR: {type(exc).__name__}: {exc} ({_state()})")
                raise ProviderError(f"connection dropped mid-answer: {exc}")
        finally:
            resp.close()
            _dbg(f"stream done in {time.monotonic() - stats['t0']:.1f}s chunks={stats['chunks']} "
                 f"keepalive={stats['keepalive']} finish={stats['finish']} usage={usage}")

        tool_calls = []
        for idx in sorted(calls):
            c = calls[idx]
            if not c["name"]:
                continue
            call = {
                "id": c["id"] or _new_id(),
                "name": c["name"],
                "arguments": _parse_args(c["args"]),
                "raw": c["args"],
            }
            if c.get("extra"):
                call["extra_content"] = c["extra"]
            tool_calls.append(call)
        return LLMResponse(_strip_think("".join(parts)), tool_calls, usage)

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=15)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"cannot list models: {exc}")
        ids = sorted(m["id"] for m in r.json().get("data", []))
        return [i for i in ids if not re.search(r"whisper|orpheus|tts|guard|safeguard|embed", i)]


class GroqProvider(OpenAICompatProvider):
    name = "groq"
    default_base_url = "https://api.groq.com/openai/v1"
    env_key = "GROQ_API_KEY"
    env_base_url = "GROQ_BASE_URL"
    key_hint = "Groq keys start with 'gsk_'."
    tool_result_limit = 10_000
    context_chars = 120_000   # free tier has small TPM limits; override with AGENT_CONTEXT_CHARS

    def __init__(self, model: str, **kw):
        super().__init__(model, **kw)
        self.reasoning_effort = os.getenv("GROQ_REASONING_EFFORT", "medium")

    def extra_params(self) -> dict:
        if self.model.startswith("openai/gpt-oss") and self.reasoning_effort:
            return {"reasoning_effort": self.reasoning_effort}
        return {}

    def _headers(self) -> dict:
        if not self.api_key:
            raise ProviderError(
                "GROQ_API_KEY is not set. Get a key at https://console.groq.com/keys "
                "and put it into .env (see .env.example)."
            )
        return super()._headers()


class DiscoveringProvider(OpenAICompatProvider):
    """Provider whose model list changes often (NVIDIA, Gemini): finds a working model by itself.

    NVIDIA removes models from the free endpoint often (HTTP 410 "end of life"), so the
    default model is "auto": the first request probes the models of /v1/models and picks
    one that answers and supports tool calling. If the current model turns 410 mid-session,
    the provider switches to another working one and retries.
    """
    context_chars = 150_000

    AUTO = "auto"
    DEFAULT_MODEL = "auto"          # "auto" = probe on first use; any other id = try it first
    key_url = ""
    PROBE_OK = (200, 429)           # 429 = model exists, just rate limited
    # earlier = preferred; matched as substrings of the model id
    PREFER = ("kimi-k2", "deepseek", "glm", "qwen3", "qwen", "gpt-oss", "mistral",
              "minimax", "nemotron", "llama", "gemma")
    SKIP = re.compile(r"embed|rerank|guard|safety|gliner|parse|clip|vl-|-vl|vision|"
                      r"diffusion|multimodal|whisper|asr|tts|riva|nvclip|reward|retriev|"
                      r"stable-diffusion|flux|cosmos|segment|ocr|translate|pii", re.I)
    MAX_PROBES = 20

    def __init__(self, model: str | None = None, **kw):
        super().__init__(model or self.DEFAULT_MODEL, **kw)
        self._pinned = bool(model) and model != self.AUTO      # user chose the model: never swap it silently on timeouts
        self._dead: set[str] = set()

    def _headers(self) -> dict:
        if not self.api_key:
            raise ProviderError(
                f"{self.env_key} is not set. Get a key at {self.key_url} "
                "and put it into .env (see .env.example)."
            )
        return super()._headers()

    # -- model discovery ---------------------------------------------------- #
    def _rank(self, m: str) -> int:
        for i, key in enumerate(self.PREFER):
            if key in m.lower():
                return i
        return len(self.PREFER)

    def _candidates(self) -> list[str]:
        ids = [m for m in self.list_models() if m not in self._dead and not self.SKIP.search(m)]
        return sorted(ids, key=lambda m: (self._rank(m), m))

    def _probe(self, model: str, timeout: float = 20) -> bool:
        """True when the model answers a 1-token request that carries a tool definition."""
        return self._probe_detail(model, timeout)[0]

    def probe_all(self, limit: int = 30, timeout: float = 30) -> list[tuple[str, str, float, bool]]:
        """Time a tiny tool-carrying request against many models in parallel.
        Returns [(model, status, seconds, ok)], working ones first, fastest first."""
        from concurrent.futures import ThreadPoolExecutor
        ids = self._candidates()[:limit]
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(lambda m: self._row(m, timeout), ids))
        return sorted(rows, key=lambda r: (not r[3], r[2]))

    def _row(self, model: str, timeout: float) -> tuple[str, str, float, bool]:
        ok, status, secs = self._probe_detail(model, timeout)
        return (model, status, secs, ok)

    def _probe_detail(self, model: str, timeout: float = 20) -> tuple[bool, str, float]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "tools": [{"type": "function", "function": {
                "name": "noop", "description": "noop",
                "parameters": {"type": "object", "properties": {}}}}],
        }
        t0 = time.monotonic()
        try:
            r = requests.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                              json=payload, timeout=(10, timeout))
        except requests.Timeout:
            secs = time.monotonic() - t0
            _dbg(f"probe {model}: timeout after {secs:.1f}s", console=False)
            return False, "timeout", secs
        except requests.RequestException as exc:
            secs = time.monotonic() - t0
            _dbg(f"probe {model}: {type(exc).__name__} after {secs:.1f}s", console=False)
            return False, type(exc).__name__, secs
        secs = time.monotonic() - t0
        status = r.status_code
        _dbg(f"probe {model}: HTTP {status} in {secs:.1f}s body={r.text[:200]!r}", console=False)
        r.close()
        return status in self.PROBE_OK, str(status), secs

    FAST_ENOUGH = 10.0   # seconds a probe may take to still count as "fast"

    def find_working_model(self) -> str:
        """Probe candidates in parallel; take the most preferred one that answers fast."""
        _say(f"{self.name}: looking for a fast working model ...")
        rows = self.probe_all(limit=self.MAX_PROBES, timeout=15)
        good = [r for r in rows if r[3]]
        for model, status, secs, ok in rows:
            if not ok:
                self._dead.add(model)
        if not good:
            raise ProviderError(
                f"no working {self.name} chat model answered in time (rate limit, quota or "
                f"overload). Try again later, or run `zamagent -p {self.name} --probe`.")
        fast = [r for r in good if r[2] <= self.FAST_ENOUGH] or good

        best = min(fast, key=lambda r: (self._rank(r[0]), r[2]))
        self.model = best[0]
        _say(f"{self.name}: using {best[0]} ({best[2]:.1f}s probe)")
        return best[0]

    # -- chat with fallback ------------------------------------------------- #
    def chat(self, messages, tools, on_token=None) -> LLMResponse:
        if self.model == self.AUTO:
            self.find_working_model()
        try:
            return super().chat(messages, tools, on_token)
        except ProviderError as exc:
            gone = exc.status in (404, 410) and not isinstance(exc, ToolUseFailed)
            slow = isinstance(exc, ProviderTimeout) and not self._pinned
            if not (gone or slow):
                raise
            why = f"is gone ({exc.status})" if gone else "is not answering"
            _say(f"{self.name}: {self.model} {why}, looking for another model")
            self._dead.add(self.model)
            self.find_working_model()
            return super().chat(messages, tools, on_token)


class NvidiaProvider(DiscoveringProvider):
    """NVIDIA NIM API (https://build.nvidia.com)."""
    name = "nvidia"
    default_base_url = "https://integrate.api.nvidia.com/v1"
    env_key = "NVIDIA_API_KEY"
    env_base_url = "NVIDIA_BASE_URL"
    key_hint = "NVIDIA keys start with 'nvapi-'."
    key_url = "https://build.nvidia.com/settings/api-keys"


def _strip_schema_keys(node, drop=("additionalProperties", "$schema")):
    if isinstance(node, dict):
        return {k: _strip_schema_keys(v, drop) for k, v in node.items() if k not in drop}
    if isinstance(node, list):
        return [_strip_schema_keys(v, drop) for v in node]
    return node


class GeminiProvider(DiscoveringProvider):
    """Google Gemini through its OpenAI-compatible endpoint. Key: https://aistudio.google.com/api-keys

    Notes: free-tier requests may be used by Google to improve its products (paid tier: not).
    Model ids come back as "models/<id>"; Gemini 3 needs "thought signatures" echoed back
    (handled via extra_content on tool calls).
    """
    name = "gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    env_key = "GEMINI_API_KEY"
    env_base_url = "GEMINI_BASE_URL"
    key_hint = "Create a key at https://aistudio.google.com/api-keys."
    key_url = "https://aistudio.google.com/api-keys"
    DEFAULT_MODEL = "gemini-2.5-flash"
    PROBE_OK = (200,)   # on the free tier a paid-only model answers 429 with "limit: 0"
    tool_result_limit = 12_000
    context_chars = 300_000
    PREFER = ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-3", "flash", "pro", "gemini")
    SKIP = re.compile(r"embed|tts|image|live|audio|imagen|veo|aqa|learnlm|gemma|robotics|"
                      r"computer-use|deep-research|gemini-1\.|gemini-2\.0|nano-banana", re.I)

    def __init__(self, model: str | None = None, **kw):
        kw.setdefault("api_key", os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "")
        super().__init__(model, **kw)

    def _rank(self, m: str) -> int:
        penalty = 3 if re.search(r"lite|preview|exp|latest", m, re.I) else 0
        return super()._rank(m) * 10 + penalty

    def list_models(self) -> list[str]:
        return sorted({i.removeprefix("models/") for i in super().list_models()})

    def prepare_tools(self, tools: list[dict]) -> list[dict]:
        return _strip_schema_keys(tools)


# --------------------------------------------------------------------------- #
# Ollama (native API)                                                          #
# --------------------------------------------------------------------------- #

class OllamaProvider(Provider):
    name = "ollama"
    tool_result_limit = 3000
    context_chars = 40_000

    def __init__(self, model: str, host: str | None = None, temperature: float = 0.2):
        super().__init__(model)
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
        self.temperature = temperature

    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            role = m.get("role")
            if role == "assistant" and m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    fn = tc["function"]
                    calls.append({"function": {
                        "name": fn["name"],
                        "arguments": _parse_args(fn.get("arguments")) or {},
                    }})
                out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
            elif role == "tool":
                out.append({"role": "tool", "tool_name": m.get("_name", ""), "content": m.get("content", "")})
            else:
                out.append({"role": role, "content": m.get("content") or ""})
        return out

    def chat(self, messages, tools, on_token=None) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": self._convert(messages),
            "stream": True,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools
        if "qwen3" in self.model:
            payload["think"] = False

        try:
            resp = requests.post(f"{self.host}/api/chat", json=payload, stream=True, timeout=(5, None))
        except requests.ConnectionError:
            raise ProviderError(f"Ollama is not reachable at {self.host}. Start it with `ollama serve`.")
        if resp.status_code != 200:
            text = resp.text[:300]
            resp.close()
            hint = " Pick a model with tool support (e.g. qwen3, llama3.1)." if "tools" in text else ""
            raise ProviderError(f"{resp.status_code}: {text}{hint}", resp.status_code)

        parts: list[str] = []
        calls: list[dict] = []
        usage: dict = {}
        try:
            for raw in resp.iter_lines():
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise ProviderError(str(chunk["error"]))
                msg = chunk.get("message") or {}
                if msg.get("content"):
                    parts.append(msg["content"])
                    if on_token:
                        on_token(msg["content"])
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    args_dict = _parse_args(args)
                    calls.append({
                        "id": _new_id(),
                        "name": fn.get("name", ""),
                        "arguments": args_dict,
                        "raw": args if isinstance(args, str) else json.dumps(args or {}),
                    })
                if chunk.get("done"):
                    usage = {"prompt_tokens": chunk.get("prompt_eval_count", 0),
                             "completion_tokens": chunk.get("eval_count", 0)}
                    break
        finally:
            resp.close()
        return LLMResponse(_strip_think("".join(parts)), [c for c in calls if c["name"]], usage)

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"cannot list models: {exc}")
        return sorted(m["name"] for m in r.json().get("models", []))


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #

PROVIDERS = ("groq", "nvidia", "gemini", "ollama", "openai")


def default_provider_name() -> str:
    load_env()
    if os.getenv("AGENT_PROVIDER"):
        return os.environ["AGENT_PROVIDER"].lower()
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("NVIDIA_API_KEY"):
        return "nvidia"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "ollama"


def make_provider(name: str | None = None, model: str | None = None) -> Provider:
    load_env()
    name = (name or default_provider_name()).lower()
    model = resolve_model(name, model)
    if name == "groq":
        return GroqProvider(model or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b")
    if name == "nvidia":
        return NvidiaProvider(model or os.getenv("NVIDIA_MODEL") or None)
    if name == "gemini":
        return GeminiProvider(model or os.getenv("GEMINI_MODEL") or None)
    if name == "ollama":
        return OllamaProvider(model or os.getenv("OLLAMA_MODEL") or "qwen3:8b")
    if name in ("openai", "custom"):
        return OpenAICompatProvider(model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini")
    raise ProviderError(f"unknown provider {name!r}. Available: {', '.join(PROVIDERS)}")