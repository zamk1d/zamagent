"""
LLM providers for zamagent.

* GroqProvider    - Groq Cloud (OpenAI-compatible, native tool calling, SSE streaming)
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
    "llama": "llama-3.3-70b-versatile",
    "fast": "llama-3.1-8b-instant",
}


def resolve_model(provider: str, model: str | None) -> str | None:
    if model and provider == "groq":
        return GROQ_ALIASES.get(model.lower(), model)
    return model


# --------------------------------------------------------------------------- #
# Types                                                                        #
# --------------------------------------------------------------------------- #

class ProviderError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


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

class OpenAICompatProvider(Provider):
    name = "openai"
    default_base_url = "https://api.openai.com/v1"
    env_key = "OPENAI_API_KEY"
    env_base_url = "OPENAI_BASE_URL"

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
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(url, headers=self._headers(), json=payload,
                                     stream=True, timeout=(10, 180))
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == MAX_RETRIES:
                    raise ProviderError(f"connection failed: {exc}")
                _say(f"connection problem, retry {attempt}/{MAX_RETRIES}")
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code == 200:
                return resp

            message, code, failed = _error_info(resp)
            status = resp.status_code
            resp.close()

            if code == "tool_use_failed":
                raise ToolUseFailed(message, failed)

            if status == 401:
                key = self.api_key
                src = ENV_SOURCES.get(self.env_key, "the shell environment (not .env!)")
                raise ProviderError(
                    f"401 {message}. {self.env_key} was taken from {src}: {len(key)} chars, "
                    f"starts with {key[:4]!r}, ends with {key[-2:]!r}. Groq keys start with 'gsk_'. "
                    f"If it comes from the shell, remove it (bash: unset {self.env_key} / "
                    f"fish: set -e {self.env_key}) so .env is used.", 401)

            if status in (429, 500, 502, 503, 504):
                try:
                    wait = float(resp.headers.get("retry-after", ""))
                except ValueError:
                    wait = delay
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
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        payload.update(self.extra_params())

        resp = self._post(payload)
        parts: list[str] = []
        calls: dict[int, dict] = {}
        usage: dict = {}
        try:
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", "replace")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if chunk.get("error"):
                    err = chunk["error"]
                    if isinstance(err, dict) and err.get("code") == "tool_use_failed":
                        raise ToolUseFailed(err.get("message", ""), err.get("failed_generation", ""))
                    raise ProviderError(str(err.get("message", err)) if isinstance(err, dict) else str(err))

                u = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")
                if u:
                    usage = u

                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        parts.append(text)
                        if on_token:
                            on_token(text)
                    for tc in delta.get("tool_calls") or []:
                        entry = calls.setdefault(
                            tc.get("index", 0), {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            entry["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name") and not entry["name"]:
                            entry["name"] = fn["name"]
                        piece = fn.get("arguments")
                        if isinstance(piece, dict):
                            piece = json.dumps(piece)
                        if piece:
                            entry["args"] += piece
        finally:
            resp.close()

        tool_calls = []
        for idx in sorted(calls):
            c = calls[idx]
            if not c["name"]:
                continue
            tool_calls.append({
                "id": c["id"] or _new_id(),
                "name": c["name"],
                "arguments": _parse_args(c["args"]),
                "raw": c["args"],
            })
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

PROVIDERS = ("groq", "ollama", "openai")


def default_provider_name() -> str:
    load_env()
    return (os.getenv("AGENT_PROVIDER") or ("groq" if os.getenv("GROQ_API_KEY") else "ollama")).lower()


def make_provider(name: str | None = None, model: str | None = None) -> Provider:
    load_env()
    name = (name or default_provider_name()).lower()
    model = resolve_model(name, model)
    if name == "groq":
        return GroqProvider(model or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b")
    if name == "ollama":
        return OllamaProvider(model or os.getenv("OLLAMA_MODEL") or "qwen3:8b")
    if name in ("openai", "custom"):
        return OpenAICompatProvider(model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini")
    raise ProviderError(f"unknown provider {name!r}. Available: {', '.join(PROVIDERS)}")