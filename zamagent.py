import argparse
import sys
import json
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.prompt import Prompt
from rich import box

import agent as _agent
from agent import run

console = Console(highlight=False)

# Буфер токенов — накапливаем стрим и печатаем инкрементально
_stream_started = False
_stream_buf: list[str] = []

LOGO = (
    "╭─────────────────────────────╮\n"
    "│  [bold cyan]agent[/bold cyan]  ·  [dim]type [italic]exit[/italic] to quit[/dim]│\n"
    "╰─────────────────────────────╯"
)


# --------------------------------------------------------------------------- #
# Wiring — подключаем callbacks агента к выводу rich                          #
# --------------------------------------------------------------------------- #

def _on_token(token: str):
    """
    Вызывается на каждый токен во время генерации.
    Если это начало ответа — печатаем префикс, потом льём токены прямо в терминал.
    Если модель генерирует JSON (tool call) — не показываем мусор пользователю.
    """
    global _stream_started, _stream_buf
    _stream_buf.append(token)

    # Пока не набрали 10 символов — не знаем JSON это или текст
    combined = "".join(_stream_buf)
    if len(combined) < 10:
        return

    if combined.lstrip().startswith("["):
        # Это tool call — молчим, покажем лог после через on_tool_call
        return

    # Это текстовый ответ — включаем стрим
    if not _stream_started:
        _stream_started = True
        console.print()
        # Печатаем префикс answer один раз
        console.print("[bold green]answer[/bold green]  ", end="")
        # Сбрасываем буфер накопленного
        console.print(combined, end="", highlight=False)
    else:
        console.print(token, end="", highlight=False)


def _reset_stream():
    global _stream_started, _stream_buf
    if _stream_started:
        console.print()   # перенос строки после последнего токена
    _stream_started = False
    _stream_buf = []


def _on_tool_call(tool: str, arguments: dict):
    ts = _ts()
    t = Text()
    t.append(f" {ts} ", style="dim")
    t.append(" call ", style="bold black on yellow")
    t.append("  ")
    t.append(tool, style="yellow")
    args_short = _fmt_args(arguments)
    if args_short:
        t.append(f"  {args_short}", style="dim")
    console.print(t)


def _on_tool_result(tool: str, result: dict):
    ts = _ts()
    status = result.get("status", "?")
    payload = result.get("result", result)

    t = Text()
    t.append(f" {ts} ", style="dim")

    if status == "ok":
        t.append(" done ", style="bold black on green")
        t.append("  ")
        t.append(tool, style="green")
        t.append("  ")
        t.append(_short(payload), style="dim")
    else:
        t.append("  err ", style="bold black on red")
        t.append("  ")
        t.append(tool, style="red")
        t.append("  ")
        t.append(str(payload)[:120], style="dim red")

    console.print(t)


def _on_step(step: int):
    if step == 0:
        return
    t = Text()
    t.append(f" {_ts()} ", style="dim")
    t.append(f" step {step + 1} ", style="bold black on bright_black")
    console.print(t)


_agent.on_tool_call   = _on_tool_call
_agent.on_tool_result = _on_tool_result
_agent.on_step        = _on_step
_agent.on_token       = _on_token


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_args(args: dict) -> str:
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 60:
            v_str = v_str[:57] + "…"
        parts.append(f"{k}={v_str!r}")
    return "  ".join(parts)


def _short(value) -> str:
    if isinstance(value, list):
        preview = ", ".join(str(x) for x in value[:5])
        suffix = f"  +{len(value) - 5} more" if len(value) > 5 else ""
        return f"[{preview}{suffix}]"
    s = str(value)
    return s[:100] + "…" if len(s) > 100 else s


# --------------------------------------------------------------------------- #
# Print helpers                                                                 #
# --------------------------------------------------------------------------- #

def _print_header():
    console.print()
    console.print(LOGO)
    console.print()


def _print_answer(text: str):
    console.print()
    console.print(
        Panel(
            text,
            title="[bold green]answer[/bold green]",
            title_align="left",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )
    console.print()


def _print_error(text: str):
    console.print()
    console.print(
        Panel(
            text,
            title="[bold red]error[/bold red]",
            title_align="left",
            border_style="red",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )
    console.print()


def _print_separator():
    console.print(Rule(style="dim"))
    console.print()


def _print_thinking():
    t = Text()
    t.append(f" {_ts()} ", style="dim")
    t.append(" … ", style="bold black on cyan")
    t.append("  thinking", style="dim cyan")
    console.print(t)


# --------------------------------------------------------------------------- #
# REPL                                                                          #
# --------------------------------------------------------------------------- #

def repl():
    _print_header()

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]you[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]goodbye[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]goodbye[/dim]")
            break

        console.print()
        _print_thinking()

        try:
            result = run(user_input)
        except KeyboardInterrupt:
            _reset_stream()
            console.print("\n[dim]interrupted[/dim]")
            _print_separator()
            continue
        except Exception as exc:
            _reset_stream()
            _print_error(str(exc))
            _print_separator()
            continue

        _reset_stream()
        # Если on_token уже напечатал ответ в стриме — не дублируем в Panel.
        # Если ответ пришёл как tool call и потом финальный текст — он тоже
        # прошёл через on_token. Поэтому Panel показываем только когда стрим
        # не печатал (например модель вернула пустую строку или всё было JSON).
        if not _stream_buf and result:
            _print_answer(result)
        _print_separator()


# --------------------------------------------------------------------------- #
# One-shot                                                                      #
# --------------------------------------------------------------------------- #

def run_once(prompt: str):
    t = Text()
    t.append("you › ", style="bold cyan")
    t.append(prompt, style="white")
    console.print()
    console.print(t)
    console.print()
    _print_thinking()

    try:
        result = run(prompt)
    except Exception as exc:
        _reset_stream()
        _print_error(str(exc))
        sys.exit(1)

    _reset_stream()
    if not _stream_buf and result:
        _print_answer(result)


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="  No arguments  →  interactive REPL\n  PROMPT         →  run once and exit",
    )
    parser.add_argument("prompt", nargs="?", help="Prompt for the agent")
    args = parser.parse_args()

    if args.prompt:
        run_once(args.prompt)
    else:
        repl()


if __name__ == "__main__":
    main()