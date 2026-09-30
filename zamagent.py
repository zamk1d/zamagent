try:  # proper line editing for input(): UTF-8 backspace, arrows, history
    import readline  # noqa: F401
except ImportError:  # e.g. Windows
    pass

import argparse
import difflib
import os
import sys
import time
from datetime import datetime

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

import providers
from agent import Agent, WORK_DIR, DANGEROUS
from providers import ProviderError, make_provider, resolve_model, PROVIDERS
from tools_inspector import get_tool_schemas

console = Console(highlight=False)

SESSIONS_DIR = os.path.join(WORK_DIR, ".zamagent", "sessions")

HELP = """\
[bold]Commands[/bold]
  [cyan]/model[/cyan] [dim]<name|alias>[/dim]      switch model (groq aliases: smart, oss-small, llama, fast)
  [cyan]/provider[/cyan] [dim]<groq|ollama|openai> [model][/dim]
  [cyan]/models[/cyan]                 list models of the current provider
  [cyan]/yes[/cyan]                    toggle auto-approve for shell / run / delete / move
  [cyan]/tools[/cyan]                  list available tools
  [cyan]/usage[/cyan]                  token usage of this session
  [cyan]/clear[/cyan]                  forget the conversation
  [cyan]/save[/cyan] [dim][name][/dim]   /  [cyan]/load[/cyan] [dim]<name>[/dim]   save / restore a conversation
  [cyan]/help[/cyan]   [cyan]/exit[/cyan]"""


# --------------------------------------------------------------------------- #
# Live output                                                                  #
# --------------------------------------------------------------------------- #

_status = None
_streamed = False
_line_open = False        # True while a streamed line has no trailing newline


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _start_status(label="thinking"):
    global _status
    _stop_status()
    _status = console.status(f"[cyan]{label}[/cyan]", spinner="dots")
    _status.start()


def _stop_status():
    global _status
    if _status:
        _status.stop()
        _status = None


def _end_line():
    global _line_open
    if _line_open:
        console.print()
        _line_open = False


def _on_step(step: int):
    global _streamed
    _streamed = False
    _end_line()
    if step:
        t = Text()
        t.append(f" {_ts()} ", style="dim")
        t.append(f" step {step + 1} ", style="bold black on bright_black")
        console.print(t)
    _start_status()


def _on_token(token: str):
    global _streamed, _line_open
    _stop_status()
    if not _streamed:
        _streamed = True
        console.print("\n[bold green]agent[/bold green]  ", end="")
    _line_open = not token.endswith("\n")
    console.out(token, end="", highlight=False)


def _fmt_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v).replace("\n", "\\n")
        parts.append(f"{k}={(s[:57] + '…') if len(s) > 60 else s!r}")
    return "  ".join(parts)


def _short(value) -> str:
    if isinstance(value, list):
        preview = ", ".join(str(x) for x in value[:5])
        return f"[{preview}{f'  +{len(value) - 5} more' if len(value) > 5 else ''}]"
    if isinstance(value, dict):
        if "exit_code" in value:
            out = (value.get("stdout") or value.get("stderr") or "").strip().replace("\n", " ⏎ ")
            return f"exit {value['exit_code']}  {out[:90]}"
        return str(value)[:100]
    s = str(value).replace("\n", " ⏎ ")
    return s[:100] + "…" if len(s) > 100 else s


def _diff(old: str, new: str) -> Syntax:
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), "old", "new", lineterm="", n=1))
    body = "\n".join(lines[:24] + (["…"] if len(lines) > 24 else []))
    return Syntax(body, "diff", theme="ansi_dark", background_color="default")


def _on_tool_call(tool: str, args: dict):
    _stop_status()
    _end_line()
    t = Text()
    t.append(f" {_ts()} ", style="dim")
    t.append(" call ", style="bold black on yellow")
    t.append("  ")
    t.append(tool, style="yellow")
    if tool == "write_file" and "content" in args:
        t.append(f"  {args.get('filepath')!r}  ({len(str(args['content']).splitlines())} lines)", style="dim")
    elif tool == "edit_file":
        t.append(f"  {args.get('filepath')!r}", style="dim")
    else:
        t.append(f"  {_fmt_args(args)}", style="dim")
    console.print(t)
    if tool == "edit_file" and "old_str" in args:
        console.print(_diff(str(args["old_str"]), str(args.get("new_str", ""))))


def _on_tool_result(tool: str, result: dict):
    t = Text()
    t.append(f" {_ts()} ", style="dim")
    payload = result.get("result", result)
    if result.get("status") == "ok":
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
        t.append(_short(payload), style="dim red")
    console.print(t)


def _confirm(tool: str, args: dict) -> str:
    _stop_status()
    detail = args.get("command") or " -> ".join(str(args[k]) for k in ("filepath", "src", "dst") if k in args)
    if tool == "run_python":
        detail = f"python {args.get('filepath')} {args.get('args', '')}".strip()
    console.print(Panel(Text(detail), title=f"[yellow]{tool}[/yellow]", title_align="left",
                        border_style="yellow", box=box.ROUNDED, padding=(0, 1)))
    return Prompt.ask("[yellow]allow?[/yellow] [dim](a = always for this tool)[/dim]",
                      choices=["y", "n", "a"], default="y", console=console)


def _print_panel(text: str, title: str, color: str):
    console.print()
    console.print(Panel(text, title=f"[bold {color}]{title}[/bold {color}]", title_align="left",
                        border_style=color, box=box.ROUNDED, padding=(0, 1)))
    console.print()


providers.notice = lambda msg: (_stop_status(), console.print(f" [dim]{_ts()}[/dim] [magenta]{msg}[/magenta]"))


# --------------------------------------------------------------------------- #
# Running                                                                      #
# --------------------------------------------------------------------------- #

def _tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _run(agent: Agent, prompt: str) -> bool:
    before = dict(agent.usage)
    t0 = time.time()
    ok = True
    try:
        result = agent.run(prompt)
        _end_line()
        if not _streamed and result:
            _print_panel(result, "agent", "green")
    except KeyboardInterrupt:
        _end_line()
        console.print("\n[dim]interrupted (this turn was discarded)[/dim]")
        ok = False
    except ProviderError as exc:
        _end_line()
        _print_panel(str(exc), "error", "red")
        ok = False
    except Exception as exc:
        _end_line()
        _print_panel(f"{type(exc).__name__}: {exc}", "error", "red")
        ok = False
    finally:
        _stop_status()

    used = agent.usage["prompt"] + agent.usage["completion"] - before["prompt"] - before["completion"]
    console.print(f"\n[dim]{time.time() - t0:.1f}s · {agent.provider.label}"
                  f"{f' · {_tokens(used)} tok' if used else ''}[/dim]")
    console.print(Rule(style="dim"))
    console.print()
    return ok


# --------------------------------------------------------------------------- #
# Slash commands                                                               #
# --------------------------------------------------------------------------- #

def _command(agent: Agent, line: str) -> bool:
    """Returns True when the line was a slash command."""
    if not line.startswith("/"):
        return False
    cmd, _, rest = line[1:].partition(" ")
    rest = rest.strip()

    try:
        if cmd in ("exit", "quit", "q"):
            raise EOFError
        elif cmd == "help":
            console.print(HELP)
        elif cmd == "model":
            if rest:
                agent.provider.model = resolve_model(agent.provider.name, rest)
            console.print(f"model: [bold]{agent.provider.label}[/bold]")
        elif cmd == "provider":
            name, _, model = rest.partition(" ")
            if not name:
                console.print(f"provider: [bold]{agent.provider.label}[/bold]   available: {', '.join(PROVIDERS)}")
            else:
                agent.provider = make_provider(name, model.strip() or None)
                console.print(f"switched to [bold]{agent.provider.label}[/bold] (conversation kept)")
        elif cmd == "models":
            for m in agent.provider.list_models():
                mark = "[green]●[/green]" if m == agent.provider.model else " "
                console.print(f" {mark} {m}")
        elif cmd == "yes":
            agent.auto_approve = not agent.auto_approve
            console.print(f"auto-approve: [bold]{'ON' if agent.auto_approve else 'off'}[/bold]")
        elif cmd == "tools":
            for t in get_tool_schemas():
                f = t["function"]
                mark = "[yellow]![/yellow]" if f["name"] in DANGEROUS else " "
                console.print(f" {mark} [cyan]{f['name']}[/cyan]  [dim]{f['description'][:80]}[/dim]")
        elif cmd == "usage":
            u = agent.usage
            console.print(f"{u['calls']} model calls · {_tokens(u['prompt'])} prompt · "
                          f"{_tokens(u['completion'])} completion tokens · history {len(agent.messages)} messages")
        elif cmd == "clear":
            agent.reset()
            console.print("[dim]conversation cleared[/dim]")
        elif cmd == "save":
            name = rest or datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(SESSIONS_DIR, f"{name}.json")
            agent.save(path)
            console.print(f"saved → {os.path.relpath(path)}")
        elif cmd == "load":
            path = os.path.join(SESSIONS_DIR, f"{rest}.json")
            if not rest or not os.path.isfile(path):
                names = [f[:-5] for f in os.listdir(SESSIONS_DIR)] if os.path.isdir(SESSIONS_DIR) else []
                console.print(f"[red]no such session.[/red] available: {', '.join(names) or '(none)'}")
            else:
                agent.load(path)
                console.print(f"loaded {rest} ({len(agent.messages)} messages)")
        else:
            console.print(f"[red]unknown command[/red] /{cmd}  (try /help)")
    except ProviderError as exc:
        console.print(f"[red]{exc}[/red]")
    return True


# --------------------------------------------------------------------------- #
# Entry                                                                        #
# --------------------------------------------------------------------------- #

def _header(agent: Agent):
    mode = "auto-approve" if agent.auto_approve else "asks before shell/delete/move"
    console.print()
    console.print(Panel(
        f"[bold cyan]zamagent[/bold cyan]  [dim]{agent.provider.label}[/dim]\n"
        f"[dim]workspace[/dim] {WORK_DIR}\n"
        f"[dim]safety[/dim]    {mode}\n"
        f"[dim]/help for commands · Ctrl+C interrupts · exit quits[/dim]",
        border_style="cyan", box=box.ROUNDED, padding=(0, 1)))
    console.print()


def repl(agent: Agent):
    _header(agent)
    while True:
        try:
            line = Prompt.ask("[bold cyan]you[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]goodbye[/dim]")
            return
        except UnicodeDecodeError:
            console.print("[red]the terminal sent broken characters - type the message again[/red]")
            continue
        if not line:
            continue
        if line.lower() in ("exit", "quit", "q"):
            console.print("[dim]goodbye[/dim]")
            return
        try:
            if _command(agent, line):
                continue
        except EOFError:
            console.print("[dim]goodbye[/dim]")
            return
        console.print()
        _run(agent, line)


def main():
    parser = argparse.ArgumentParser(description="zamagent - your own terminal agent",
                                     epilog="No prompt -> interactive REPL. With a prompt -> run once and exit.")
    parser.add_argument("prompt", nargs="?", help="prompt for a one-shot run")
    parser.add_argument("-p", "--provider", choices=PROVIDERS, help="groq | ollama | openai (default: groq if GROQ_API_KEY is set, else ollama)")
    parser.add_argument("-m", "--model", help="model id or alias (groq: smart, oss-small, llama, fast)")
    parser.add_argument("-y", "--yes", action="store_true", help="auto-approve shell/run/delete/move")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--list-models", action="store_true", help="print models of the provider and exit")
    args = parser.parse_args()

    try:
        provider = make_provider(args.provider, args.model)
        if args.list_models:
            print("\n".join(provider.list_models()))
            return
    except ProviderError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    agent = Agent(provider, auto_approve=args.yes, max_steps=args.max_steps)
    agent.on_step, agent.on_token = _on_step, _on_token
    agent.on_tool_call, agent.on_tool_result, agent.confirm = _on_tool_call, _on_tool_result, _confirm

    if args.prompt:
        console.print(f"\n[bold cyan]you ›[/bold cyan] {args.prompt}\n")
        sys.exit(0 if _run(agent, args.prompt) else 1)
    repl(agent)


if __name__ == "__main__":
    main()