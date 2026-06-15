import argparse
from rich.console import Console
from rich.panel import Panel
from rich import box

from agent import run

console = Console()


def repl():
    console.print("[bold cyan]Agent REPL started (type 'exit' to quit)[/bold cyan]\n")

    while True:
        try:
            user = input("> ").strip()

            if user.lower() in ("exit", "quit"):
                break

            if not user:
                continue

            with console.status("[bold green]thinking..."):
                result = run(user)

            console.print(
                Panel(
                    result,
                    title="Answer",
                    border_style="green",
                    box=box.ROUNDED
                )
            )

        except KeyboardInterrupt:
            console.print("\n[red]Interrupted[/red]")
            break


def run_once(prompt: str):
    with console.status("[bold green]running agent..."):
        result = run(prompt)

    console.print(
        Panel(
            result,
            title="Answer",
            border_style="green",
            box=box.ROUNDED
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Agent CLI")
    parser.add_argument("prompt", nargs="?", help="User prompt")
    args = parser.parse_args()

    if args.prompt:
        run_once(args.prompt)
    else:
        repl()


if __name__ == "__main__":
    main()