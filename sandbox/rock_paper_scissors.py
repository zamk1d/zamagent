"""CLI Rock‑Paper‑Scissors game.

Run with:
    python sandbox/rock_paper_scissors.py

The player plays against the computer. The first to reach
`WIN_SCORE` wins the match. After each round the result and the
current score are shown.
"""

import random
import sys

WIN_SCORE = 3
OPTIONS = ["rock", "paper", "scissors"]

RULES = {
    ("rock", "scissors"): "Rock crushes Scissors",
    ("scissors", "paper"): "Scissors cut Paper",
    ("paper", "rock"): "Paper covers Rock",
}

def decide(player: str, computer: str) -> str:
    """Return 'win', 'lose' or 'draw' for the player.
    """
    if player == computer:
        return "draw"
    if (player, computer) in RULES:
        return "win"
    return "lose"

def main() -> None:
    print("Welcome to Rock‑Paper‑Scissors!")
    print(f"First to {WIN_SCORE} points wins the match.")
    player_score = 0
    computer_score = 0
    round_num = 0
    while player_score < WIN_SCORE and computer_score < WIN_SCORE:
        round_num += 1
        print(f"\nRound {round_num} – Score {player_score}:{computer_score}")
        player_choice = input("Choose rock, paper or scissors: ").strip().lower()
        if player_choice not in OPTIONS:
            print("Invalid choice, try again.")
            continue
        computer_choice = random.choice(OPTIONS)
        print(f"Computer chose {computer_choice}.")
        outcome = decide(player_choice, computer_choice)
        if outcome == "win":
            print(f"You win this round! {RULES[(player_choice, computer_choice)]}.")
            player_score += 1
        elif outcome == "lose":
            print(f"You lose this round. {RULES[(computer_choice, player_choice)]}.")
            computer_score += 1
        else:
            print("It's a draw.")
    print("\nGame over!")
    if player_score > computer_score:
        print("Congratulations, you won the match!")
    else:
        print("Computer wins the match. Better luck next time!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nGame interrupted. Bye!")
