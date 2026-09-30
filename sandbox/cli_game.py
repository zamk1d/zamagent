"""Simple CLI Number Guessing Game.

Run the script directly:
    python sandbox/cli_game.py

The game picks a random integer between 1 and 100 (inclusive).
You have to guess it; after each guess the program tells you
whether the secret number is higher or lower. The game ends when
you guess correctly, and the number of attempts is shown.
"""

import random
import sys

def main() -> None:
    secret = random.randint(1, 100)
    attempts = 0
    print("Welcome to the Number Guessing Game!")
    print("I have selected a number between 1 and 100.")
    while True:
        attempts += 1
        try:
            guess_str = input(f"Attempt #{attempts}: Your guess? ")
            guess = int(guess_str)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if guess < secret:
            print("Higher!")
        elif guess > secret:
            print("Lower!")
        else:
            print(f"Congratulations! You guessed the number {secret} in {attempts} attempts.")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nGame interrupted. Bye!")
