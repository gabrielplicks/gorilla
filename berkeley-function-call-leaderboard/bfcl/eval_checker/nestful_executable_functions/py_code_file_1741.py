from typing import Iterable



def print_pattern(string: str) -> None:

    """Prints a string in a pattern that repeats each character a number of times equal to its position in the string.



    Args:

        string: The input string.

    """

    for index, character in enumerate(string):

        for _ in range(index + 1):

            print(character, end='')

        print()

