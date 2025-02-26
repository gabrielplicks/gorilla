from typing import List



def parse_ascii_art(input_string: str) -> None:

    """Parses and draws ASCII art from a given string.



    Args:

        input_string: A string representing a rectangular grid of characters.



    Returns:

        None

    """

    rows: List[str] = input_string.split('\n')

    for row in rows:

        characters: List[str] = list(row)

        for character in characters:

            if character == '*' or character == ' ':

                print(character, end='')

            else:

                return "Error: Invalid character in ASCII art"

