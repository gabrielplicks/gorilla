from typing import List



def parse_string_to_numbers(string: str) -> List[int]:

    """Parses a string and extracts all numbers, converting them to integers if possible.



    Args:

        string: The input string to parse.



    Returns:

        A list of integers extracted from the input string.

    """

    if not string or not any(char.isdigit() for char in string):

        return []

    characters = list(string)

    numbers = []

    for char in characters:

        if char.isdigit():

            numbers.append(int(char))



    return numbers

