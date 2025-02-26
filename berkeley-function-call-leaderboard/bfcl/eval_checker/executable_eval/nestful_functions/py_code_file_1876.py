from typing import List



def map_to_next_in_alphabet(input_str: str) -> str:

    """Maps each character in a string to the next character in the alphabet.



    Args:

        input_str: The input string.



    Returns:

        A string where each alphabetic character is mapped to the next character in the alphabet.

    """

    result: List[str] = []

    for char in input_str:

        if char.isalpha():

            result.append(chr(ord(char) + 1))

        else:

            result.append(char)

    return ''.join(result)

