from typing import Tuple



def reverse_and_length(input_string: str) -> Tuple[str, int]:

    """Reverses a string and returns its length.



    Args:

        input_string: The input string to reverse and get the length of.



    Returns:

        A tuple containing the reversed string and its length.

    """

    reversed_string = input_string[::-1]

    length = len(input_string)

    return reversed_string, length

