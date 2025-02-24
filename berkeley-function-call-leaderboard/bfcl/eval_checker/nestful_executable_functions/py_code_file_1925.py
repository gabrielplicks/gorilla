from typing import List



def ascii_to_integer_list(s: str) -> List[int]:

    """Converts each character of the input string to its corresponding integer value based on the ASCII table.



    Args:

        s: The input string.



    Returns:

        A list of integers representing the ASCII values of the input string's characters.

    """

    return [ord(c) for c in s]

