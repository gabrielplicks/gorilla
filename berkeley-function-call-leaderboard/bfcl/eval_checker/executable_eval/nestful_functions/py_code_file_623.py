from typing import List



def reverse_string_to_list(string: str) -> List[str]:

    """Converts a string to a list of strings in a reverse manner, with each element of the list representing a letter of the string.



    Args:

        string: The input string.



    Returns:

        A list of strings representing the input string in reverse order.

    """

    return [chr(char) for char in reversed([ord(c) for c in string])]

