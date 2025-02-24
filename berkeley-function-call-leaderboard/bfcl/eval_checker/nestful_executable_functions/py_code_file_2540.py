from typing import Union



def reverse_alphanumeric(s: str) -> str:

    """Reverses the order of alphanumeric characters in a string.

    Args:

        s: The input string.

    Returns:

        A new string with all the alphanumeric characters in the original string but in the reverse order.

        For any non-alphanumeric character, including spaces and special characters, they are copied to the new

        string and left in their original order.

    """

    new_string = ''

    for c in reversed(s):

        if c.isalnum():

            new_string += c

        else:

            new_string += c

    return new_string

