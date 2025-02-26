from typing import List



def convert_ascii_to_text(ascii_list: List[int]) -> str:

    """Converts a list of ASCII values to a string.



    Args:

        ascii_list: A list of ASCII values.



    Returns:

        A string formed by converting each ASCII value to its corresponding character and concatenating the characters.

    """

    return ''.join(map(chr, ascii_list))

