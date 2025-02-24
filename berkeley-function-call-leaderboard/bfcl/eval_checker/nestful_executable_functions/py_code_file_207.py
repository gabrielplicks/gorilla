from typing import List, Union



def cut_string(string: str, length: int) -> str:

    """Cuts a string at the nearest word boundary to the specified length.



    Args:

        string: The input string.

        length: The desired length of the output string.



    Returns:

        The input string, cut at the nearest word boundary to the specified length.

        If the length is less than or equal to 0, an empty string is returned.

    """

    if length <= 0:

        return ""

    words: List[str] = string.split()

    current_length: int = 0

    for i, word in enumerate(words):

        current_length += len(word) + 1  # Add one for the space

        if current_length > length:

            return " ".join(words[:i])

    return string

