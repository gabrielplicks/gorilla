from typing import AnyStr



def check_adjacent_characters(text: AnyStr) -> bool:

    """Determines whether a given string contains two adjacent characters.



    Args:

        text: The input string.



    Raises:

        ValueError: If two adjacent characters are found.

    """

    result = ""



    for i in range(len(text) - 1):

        if text[i] == text[i + 1]:

            raise ValueError("Two adjacent characters found")

        else:

            result += text[i]



    return len(result) > 0

