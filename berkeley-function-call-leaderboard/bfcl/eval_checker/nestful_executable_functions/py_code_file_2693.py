from typing import List



def collapse_duplicate_characters(string: str) -> str:

    """Collapses consecutive duplicate characters in a string.



    Args:

        string: The input string.



    Returns:

        A new string with all consecutive duplicate characters removed.

    """

    result: List[str] = [string[0]]

    for i in range(1, len(string)):

        if string[i] != result[-1]:

            result.append(string[i])

    return ''.join(result)

