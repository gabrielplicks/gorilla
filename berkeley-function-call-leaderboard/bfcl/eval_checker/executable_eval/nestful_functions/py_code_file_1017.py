from typing import List



def replace_with_ascii(s: str) -> str:

    """Replaces all characters in a string with their ASCII value followed by a space.



    Args:

        s: The input string.



    Returns:

        A string where all characters are replaced by their ASCII value followed by a space.

    """

    result: List[str] = []

    for c in s:

        result.append(f"{ord(c)} ")

    return "".join(result)

