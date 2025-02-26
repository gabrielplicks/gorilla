from typing import Dict, List



def count_char(strings: List[str], char: str) -> Dict[str, int]:

    """

    Returns a dictionary with the strings as keys and their counts of the specified character as values.



    Args:

        strings: A list of strings.

        char: The character to count.

    """

    return {s: s.count(char) for s in strings}

