from typing import List



def generate_all_strings(s: str) -> List[str]:

    """

    Generates all possible strings by removing a single character from `s`.



    Args:

        s: The input string.



    Returns:

        A list of all possible strings.

    """

    result = []

    for i in range(len(s)):

        result.append(s[:i] + s[i+1:])

    return result

