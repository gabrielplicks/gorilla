from typing import List



def iterate_and_concatenate(L: List[str], s: str) -> str:

    """Iterates through a list of strings and concatenates each string to a variable `result` with a space between each string.

    If the current string in the list exactly matches the string `s`, returns the resulting variable `result`.

    Args:

        L: A list of strings.

        s: A string to match.

    """

    result = ""

    for item in L:

        result += item + " "

        if item == s:

            return result



    return result

