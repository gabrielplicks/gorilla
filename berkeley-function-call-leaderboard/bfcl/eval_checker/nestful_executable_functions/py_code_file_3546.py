from typing import List



def find_substrings(string: str, n: int) -> List[str]:

    """Returns all substrings of length `n` from a given input `string`.



    Args:

        string: The input string.

        n: The length of the substrings.

    """

    return [string[i:i+n] for i in range(len(string)-n+1)]

