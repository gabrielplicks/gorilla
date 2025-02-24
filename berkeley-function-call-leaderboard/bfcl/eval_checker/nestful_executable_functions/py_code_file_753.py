from typing import Optional



def count_unique_substrings(s: Optional[str], k: int) -> int:

    """Calculates the number of unique substrings of length `k` in a given string `s`.

    Also handles the case where the input string is `None`, `s` is empty, or `k` is greater than the length of the input string.

    Args:

        s: The input string.

        k: The length of the substrings to be considered.

    """

    if s is None or len(s) == 0:

        return 0



    if k > len(s):

        return 0



    unique_substrings = set()



    for i in range(len(s) - k + 1):

        unique_substrings.add(s[i:i+k])



    return len(unique_substrings)

