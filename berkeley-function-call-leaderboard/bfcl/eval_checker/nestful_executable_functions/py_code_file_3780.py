from typing import Optional



def get_first_k_characters(s: str, k: int) -> Optional[str]:

    """Returns the first `k` characters of the string `s`.



    Args:

        s: The input string.

        k: The number of characters to extract from the beginning of `s`.



    Returns:

        The first `k` characters of `s`, or `s` itself if its length is less than `k`.

    """

    if len(s) < k:

        return s

    else:

        return s[:k]

