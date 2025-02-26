from typing import List



def enumerate_suffixes(s: str) -> List[str]:

    """Returns a list of strings where each string is the original string with a suffix of the form <"@", index> appended, where index is a positive integer.



    Args:

        s: The input string.



    Returns:

        A list of strings.

    """

    return [s[i:] + f"@{i+1}" for i in range(len(s))]

