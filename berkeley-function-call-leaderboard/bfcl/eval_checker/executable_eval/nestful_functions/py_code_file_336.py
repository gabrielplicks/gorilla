from typing import List



def rearrange_words(string: str) -> str:

    """Rearranges the characters of each word in the input string in such a way that the first character of each word is capitalized, and all other characters are in lowercase.



    Args:

        string: The input string.



    Returns:

        The rearranged string.

    """

    words: List[str] = string.split()

    rearranged_words: List[str] = [word.capitalize().lower() for word in words]

    return ' '.join(rearranged_words)

