from typing import List



def ascii_values(words: List[str]) -> List[List[int]]:

    """

    Returns a list of lists, where each inner list contains the ASCII values of the characters in the corresponding word.



    Args:

        words: A list of strings.



    Returns:

        A list of lists, where each inner list contains the ASCII values of the characters in the corresponding word.

    """

    return [[ord(c) for c in word] for word in words]

