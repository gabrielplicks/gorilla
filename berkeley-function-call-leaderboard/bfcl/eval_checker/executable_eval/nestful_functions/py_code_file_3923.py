from typing import List



def split_and_remove_empty(string: str) -> List[str]:

    """Splits a string on spaces and removes empty tokens.



    Args:

        string: The input string.



    Returns:

        A list of tokens produced by splitting the string on spaces and removing empty tokens.

    """

    tokens = string.split(" ")

    return [token for token in tokens if token != ""]

