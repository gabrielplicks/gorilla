from typing import List



def find_prefixes(text: str) -> List[str]:

    """Finds all possible prefixes of the input string.



    Args:

        text: The input string.



    Returns:

        A list of all possible prefixes of the input string.

    """

    prefixes = []



    for i in range(len(text)):

        prefixes.append(text[:i+1])



    return prefixes

