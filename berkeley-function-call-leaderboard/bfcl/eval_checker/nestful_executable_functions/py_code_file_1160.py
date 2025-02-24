from typing import List



def concatenate_strings_with_index(strings: List[str]) -> List[str]:

    """Concatenates each string in the input list with its index.



    Args:

        strings: A list of strings.



    Returns:

        A new list with each string in the original list concatenated with its index.

    """

    return [s + str(i) for i, s in enumerate(strings)]

