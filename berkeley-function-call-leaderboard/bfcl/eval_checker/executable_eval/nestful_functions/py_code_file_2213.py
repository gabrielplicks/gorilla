from typing import Optional



def find_indices_of_substring(base_str: str, sub_str: str) -> Optional[int]:

    """

    Searches for the first appearance of a given substring and returns the corresponding index within the base string.

    Returns -1 if the substring is not found.



    Args:

        base_str: The base string to search within.

        sub_str: The substring to search for.

    """

    index = base_str.find(sub_str)

    if index != -1:

        return index

    else:

        return -1

