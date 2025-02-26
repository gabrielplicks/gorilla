from typing import List



def is_alphabetically_sorted(list_of_strings: List[str]) -> bool:

    """Checks if a list of strings is sorted in alphabetical order.

    Args:

        list_of_strings: A list of strings to check.

    Returns:

        True if the list is sorted in ascending order, False otherwise.

    """

    return sorted(list_of_strings) == list_of_strings

