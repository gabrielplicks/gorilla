from typing import List



def sort_by_len_and_lex(list_of_strings: List[str]) -> List[str]:

    """Sorts a list of strings by length in ascending order, and for strings of equal length, sorts them lexicographically.



    Args:

        list_of_strings: The list of strings to be sorted.



    Returns:

        The sorted list of strings.

    """

    sorted_list = sorted(list_of_strings, key=lambda x: (len(x), x))

    return sorted_list

