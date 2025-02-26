from typing import List



def remove_substring_from_list(list_of_strings: List[str], search_string: str) -> List[str]:

    """Removes all occurrences of a search string from a list of strings using list comprehension.



    Args:

        list_of_strings: The list of strings to search.

        search_string: The string to search for and remove.



    Returns:

        The modified list of strings with all occurrences of the search string removed.

    """

    return [s for s in list_of_strings if search_string not in s]

