from typing import List



def map_index(list_of_strings: List[str]) -> dict:

    """Creates a dictionary that maps each value to its index.



    Args:

        list_of_strings: A non-empty list of strings.



    Returns:

        A dictionary that maps each string to its index in the list.

    """

    return {string: idx for idx, string in enumerate(list_of_strings)}

