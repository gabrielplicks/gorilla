from typing import Dict, List



def string_index_map(strings: List[str]) -> Dict[str, List[int]]:

    """Creates a dictionary that maps each unique string to a list of its indices in the input list.



    Args:

        strings: A list of strings.



    Returns:

        A dictionary that maps each unique string to a list of its indices in the input list.

    """

    index_map = {}

    for index, string in enumerate(strings):

        if string not in index_map:

            index_map[string] = [index]

        else:

            index_map[string].append(index)

    return index_map

