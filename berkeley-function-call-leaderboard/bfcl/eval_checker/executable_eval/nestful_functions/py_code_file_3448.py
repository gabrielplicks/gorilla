from typing import List, Union



def get_flattened_string(nested_list: List[Union[List[str], str]]) -> str:

    """Flattens a list of lists of strings into a single string.



    The function can handle an arbitrary number of nested lists and strings,

    including empty lists.



    Args:

        nested_list: The list of lists of strings to be flattened.



    Returns:

        The flattened string.

    """

    flattened_string = ''

    for item in nested_list:

        if isinstance(item, list):

            flattened_string += get_flattened_string(item)

        elif isinstance(item, str):

            flattened_string += item



    return flattened_string

