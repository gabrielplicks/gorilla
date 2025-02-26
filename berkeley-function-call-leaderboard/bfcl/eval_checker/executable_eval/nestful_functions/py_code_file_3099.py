from typing import List, Union



def flatten_list_recursively(nested_list: List[Union[int, List[int]]]) -> List[int]:

    """Flattens a list with nested lists into a flat list using recursion.



    Args:

        nested_list: The list to be flattened.



    Returns:

        A flat list.

    """

    flattened_list = []

    for element in nested_list:

        if isinstance(element, list):

            flattened_list.extend(flatten_list_recursively(element))

        else:

            flattened_list.append(element)

    return flattened_list

