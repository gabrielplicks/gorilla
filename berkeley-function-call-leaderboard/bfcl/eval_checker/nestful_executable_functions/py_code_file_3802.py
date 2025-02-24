from typing import List, Union



def flatten_nested_list(nested_list: List[Union[int, float, str, List]]) -> List:

    """Flattens a nested list into a single list containing all elements.



    Args:

        nested_list: A list containing nested lists.



    Raises:

        ValueError: If the input is not a list.

    """

    if not isinstance(nested_list, list):

        raise ValueError('Input must be a list')

    result = []

    for element in nested_list:

        if isinstance(element, list):

            result.extend(flatten_nested_list(element))

        else:

            result.append(element)

    return result

