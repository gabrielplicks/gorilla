from itertools import chain

from typing import List, Union



def flatten_and_ignore_nulls(nested_list: List[Union[List, Union[int, float, None]]]) -> List[Union[int, float]]:

    """Flattens a nested list structure and ignores null values.



    Args:

        nested_list: The nested list structure to flatten.



    Returns:

        A flattened list of non-null values.

    """

    flat_list = []

    for item in nested_list:

        if isinstance(item, list):

            flat_list.extend(flatten_and_ignore_nulls(item))

        elif item is not None:

            flat_list.append(item)

    return flat_list

