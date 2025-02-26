from typing import List



def get_integer_list(items: List[object]) -> List[int]:

    """Returns a list of indices of integers in the input list.



    Args:

        items: A list of items.



    Returns:

        A list of indices of integers in the input list.

    """

    if not isinstance(items, list):

        return []



    indices = []



    for index, item in enumerate(items):

        if isinstance(item, int):

            indices.append(index)



    return indices

