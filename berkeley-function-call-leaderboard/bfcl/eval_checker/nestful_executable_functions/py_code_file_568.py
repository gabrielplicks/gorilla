from typing import List



def remove_none(input_list: List[object]) -> List[object]:

    """Filters out None objects from a list.



    Args:

        input_list: The input list to filter.



    Returns:

        A new list that only contains non-None objects.

    """

    return [item for item in input_list if item is not None]

