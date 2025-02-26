from typing import List, Union



def flatten_unique(lst: List[Union[List, str]]) -> List[str]:

    """Flattens a list of lists containing mixed types and returns a list of unique string elements.



    Args:

        lst: A list of lists containing mixed types (string, int, float, and other types).



    Returns:

        A list of unique string elements.

    """

    unique_list = set()

    for item in lst:

        if isinstance(item, str):

            unique_list.add(item)

        elif isinstance(item, list):

            unique_list.update(flatten_unique(item))

    return list(unique_list)

