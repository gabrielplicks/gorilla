from typing import List, Tuple, Union, Dict



def create_dict_from_list_pairs(pairs: List[Union[Tuple[Any, Any], List[Any]]]) -> Dict[Any, Any]:

    """

    Creates a dictionary from a list, where each list entry should have two items.

    If any entry does not have two items, a ValueError exception is raised.

    Args:

        pairs: A list of tuples or lists containing key-value pairs.

    Raises:

        ValueError: If any entry does not have two items.

    """

    if not pairs:

        return {}



    for pair in pairs:

        if len(pair) != 2:

            raise ValueError("Invalid entry: expected two items, found {}".format(len(pair)))



    return dict(pairs)

