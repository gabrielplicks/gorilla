from typing import List, Tuple



def tuple_list_to_dict(tuple_list: List[Tuple[Any, Any]]) -> Dict[Any, Any]:

    """Converts a list of 2-element tuples into a dictionary.

    Args:

        tuple_list: A list of 2-element tuples.

    Returns:

        A dictionary where the first element of each tuple is the key and the second element is the value.

    """

    return {key: value for key, value in tuple_list}

