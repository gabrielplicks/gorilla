from typing import List, Tuple, Dict



def list_to_dict(tuple_list: List[Tuple[int, str]]) -> Dict[int, str]:

    """Creates a dictionary from a list of tuples.



    Args:

        tuple_list: A list of tuples where the first element is the key and the second element is the value.



    Returns:

        A dictionary where keys are the tuples' first elements and values are the tuples' second elements.

    """

    return dict(tuple_list)

