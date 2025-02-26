from typing import List, Dict, Any



def map_items_to_indices(input_list: List[List[Any]]) -> Dict[Any, int]:

    """Creates a dictionary mapping each list item to its index in the list.



    Args:

        input_list: A list of lists.



    Returns:

        A dictionary mapping each list item to its index in the list.

    """

    return {item: index for sublist in input_list for index, item in enumerate(sublist)}

