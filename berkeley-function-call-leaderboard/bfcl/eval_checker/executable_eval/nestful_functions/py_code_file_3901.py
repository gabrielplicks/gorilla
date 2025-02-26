from typing import List, Tuple



def create_tuple_list(original_list: List[int]) -> List[Tuple[int, int]]:

    """Creates a list of tuples from a given list, where each tuple contains the value and its index.



    Args:

        original_list: The list of values to create tuples from.



    Returns:

        A list of tuples, where each tuple contains the value and its index.

    """

    return [(value, index) for index, value in enumerate(original_list)]

