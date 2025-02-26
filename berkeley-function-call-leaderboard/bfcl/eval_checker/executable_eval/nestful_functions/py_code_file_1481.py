from typing import List



def get_first_and_last_items(input_list: List[int]) -> List[int]:

    """Returns a sublist containing only the first and last two elements of the input list.



    Args:

        input_list: The input list or array.



    Returns:

        A sublist containing the first and last two elements of the input list.

        If the input list has fewer than four elements, returns an empty list.

    """

    if len(input_list) >= 4:

        return input_list[:2] + input_list[-2:]

    else:

        return []

