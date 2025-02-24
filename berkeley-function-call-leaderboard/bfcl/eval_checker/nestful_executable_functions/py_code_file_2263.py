from typing import List



def flatten_extend(list_of_lists: List[List[int]]) -> List[int]:

    """Flattens a list of lists and returns a single list containing all the elements of the input list, in order.



    Args:

        list_of_lists: A list of lists.



    Returns:

        A single list containing all the elements of the input list, in order.

    """

    flattened_list = [element for sublist in list_of_lists for element in sublist]

    result_list = []

    for element in flattened_list:

        result_list.extend([element])



    return result_list

