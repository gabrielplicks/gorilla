from typing import List



def convert_lists(input_list: List[List[int]]) -> List[int]:

    """Converts a list of lists into a flat list.



    Args:

        input_list: A list of lists.



    Returns:

        A flat list containing all the elements from the input list of lists.

    """

    return [element for sublist in input_list for element in sublist]

