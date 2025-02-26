from typing import List, Union



def first_items(input_list: List[List[Union[int, float, str]]]) -> List[Union[int, float, str]]:

    """

    Returns a list of the first items in each sublist of the input list.

    Args:

        input_list: A list of lists, where each sublist contains at least one item.

    Returns:

        A list of the first items in each sublist.

    """

    output_list = []

    for sublist in input_list:

        output_list.append(sublist[0])

    return output_list

