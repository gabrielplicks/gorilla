from typing import List, Union



def dedup_in_nested_lists(nested_list: List[List[Union[int, float]]]) -> List[Union[int, float]]:

    """

    Takes a list of lists as input and returns a new list containing the unique elements from all the sublists.

    If an element appears in a sublist more than once, it should only appear once in the new list.

    This is done without using any built-in functions.



    Args:

        nested_list: A list of lists containing integers or floats.



    Returns:

        A list of unique elements from all the sublists.

    """

    unique_elements = set()

    for sublist in nested_list:

        for element in sublist:

            unique_elements.add(element)

    return list(unique_elements)

