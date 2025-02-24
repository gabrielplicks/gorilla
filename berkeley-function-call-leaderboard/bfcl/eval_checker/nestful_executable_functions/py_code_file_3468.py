from typing import List



def convert_to_single(elements: List[int]) -> List[List[int]]:

    """Converts a list of elements to a list containing a single element if the list contains only one or no elements, or returns the original list if the list contains more than one element.



    Args:

        elements: The list of elements.

    """

    if len(elements) <= 1:

        return elements

    return [elements]

