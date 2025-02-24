from typing import List



def flatten_list_with_type_validation(x: List[List]) -> List:

    """Flattens a list of lists, but only if all the elements in the top-level list are lists.

    If there is an element that is not a list, the function raises a TypeError exception.



    Args:

        x: A list of lists.



    Raises:

        TypeError: If any element in the top-level list is not a list.

    """

    flattened = []



    for element in x:

        if not isinstance(element, list):

            raise TypeError('Only lists can be flattened')

        flattened.extend(element)



    return flattened

