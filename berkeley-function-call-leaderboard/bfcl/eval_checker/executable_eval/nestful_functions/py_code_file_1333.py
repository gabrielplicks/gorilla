from typing import List, Union



def flatten_2d_array_with_single_loop(array: List[Union[List, int]]) -> List[int]:

    """Flattens a 2D array into a 1D array using a single loop.



    Args:

        array: The 2D array to be flattened.



    Returns:

        A 1D array containing all elements from the input 2D array.

    """

    if not isinstance(array[0], list):

        return array

    result = []

    for element in array:

        if isinstance(element, list):

            result.extend(flatten_2d_array_with_single_loop(element))

        else:

            result.append(element)



    return result

