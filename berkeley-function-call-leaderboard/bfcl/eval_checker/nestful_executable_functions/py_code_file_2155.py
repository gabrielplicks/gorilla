from typing import List



def double_odd_elements(array: List[int]) -> List[int]:

    """Doubles the elements at odd indices in an array.



    Args:

        array: The input array.



    Returns:

        A new array in which the elements at odd indices are doubled.

    """

    new_array = [0] * len(array)

    for i, element in enumerate(array):

        if i % 2 == 1:

            new_array[i] = element * 2

        else:

            new_array[i] = element



    return new_array

