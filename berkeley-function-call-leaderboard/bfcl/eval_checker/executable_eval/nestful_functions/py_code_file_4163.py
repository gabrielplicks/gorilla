from typing import List



def is_monotonic(array: List[float]) -> bool:

    """Determines if an array is monotonic, meaning that the sequence of values is consistently increasing or decreasing (not both).



    Args:

        array: A list of numbers.



    Returns:

        True if the array is monotonic, False otherwise.

    """

    if len(array) <= 1:

        return True



    increase = array[0] < array[1]

    for i in range(1, len(array) - 1):

        if increase:

            if array[i] > array[i + 1]:

                return False

        else:

            if array[i] < array[i + 1]:

                return False

    return True

