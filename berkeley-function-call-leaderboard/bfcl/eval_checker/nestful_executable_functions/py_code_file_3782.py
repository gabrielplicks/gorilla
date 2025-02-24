from typing import List



def find_highest_value(array: List[float]) -> float:

    """

    Finds the highest value in a list.

    Args:

        array: A list of numbers.

    Raises:

        TypeError: If the input parameter is not a list.

    """

    if not isinstance(array, list):

        raise TypeError("Input parameter is not a list")

    if not array:

        print("Warning: The input list is empty.")

        return None

    highest = array[0]

    for element in array:

        highest = max(highest, element)



    return highest

