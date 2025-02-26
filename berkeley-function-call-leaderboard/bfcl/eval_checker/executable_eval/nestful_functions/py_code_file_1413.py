from typing import List, Dict, Any



def find_uniques(array: List[int]) -> int:

    """Finds the index of the first unique element in an array of integers.



    If no unique element exists, returns -1.



    Args:

        array: A list of integers.



    Returns:

        The index of the first unique element if one exists, otherwise -1.

    """

    if not array:

        return -1

    frequencies: Dict[int, int] = {}

    for num in array:

        if num in frequencies:

            frequencies[num] += 1

        else:

            frequencies[num] = 1

    for i, num in enumerate(array):

        if frequencies[num] == 1:

            return i

    return -1

