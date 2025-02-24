from typing import List, Tuple



def traverse_and_find_max(list_of_numbers: List[float]) -> Tuple[float, int, List[float]]:

    """Traverses a list of numbers and finds the maximum value and its index.

    Returns the maximum value, the index of the maximum value, and the original list.



    Args:

        list_of_numbers: A list of numbers.



    Returns:

        A tuple containing the maximum value, the index of the maximum value, and the original list.

    """

    max_value = list_of_numbers[0]

    max_index = 0



    for i, num in enumerate(list_of_numbers):

        if num > max_value:

            max_value = num

            max_index = i



    return max_value, max_index, list_of_numbers

