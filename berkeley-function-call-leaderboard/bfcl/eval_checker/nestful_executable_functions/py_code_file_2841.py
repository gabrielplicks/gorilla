from typing import List



def top_three(numbers: List[int]) -> List[int]:

    """Returns the top 3 highest values from a list of integers.



    Args:

        numbers: A list of integers.



    Returns:

        A list containing the top 3 highest values from the input list.

        If the input list is shorter than 3, the entire list is returned.

    """

    sorted_numbers = sorted(numbers)

    top_values = []

    for i in range(len(sorted_numbers) - 1, -1, -1):

        if len(top_values) < 3:

            top_values.append(sorted_numbers[i])

        else:

            break

    return top_values

