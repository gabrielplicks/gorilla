from typing import List, Union



def remove_smallest_largest(numbers: List[Union[int, float]]) -> List[Union[int, float]]:

    """Removes the smallest and largest numbers from a list of numbers.



    Args:

        numbers: A list of numbers.



    Raises:

        ValueError: If the input is not a list of numbers.

    """

    if not isinstance(numbers, list):

        raise ValueError("Input must be a list.")



    for num in numbers:

        if not isinstance(num, (int, float)):

            raise ValueError("Input list must contain only numbers.")

    sorted_numbers = sorted(numbers)

    new_list = sorted_numbers[1:-1]



    return new_list

