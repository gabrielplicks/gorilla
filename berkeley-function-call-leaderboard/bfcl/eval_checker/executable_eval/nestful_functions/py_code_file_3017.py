from typing import List, Union



def sum_numbers_as_float(numbers: List[Union[int, float, str]]) -> float:

    """Calculates the sum of all numbers in a sequence, dynamically casting each number to a float.

    Args:

        numbers: A sequence of numbers, which can be integers, floats, or strings.

    """

    total = 0

    for number in numbers:

        try:

            total += float(number)

        except (ValueError, TypeError):

            print(f"Error casting {number} to float. Skipping.")

    return total

