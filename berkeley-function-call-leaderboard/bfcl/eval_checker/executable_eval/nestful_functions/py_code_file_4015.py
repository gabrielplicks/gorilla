from typing import List



def sum_of_positive_squares(lst: List[int]) -> int:

    """Calculates the sum of the squares of all positive integers in the list.



    Args:

        lst: A list of integers.



    Returns:

        The sum of the squares of all positive integers in the list.

    """

    squares = [x**2 for x in lst if x > 0]

    filtered_squares = [x for x in squares if x > 0]

    return sum(filtered_squares)

