from typing import Tuple



def check_and_swap(input_tuple: Tuple[int, int]) -> Tuple[int, int]:

    """Checks if the first element of a tuple is greater than the second, and swaps them if true.



    Args:

        input_tuple: A tuple of two integers.



    Returns:

        A tuple of two integers.

    """

    a, b = input_tuple  # Tuple destructuring

    if a > b:  # Check if the first element is greater than the second

        a, b = b, a  # Variable reassignment

    return (a, b)  # Return the new tuple

