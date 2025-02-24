from typing import List, Tuple



def check_if_all_numbers(input_list: List[Tuple[int, float, complex]]) -> bool:

    """Checks if all numbers in a list of tuples are valid.



    Args:

        input_list: A list of tuples containing numbers.



    Returns:

        True if all numbers are valid, False otherwise.

    """

    if not isinstance(input_list, list):

        return False

    for tup in input_list:

        if not isinstance(tup, tuple):

            return False

        for num in tup:

            if not (isinstance(num, int) or isinstance(num, float) or isinstance(num, complex)):

                return False

    return True

