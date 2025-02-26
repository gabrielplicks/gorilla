from typing import List



def smallest_absolute_exponent(list_of_numbers: List[float]) -> int:

    """

    Returns the index of the number in the input list that has the smallest absolute value after being exponentiated.



    Args:

        list_of_numbers: A list of numbers.

    """

    exponentiated = [abs(num**2) for num in list_of_numbers]

    smallest = min(exponentiated)

    return exponentiated.index(smallest)

