from typing import List



def filter_and_sort_integers(integers: List[int]) -> List[int]:

    """Filters and sorts a list of integers.



    Args:

        integers: A list of integers.



    Returns:

        A new list of integers that contains only the even numbers in the original list,

        sorted in ascending order, with any boundary values removed.

    """

    filtered_integers = [x for x in integers if x not in [0, 1]]  # Remove boundary values

    even_integers = [x for x in filtered_integers if x % 2 == 0]  # Filter out even numbers

    sorted_integers = sorted(even_integers)  # Sort the resulting list in ascending order

    return sorted_integers

