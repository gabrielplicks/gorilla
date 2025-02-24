from typing import List, Set



def smallest_unused_number(array: List[int]) -> int:

    """Finds the smallest unused number in an array.



    Args:

        array: A list of integers.



    Returns:

        The smallest unused number.

    """

    existing_numbers: Set[int] = set()

    for num in array:

        existing_numbers.add(num)



    smallest_unused: int = 1

    while smallest_unused in existing_numbers:

        smallest_unused += 1



    return smallest_unused

