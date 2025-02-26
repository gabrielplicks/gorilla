from typing import List, Set



def find_smallest_missing_integer(values: List[int]) -> int:

    """Finds the smallest integer that is not present in the list.



    Args:

        values: A list of integers.



    Returns:

        The smallest integer that is not present in the list.

    """

    max_value = max(values)

    expected_integers = set(range(max_value + 1))

    actual_integers = set(values)

    missing_integers = expected_integers - actual_integers

    return min(missing_integers) if missing_integers else None

