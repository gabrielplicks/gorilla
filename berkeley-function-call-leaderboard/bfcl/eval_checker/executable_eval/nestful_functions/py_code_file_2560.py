from typing import List, Set



def smallest_positive_integer(strings: List[str]) -> int:

    """Calculates the smallest positive integer that is not present in the input list of strings.



    Args:

        strings: A list of strings.



    Returns:

        The smallest positive integer that is not present in the input list of strings.

    """

    integers: Set[int] = set()

    for string in strings:

        try:

            integer = int(string)

            if integer > 0:

                integers.add(integer)

        except ValueError:

            pass

    i = 1

    while i in integers:

        i += 1



    return i

