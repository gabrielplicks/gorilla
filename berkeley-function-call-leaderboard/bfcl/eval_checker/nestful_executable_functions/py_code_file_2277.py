from typing import List



def convert_and_filter(strings: List[str], n: int) -> List[int]:

    """Converts a list of strings to integers and filters out all values that are divisible by `n`.



    Args:

        strings: A list of strings to convert to integers.

        n: The number to filter out values that are divisible by.



    Returns:

        A list of integers that are divisible by `n`.

    """

    integers = list(map(int, strings))

    divisible_by_n = list(filter(lambda x: x % n == 0, integers))

    return divisible_by_n

