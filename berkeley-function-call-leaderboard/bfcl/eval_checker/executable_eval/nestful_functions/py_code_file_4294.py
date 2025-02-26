from typing import List



def get_positive_numbers(n: int) -> List[int]:

    """Generates a list of positive integers up to n.

    Args:

        n: The upper limit for the positive integers.

    """

    if n < 0:

        return []



    positive_numbers = []

    for i in range(1, n + 1):

        positive_numbers.append(i)



    return positive_numbers

