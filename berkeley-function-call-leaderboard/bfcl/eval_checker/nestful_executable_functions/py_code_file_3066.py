from typing import List, Tuple



def square_tuples(numbers: List[int]) -> List[Tuple[int, int]]:

    """Creates a list of tuples where each tuple contains the original integer and its square.



    Args:

        numbers: A list of integers.



    Returns:

        A list of tuples. Each tuple contains the original integer and its square.

    """

    if not numbers:

        return []



    return [(number, number ** 2) for number in numbers]

