from typing import List, Tuple



def broken_sum(list_of_tuples: List[Tuple[int, int]]) -> int:

    """Calculates the sum of the numbers in a list of tuples.



    Args:

        list_of_tuples: A list of tuples where each tuple has two elements representing a number and its square.



    Returns:

        The sum of the numbers in the tuple.

    """

    return sum(map(lambda element: element[0], [element for element in list_of_tuples if len(element) == 2]))

