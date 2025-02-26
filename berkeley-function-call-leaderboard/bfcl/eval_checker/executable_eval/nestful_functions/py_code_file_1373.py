from typing import List, Tuple



def fruit_count(fruits: List[Tuple[str, int]]) -> List[int]:

    """Returns a list of integers containing the number of each fruit in the tuples.



    Args:

        fruits: A list of tuples, where each tuple contains a string and an integer.

    """

    return [count for fruit, count in fruits]

