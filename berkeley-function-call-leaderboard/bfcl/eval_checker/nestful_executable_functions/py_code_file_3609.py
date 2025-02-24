from typing import List



def double_all(numbers: List[float]) -> List[float]:

    """Doubles each number in a list and returns a new list with the doubled numbers.



    Args:

        numbers: A list of numbers to double.



    Returns:

        A new list with the numbers doubled.

    """

    return list(map(lambda x: x * 2, numbers))

