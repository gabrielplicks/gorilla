from typing import List



def get_evens(x: List[int]) -> List[int]:

    """Returns a list of even numbers from the input list `x`.



    Args:

        x: A list of integers.



    Returns:

        A list of even integers from the input list `x`.

    """

    return [num for num in x if num % 2 == 0]

