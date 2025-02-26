from typing import List



class EmptyListException(Exception):

    """Exception raised when the input list is empty."""

    pass



def linear_operator(input: List[float], operator: str) -> float:

    """Applies a linear operator to a list of numbers.



    Args:

        input: A single-dimensional Python list of numbers.

        operator: One of `'sum'`, `'mean'`, or `'max'`.



    Raises:

        EmptyListException: If the input list is empty.

    """

    if len(input) == 0:

        raise EmptyListException('The input list is empty.')



    result = 0

    if operator == 'sum':

        result = sum(input)

    elif operator == 'mean':

        result = sum(input) / len(input)

    elif operator == 'max':

        result = max(input)

    return result

