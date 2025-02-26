from typing import List



def create_list_of_even_numbers(n: int) -> List[int]:

    """Creates a list of even numbers from 0 to n.



    Args:

        n: The upper limit of the range of even numbers.



    Raises:

        TypeError: If the input is not an integer.

        ValueError: If the input is a negative integer.

    """

    if not isinstance(n, int):

        raise TypeError('Input must be an integer.')

    if n < 0:

        raise ValueError('Input must be non-negative.')



    return [i for i in range(n + 1) if i % 2 == 0]

