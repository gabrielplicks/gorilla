from typing import List



def multiples_of_n(n: int) -> List[int]:

    """

    Returns a list of all the multiples of `n` up to `200`.



    Args:

        n: The number whose multiples are to be found.



    Raises:

        TypeError: If `n` is not an integer.

    """

    if not isinstance(n, int):

        raise TypeError('n must be an integer')

    return [i * n for i in range(1, 201)]

