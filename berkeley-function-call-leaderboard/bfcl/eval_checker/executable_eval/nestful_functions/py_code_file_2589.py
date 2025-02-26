from typing import Dict



def get_squares_dict(n: int) -> Dict[int, int]:

    """

    Returns a dictionary with keys from 1 to `n` and values of the square of the key.



    Args:

        n: The upper limit for the keys in the dictionary.

    """

    squares: Dict[int, int] = dict()

    for i in range(1, n + 1):

        squares[i] = i * i

    return squares

