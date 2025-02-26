from typing import Dict



def square_dictionary() -> Dict[int, int]:

    """Creates a dictionary with keys as the numbers 1 through 100 and values as the square of the keys.



    Returns:

        A dictionary with keys as the numbers 1 through 100 and values as the square of the keys.

    """

    squares = {}

    for number in range(1, 101):

        square = number ** 2

        squares[number] = square



    return squares

