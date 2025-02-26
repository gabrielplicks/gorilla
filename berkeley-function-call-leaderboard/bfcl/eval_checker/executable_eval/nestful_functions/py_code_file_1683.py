from typing import Union



class SameInputError(Exception):

    """Exception raised when two arguments are the same."""

    pass



def my_max(x: int, y: int) -> Union[int, None]:

    """Returns the larger of two integers.



    Args:

        x: The first integer.

        y: The second integer.



    Raises:

        SameInputError: If the two arguments are the same.

    """

    if x == y:

        raise SameInputError('Two arguments cannot be the same')

    elif x > y:

        return x

    else:

        return y

