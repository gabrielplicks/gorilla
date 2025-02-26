from typing import Union



def get_sign(n: Union[int, float]) -> str:

    """Returns a string representing the sign of a number.

    Args:

        n: The number to check the sign of.

    """

    if n == 0:

        return 'zero'

    elif n < 0:

        return 'negative'

    elif n > 0:

        return 'positive'

    else:

        return 'error'

