from typing import Union



def sum_if_not_none(*args: Union[int, float, None]) -> Union[int, float, None]:

    """Calculates the sum of all non-None arguments.

    Args:

        args: The arguments to sum.

    """

    if any(x is None for x in args):

        return None

    elif all(x is None for x in args):

        return None

    else:

        return sum(x for x in args if x is not None)

