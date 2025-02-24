from typing import Union



def perform_double_dispatch(arg1: Union[str, int, float], arg2: Union[str, int, float]) -> Union[str, int, float]:

    """

    Performs double dispatch based on the data type of the first argument.



    Args:

        arg1: The first argument, which can be a string, integer, or float.

        arg2: The second argument, which can be a string, integer, or float.



    Returns:

        The sum of the two arguments.

    """

    if isinstance(arg1, str):

        return arg1 + str(arg2)

    elif isinstance(arg1, (int, float)):

        return arg1 + arg2

    else:

        raise TypeError("Invalid data type for arg1")

