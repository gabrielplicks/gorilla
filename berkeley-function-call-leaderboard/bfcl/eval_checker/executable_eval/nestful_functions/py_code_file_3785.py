import inspect

from typing import List



def get_parameters(func: callable) -> List[str]:

    """

    Returns a list of parameter names for a given function.



    Args:

        func: The function to get the parameter names for.



    Raises:

        TypeError: If the input is not a function.

    """

    if not inspect.isfunction(func):

        raise TypeError("Input is not a function.")

    params = inspect.signature(func).parameters.keys()

    return list(params)

