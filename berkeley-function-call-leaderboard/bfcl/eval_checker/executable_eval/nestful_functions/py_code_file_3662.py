from typing import Dict



def keyword_dict(**kwargs: Dict[str, any]) -> Dict[str, any]:

    """

    Returns the arguments as a dictionary of keyword arguments.



    Args:

        kwargs: The keyword arguments to convert to a dictionary.



    Raises:

        TypeError: If a positional argument is passed.

    """

    if len(kwargs) == 0:

        return {}

    if len(kwargs) == 1:

        return kwargs

    if len(kwargs) > 1:

        return kwargs

    raise TypeError("Positional arguments are not allowed")

