from typing import Union



def create_list_if_valid(x: Union[int, float, str]) -> list:

    """Creates a list containing `x` if `x` is a valid type (int, float, or string). Otherwise, it returns an empty list.



    Args:

        x: The value to check.



    Returns:

        A list containing `x` if `x` is a valid type (int, float, or string). Otherwise, an empty list.

    """

    if isinstance(x, (int, float, str)):

        return [x]

    else:

        return []

