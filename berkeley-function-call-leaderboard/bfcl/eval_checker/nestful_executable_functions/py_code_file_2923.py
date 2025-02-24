from typing import Union



def add_one_if_int(x: Union[int, str]) -> Union[int, None]:

    """Adds one to the input `x` if `x` is an integer and returns the result.

    Otherwise, if `x` is not an integer, the function prints an error message

    and returns `None`.



    Args:

        x: The input value.



    Returns:

        The input value plus one if `x` is an integer, otherwise `None`.

    """

    if isinstance(x, int):

        return x + 1

    else:

        print("Error: Input must be an integer")

        return None

