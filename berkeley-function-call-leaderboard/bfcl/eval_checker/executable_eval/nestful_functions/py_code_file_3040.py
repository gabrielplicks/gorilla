from typing import Union



def find_result(x: Union[int, float]) -> str:

    """

    Returns a string based on the value of the input integer `x`.



    Args:

        x: The input integer.



    Returns:

        A string based on the value of `x`.

    """

    if 0 <= x <= 2:

        return 'Success'

    elif 3 <= x <= 6:

        return 'Almost there!'

    elif x < 0 or x > 9:

        return 'Oh no!'

    else:

        return 'Keep trying!'

