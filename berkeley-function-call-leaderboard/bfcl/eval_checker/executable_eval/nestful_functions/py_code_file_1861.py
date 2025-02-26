from typing import Union



def mock_or_stub(x: int, y: bool) -> Union[float, int, bool]:

    """

    Returns a single value of type integer, float, or boolean.

    If `x` is less than or equal to 0, the function returns True.

    If `x` is greater than 0, the function returns `y`.

    If `y` is True, the function returns 0.0 (a float), otherwise the function returns 1.0.



    Args:

        x: An integer.

        y: A boolean.

    """

    if x <= 0:

        return True

    elif x > 0 and y:

        return 0.0

    else:

        return 1.0

