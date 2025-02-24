from typing import Union



def foo(x: Union[int, float]) -> int:

    """Returns 100 if the input parameter is less than 100. If the input parameter is greater than or equal to 100, returns 200.



    Args:

        x: The input parameter.



    Returns:

        100 or 200.

    """

    return 100 if x < 100 else 200

