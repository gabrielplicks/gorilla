from typing import Union



def check_types(x: Union[int, float], y: Union[int, float]) -> bool:

    """Checks if both `x` and `y` are of type `int` or `float`.



    Args:

        x: The first input.

        y: The second input.



    Returns:

        True if both `x` and `y` are of type `int` or `float`, False otherwise.

    """

    try:

        if isinstance(x, (int, float)) and isinstance(y, (int, float)):

            return True

        else:

            return False

    except Exception as e:

        print(f"Error: {e}")

        return False

