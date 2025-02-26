from typing import Union



def is_positive_int(x: Union[int, float]) -> None:

    """Check if an input is a positive integer.



    Args:

        x: The input to check.



    Raises:

        ValueError: If the input is not an int or float type, or if its value is not greater than or equal to zero.

    """

    if not isinstance(x, (int, float)):

        raise ValueError(

            f"The input must be an int or float type, got {type(x)}."

        )

    if isinstance(x, float) and not x.is_integer():

        raise ValueError("The input must be an integer.")

    if x < 0:

        raise ValueError("The input must be greater than or equal to zero.")

