from typing import Union



def convert_decibel_to_linear(x_decibel: Union[int, float]) -> float:

    """

    Converts a decibel value to a linear value.



    Args:

        x_decibel: The decibel value to convert.



    Returns:

        The linear value.

    """

    x_linear = 10 ** (x_decibel / 20)

    return x_linear

