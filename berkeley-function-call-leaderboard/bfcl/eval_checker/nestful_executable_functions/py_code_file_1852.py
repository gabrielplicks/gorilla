from typing import Union



def decibel_to_linear(decibel: Union[int, float]) -> float:

    """Converts a decibel value to its corresponding linear value.



    Args:

        decibel: The decibel value to be converted.



    Returns:

        The corresponding linear value.

    """

    return 10 ** (decibel / 20)

