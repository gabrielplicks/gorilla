from typing import Union



def pad_number_to_length(number: Union[int, float], length: int) -> str:

    """Pads a number with leading zeros to reach a specified length.

    Args:

        number: The number to pad.

        length: The desired length of the padded number.

    """

    return f"{number:0{length}d}"

