from typing import Union



def hex_str(n: Union[int, float]) -> str:

    """Returns a hexadecimal string representation of the number.



    Args:

        n: The number to convert to hexadecimal.

    """

    return hex(int(n))

