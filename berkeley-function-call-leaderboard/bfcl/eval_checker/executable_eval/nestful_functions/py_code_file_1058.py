from typing import Union



def hexa2dec(hex_str: Union[str, bytes]) -> int:

    """Converts a hexadecimal string to a decimal number.



    Args:

        hex_str: The hexadecimal string to be converted.

    """

    hex_str = hex_str.strip("0x")

    return int(hex_str, base=16)

