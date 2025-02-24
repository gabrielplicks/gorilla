from typing import Union



def convert_decimal_to_hex(decimal_number: Union[int, float]) -> str:

    """Converts a decimal number into a hexadecimal string.



    Args:

        decimal_number: The decimal number to be converted.



    Returns:

        The hexadecimal string representation of the input number.

    """

    return "{:x}".format(decimal_number)

