from typing import Union



def convert_ip_to_decimal_num_str(ip_str: str) -> str:

    """Converts a string representation of an IPv4 address into a decimal number string.



    Args:

        ip_str: A string representation of an IPv4 address.



    Returns:

        A string representation of the decimal number.

    """

    parts = ip_str.split('.')

    decimal_num = 0

    for i, part in enumerate(parts):

        decimal_num += int(part) << (8 * (len(parts) - i - 1))

    return str(decimal_num)

