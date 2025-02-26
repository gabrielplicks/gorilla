from typing import Union



def convert_ip_to_decimal(ip_string: str) -> int:

    """Converts an IP address from a string representation to a decimal number representation.



    Args:

        ip_string: A string in the format "a.b.c.d", where a, b, c, and d are decimal numbers ranging from 0 to 255.



    Returns:

        An integer representation of the IP address.

    """

    parts = ip_string.split('.')

    if len(parts) != 4:

        raise ValueError('Invalid IP address')

    decimal_number = 0

    for part in parts:

        decimal_number = (decimal_number << 8) | int(part)



    return decimal_number

