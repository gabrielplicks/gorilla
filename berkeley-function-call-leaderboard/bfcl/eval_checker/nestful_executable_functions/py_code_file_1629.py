from typing import Union



def to_int32(ip_string: str) -> Union[int, None]:

    """Converts an IPv4 string address to its 32-bit representation as an integer.



    Args:

        ip_string: The IPv4 string address.



    Returns:

        The 32-bit representation of the IPv4 address as an integer, or None if the input is invalid.

    """

    parts = ip_string.split(".")

    if len(parts) != 4:

        return None

    for part in parts:

        try:

            int_part = int(part)

        except ValueError:

            return None



        if int_part < 0 or int_part > 255:

            return None

    result = 0

    for part in parts:

        result = (result << 8) + int(part)



    return result

