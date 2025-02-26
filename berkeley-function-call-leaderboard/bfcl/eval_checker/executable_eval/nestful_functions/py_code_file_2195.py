from typing import Union



def convert_ip_to_int(ip_address: str) -> int:

    """Converts an IP address string into an integer using bitwise operations and integer representation.

    Args:

        ip_address: A string representing a valid IP address.

    """

    octets = ip_address.split(".")

    result = 0

    for octet in octets:

        result = (result << 8) | int(octet)

    return result

