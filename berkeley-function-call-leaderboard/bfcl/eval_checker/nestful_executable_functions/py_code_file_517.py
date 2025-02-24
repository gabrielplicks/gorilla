from typing import List



def is_ipv4(s: str) -> bool:

    """Validates a string `s` as an IPv4 address.



    Args:

        s: The string to validate as an IPv4 address.



    Returns:

        True if `s` is a valid IPv4 address, and False otherwise.

    """

    octets: List[str] = s.split('.')

    if len(octets) != 4:

        return False

    for octet in octets:

        try:

            intval = int(octet)

        except ValueError:

            return False

        if intval < 0 or intval > 255:

            return False

    return True

