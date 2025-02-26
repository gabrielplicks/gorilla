from typing import List



def check_ipv4(address: str) -> bool:

    """Check whether a given string is a valid IPv4 address or not.



    A valid IP address consists of four groups of digits ranging from 0 to 255,

    separated by dots, with no leading zeroes.



    Args:

        address: The string to check.



    Returns:

        True if the string is a valid IPv4 address, False otherwise.

    """

    groups: List[str] = address.split('.')

    if len(groups) != 4:

        return False

    for group in groups:

        if not group.isdigit():

            return False

        if int(group) < 0 or int(group) > 255:

            return False

        if group.startswith('0') and len(group) > 1:

            return False

    return True

