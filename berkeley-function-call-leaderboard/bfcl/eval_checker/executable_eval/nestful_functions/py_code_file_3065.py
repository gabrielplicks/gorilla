from typing import List



def check_ip_address(ip_address: str) -> bool:

    """Checks if a given string is a valid IP address or not.



    Args:

        ip_address: The string to check.



    Returns:

        True if the given string is a valid IP address, False otherwise.

    """

    parts: List[str] = ip_address.split('.')

    if len(parts) != 4:

        return False

    for part in parts:

        if not part.isdigit():

            return False

        num = int(part)

        if not (0 <= num <= 255):

            return False

    return True

