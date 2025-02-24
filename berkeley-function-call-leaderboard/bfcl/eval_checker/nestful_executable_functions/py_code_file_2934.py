import re



def is_ipv6(ip_address: str) -> bool:

    """Checks if a given string is a valid IPv6 address.



    Args:

        ip_address: The IP address to check.



    Returns:

        True if the IP address is valid, False otherwise.

    """

    components = re.split(r'[:\.]', ip_address)

    if len(components) != 8:

        return False

    for component in components:

        if not re.match(r'^[0-9a-fA-F]{1,4}$', component):

            return False



    return True

