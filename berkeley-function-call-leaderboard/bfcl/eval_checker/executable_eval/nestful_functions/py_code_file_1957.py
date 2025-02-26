import socket



def is_valid_ipv6_address(address: str) -> bool:

    """

    Checks whether a given IPv6 address is valid.

    Args:

        address: The IPv6 address to be checked.

    Returns:

        True if the address is valid, False otherwise.

    """

    try:

        socket.inet_pton(socket.AF_INET6, address)

        return True

    except OSError:

        return False

