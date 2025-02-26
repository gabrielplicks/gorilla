from typing import Tuple



def parse_IPv6_address(address: str) -> Tuple[int, int, int, int, int, int, int, int]:

    """Parses an IPv6 address and returns a tuple of eight integers representing the eight 16-bit chunks.



    Args:

        address: The IPv6 address as a string.



    Returns:

        A tuple of eight integers representing the eight 16-bit chunks.

    """

    chunks = address.split(':')

    return tuple(int(chunk, 16) for chunk in chunks)

