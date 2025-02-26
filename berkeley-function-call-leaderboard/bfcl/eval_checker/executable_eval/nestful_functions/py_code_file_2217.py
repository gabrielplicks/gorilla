from ipaddress import IPv4Interface, IPv4Network

from typing import Union



def get_subnet_mask(ip: str, prefix_length: int) -> str:

    """Calculates the subnet mask for a given IP address and prefix length.



    Args:

        ip: The IP address string.

        prefix_length: The prefix length.



    Returns:

        The subnet mask as a string.

    """

    interface = IPv4Interface(ip)

    network = interface.network

    subnet_mask = IPv4Network(str(network)).netmask

    return str(subnet_mask)

