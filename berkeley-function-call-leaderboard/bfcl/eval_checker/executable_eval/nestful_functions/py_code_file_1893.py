from typing import List, Tuple



def parse_addresses(addresses: str) -> List[Tuple[str, int]]:

    """Parses a string of addresses in the format "IP:port" and forms a list of tuples (IP, port).



    Args:

        addresses: A string of addresses in the format "IP:port" separated by commas.



    Returns:

        A list of tuples (IP, port), where IP is a string and port is an integer.

    """

    address_list = []

    for address in addresses.split(","):

        ip, port = address.split(":")

        address_list.append((ip, int(port)))

    return address_list

