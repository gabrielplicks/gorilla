from typing import Tuple



def parse_ip_port_username(encoded_string: str) -> Tuple[str, str, str]:

    """Parses an encoded string and returns a tuple containing the IP address, port, and user name.



    Args:

        encoded_string: The encoded string in the format "ip:port:username".



    Returns:

        A tuple containing the IP address, port, and user name.

    """

    ip_address, port, username = encoded_string.split(":")

    return ip_address, port, username

