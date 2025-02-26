import struct

import socket



def decode_socket_address(address: bytes) -> tuple:

    """Decodes a byte string representing a socket address and port.

    Args:

        address: The byte string representing the socket address and port.

    Returns:

        A tuple containing the decoded address and port.

    """

    address_bytes, port_bytes = struct.unpack('>4s2s', address)

    ip_address = socket.inet_ntoa(address_bytes)

    port = struct.unpack('>H', port_bytes)[0]

    return ip_address, port

