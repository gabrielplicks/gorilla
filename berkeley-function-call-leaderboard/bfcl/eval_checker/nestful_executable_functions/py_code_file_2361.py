import socket

from typing import Tuple



def find_first_available_port(start: int, end: int) -> int:

    """Finds the first available port in a given range using the socket programming library.



    Args:

        start: The starting port number.

        end: The ending port number.



    Raises:

        Exception: If no available ports are found in the given range.

    """

    port = start

    while port <= end:

        try:

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

                s.bind(('localhost', port))

            return port

        except socket.error:

            port += 1

    raise Exception('No available ports in the given range')

