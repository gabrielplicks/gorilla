from typing import List

import socket



def get_ports(n: int) -> List[int]:

    """

    Returns a list of `n` available ports using lambda expressions and context managers.



    Args:

        n: The number of available ports to return.

    """

    with socket.socket() as s:

        next_available_ports = [lambda start_port: start_port + port for port in range(n)]

        smallest_available_port = min(map(lambda func: func(s.getsockname()[1]), next_available_ports))

        return [smallest_available_port + i for i in range(n)]

